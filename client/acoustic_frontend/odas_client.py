"""
ODAS Python 客户端模块

通过 Socket 接收 ODAS (Open embeddeD Audition System) 的输出:
- SST (Sound Source Tracking): 声源跟踪数据 (ID, 角度, 能量)
- SSS (Sound Source Separation): 分离后的音频流

ODAS 提供工业级的 SRP-PHAT 定位和卡尔曼滤波跟踪,
比基础的 GCC-PHAT 在混响环境下精度更高
"""

import socket
import json
import logging
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ODASSourceState(Enum):
    """ODAS 声源状态"""
    INACTIVE = 0    # 未激活
    ACTIVE = 1      # 活跃
    TRACKED = 2     # 跟踪中


@dataclass
class TrackedSource:
    """
    ODAS 跟踪的声源对象
    
    包含空间位置、能量和跟踪状态信息
    """
    id: int                          # 声源 ID (ODAS 分配)
    azimuth: float                   # 方位角 (度, 0-360)
    elevation: float = 0.0           # 仰角 (度, 通常为0)
    x: float = 0.0                   # 笛卡尔 X
    y: float = 0.0                   # 笛卡尔 Y
    z: float = 0.0                   # 笛卡尔 Z
    energy: float = 0.0              # 能量 (0-1)
    activity: float = 0.0            # 活跃度
    state: ODASSourceState = ODASSourceState.INACTIVE
    timestamp: float = 0.0           # 时间戳
    
    @classmethod
    def from_odas_json(cls, data: Dict, source_id: int) -> 'TrackedSource':
        """从 ODAS JSON 数据创建"""
        x = data.get('x', 0.0)
        y = data.get('y', 0.0)
        z = data.get('z', 0.0)
        
        # 计算方位角 (从笛卡尔坐标)
        azimuth = np.degrees(np.arctan2(y, x))
        if azimuth < 0:
            azimuth += 360
        
        # 计算仰角
        r_xy = np.sqrt(x**2 + y**2)
        elevation = np.degrees(np.arctan2(z, r_xy)) if r_xy > 0.001 else 0.0
        
        return cls(
            id=source_id,
            azimuth=azimuth,
            elevation=elevation,
            x=x, y=y, z=z,
            energy=data.get('E', 0.0),
            activity=data.get('activity', 0.0),
            state=ODASSourceState.ACTIVE if data.get('activity', 0) > 0.5 else ODASSourceState.INACTIVE,
            timestamp=time.time()
        )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "azimuth": round(self.azimuth, 1),
            "elevation": round(self.elevation, 1),
            "energy": round(self.energy, 4),
            "activity": round(self.activity, 4),
            "state": self.state.name,
        }


@dataclass
class SeparatedAudio:
    """分离后的音频数据"""
    source_id: int
    audio_data: np.ndarray
    timestamp: float
    doa: float = None            # 对应的DOA角度


class ODASClient:
    """
    ODAS Socket 客户端
    
    连接 ODAS 守护进程，接收多声源定位和分离数据
    
    使用示例:
    ```python
    client = ODASClient(sst_port=9000, sss_port=9001)
    client.start()
    
    # 获取跟踪的声源
    sources = client.get_tracked_sources()
    for src in sources:
        print(f"Source {src.id}: {src.azimuth}° (energy: {src.energy})")
    
    # 获取分离的音频
    audio_dict = client.get_separated_audio()
    for src_id, audio in audio_dict.items():
        process_audio(audio)
    
    client.stop()
    ```
    """
    
    def __init__(
        self,
        sst_host: str = "127.0.0.1",
        sst_port: int = 9000,
        sss_host: str = "127.0.0.1",
        sss_port: int = 9001,
        max_sources: int = 4,
    ):
        """
        初始化 ODAS 客户端
        
        Args:
            sst_host: SST 服务器地址
            sst_port: SST 端口 (跟踪数据)
            sss_host: SSS 服务器地址  
            sss_port: SSS 端口 (分离音频)
            max_sources: 最大声源数
        """
        self.sst_host = sst_host
        self.sst_port = sst_port
        self.sss_host = sss_host
        self.sss_port = sss_port
        self.max_sources = max_sources
        
        # Socket 连接
        self._sst_socket: Optional[socket.socket] = None
        self._sss_socket: Optional[socket.socket] = None
        
        # 数据缓存
        self._tracked_sources: Dict[int, TrackedSource] = {}
        self._separated_audio: Dict[int, deque] = {i: deque(maxlen=50) for i in range(max_sources)}
        
        # 线程
        self._sst_thread: Optional[threading.Thread] = None
        self._sss_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # 回调
        self._callbacks: Dict[str, List[Callable]] = {
            "on_source_active": [],
            "on_source_inactive": [],
            "on_doa_update": [],
        }
        
        # 统计
        self._frame_count = 0
        self._last_active_sources: set = set()
        
        logger.info(f"ODASClient initialized: SST={sst_host}:{sst_port}, SSS={sss_host}:{sss_port}")
    
    def start(self):
        """启动客户端，连接 ODAS 服务"""
        if self._running:
            logger.warning("ODASClient already running")
            return
        
        self._running = True
        
        # 启动 SST 接收线程
        self._sst_thread = threading.Thread(target=self._sst_receiver, daemon=True)
        self._sst_thread.start()
        
        # 启动 SSS (Pots) 接收线程
        self._sss_thread = threading.Thread(target=self._sss_receiver, daemon=True)
        self._sss_thread.start()
        
        logger.info("ODASClient started")
    
    def stop(self):
        """停止客户端"""
        self._running = False
        
        if self._sst_socket:
            try:
                self._sst_socket.close()
            except:
                pass
        
        if self._sss_socket:
            try:
                self._sss_socket.close()
            except:
                pass
        
        logger.info("ODASClient stopped")
    
    def _sst_receiver(self):
        """SST (跟踪) 数据接收线程 - 充当 TCP 服务器等待 ODAS 连接"""
        # 创建服务器 Socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.sst_host, self.sst_port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            logger.info(f"ODAS SST Server listening on {self.sst_host}:{self.sst_port}")
        except Exception as e:
            logger.error(f"Failed to bind SST server to {self.sst_host}:{self.sst_port}: {e}")
            return

        while self._running:
            try:
                # 等待 ODAS 连接
                try:
                    self._sst_socket, addr = server_socket.accept()
                    logger.info(f"ODAS SST connected from {addr}")
                except socket.timeout:
                    continue
                
                self._sst_socket.settimeout(1.0)
                buffer = ""
                brace_count = 0
                json_start = -1
                
                while self._running:
                    try:
                        data = self._sst_socket.recv(4096).decode('utf-8')
                        if not data:
                            logger.warning("ODAS SST connection closed by peer")
                            break
                        
                        buffer += data
                        
                        # 解析多行 JSON 对象 (ODAS 输出格式是多行的)
                        i = 0
                        while i < len(buffer):
                            c = buffer[i]
                            if c == '{':
                                if brace_count == 0:
                                    json_start = i
                                brace_count += 1
                            elif c == '}':
                                brace_count -= 1
                                if brace_count == 0 and json_start >= 0:
                                    # 找到完整的 JSON 对象
                                    json_str = buffer[json_start:i+1]
                                    self._parse_sst_json(json_str)
                                    buffer = buffer[i+1:]
                                    i = -1  # 重置索引
                                    json_start = -1
                            i += 1
                        
                        # 如果 buffer 太长但没有完整 JSON，清理开头的非 JSON 内容
                        if json_start > 0 and len(buffer) > 10000:
                            buffer = buffer[json_start:]
                            json_start = 0
                    
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.error(f"SST receive error: {e}")
                        break
                
                # 关闭当前连接，准备接受下一个
                if self._sst_socket:
                    self._sst_socket.close()
                    self._sst_socket = None
                    
            except Exception as e:
                if self._running:
                    logger.error(f"SST server error: {e}")
                    time.sleep(1)
        
        server_socket.close()
        logger.info("ODAS SST Server stopped")
    
    def _parse_sst_json(self, json_str: str):
        """解析 SST JSON 数据"""
        try:
            data = json.loads(json_str)
            
            # ODAS SST 输出格式: {"src": [{"x": ..., "y": ..., "activity": ...}, ...]}
            sources = data.get("src", [])
            
            current_active = set()
            
            with self._lock:
                for i, src_data in enumerate(sources):
                    if i >= self.max_sources:
                        break
                    
                    source = TrackedSource.from_odas_json(src_data, i)
                    
                    # 更新缓存
                    old_source = self._tracked_sources.get(i)
                    self._tracked_sources[i] = source
                    
                    # 检测状态变化
                    if source.activity > 0.5:
                        current_active.add(i)
                        
                        # 新激活的声源
                        if i not in self._last_active_sources:
                            self._emit("on_source_active", source)
                        
                        # DOA 变化
                        if old_source and abs(source.azimuth - old_source.azimuth) > 5:
                            self._emit("on_doa_update", source)
                
                # 检测失活的声源
                for src_id in self._last_active_sources - current_active:
                    if src_id in self._tracked_sources:
                        self._emit("on_source_inactive", self._tracked_sources[src_id])
                
                self._last_active_sources = current_active
            
            self._frame_count += 1
            
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON from SST: {e}")
    
    def _sss_receiver(self):
        """SSS (分离音频/Pots) 数据接收线程 - 充当 TCP 服务器等待 ODAS 连接"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.sss_host, self.sss_port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            logger.info(f"ODAS SSS/Pots Server listening on {self.sss_host}:{self.sss_port}")
        except Exception as e:
            logger.error(f"Failed to bind SSS server to {self.sss_host}:{self.sss_port}: {e}")
            return

        while self._running:
            try:
                try:
                    self._sss_socket, addr = server_socket.accept()
                    logger.info(f"ODAS SSS/Pots connected from {addr}")
                except socket.timeout:
                    continue
                
                self._sss_socket.settimeout(1.0)
                while self._running:
                    try:
                        data = self._sss_socket.recv(4096)
                        if not data:
                            logger.warning("ODAS SSS/Pots connection closed by peer")
                            break
                        # 目前仅丢弃数据，以防阻塞 ODAS
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.error(f"SSS receive error: {e}")
                        break
                
                if self._sss_socket:
                    self._sss_socket.close()
                    self._sss_socket = None
                    
            except Exception as e:
                if self._running:
                    logger.error(f"SSS server error: {e}")
                    time.sleep(1)
        
        server_socket.close()
        logger.info("ODAS SSS/Pots Server stopped")
    
    def get_tracked_sources(self, active_only: bool = True) -> List[TrackedSource]:
        """
        获取当前跟踪的声源列表
        
        Args:
            active_only: 是否只返回活跃的声源
            
        Returns:
            声源列表
        """
        with self._lock:
            sources = list(self._tracked_sources.values())
        
        if active_only:
            sources = [s for s in sources if s.activity > 0.1]
        
        return sorted(sources, key=lambda s: s.energy, reverse=True)
    
    def get_primary_doa(self) -> Optional[float]:
        """
        获取主声源的 DOA 角度
        
        Returns:
            主声源方位角，无声源时返回 None
        """
        sources = self.get_tracked_sources(active_only=True)
        if sources:
            return sources[0].azimuth
        return None
    
    def get_separated_audio(self, source_id: int = 0) -> Optional[np.ndarray]:
        """
        获取指定声源的分离音频
        
        Args:
            source_id: 声源 ID
            
        Returns:
            音频数据数组
        """
        with self._lock:
            if source_id in self._separated_audio and self._separated_audio[source_id]:
                return np.concatenate(list(self._separated_audio[source_id]))
        return None
    
    def on(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, data: Any):
        """触发事件"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def is_connected(self) -> bool:
        """检查是否已连接到 ODAS"""
        return self._sst_socket is not None and self._running
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "connected": self.is_connected(),
            "frame_count": self._frame_count,
            "active_sources": len(self._last_active_sources),
            "tracked_sources": len(self._tracked_sources),
        }


class ODASManager:
    """
    ODAS 进程管理器
    
    负责启动、监控和重启 ODAS 守护进程
    """
    
    def __init__(
        self,
        odas_binary: str = "odaslive",
        config_file: str = "respeaker_6mic.cfg",
    ):
        """
        初始化 ODAS 管理器
        
        Args:
            odas_binary: odaslive 可执行文件路径
            config_file: ODAS 配置文件路径
        """
        self.odas_binary = odas_binary
        self.config_file = config_file
        self._process = None
        self._running = False
    
    def start(self) -> bool:
        """启动 ODAS 进程"""
        import subprocess
        import shutil
        
        # 检查二进制文件是否存在
        if not shutil.which(self.odas_binary):
            logger.error(f"ODAS binary not found: {self.odas_binary}")
            return False
        
        try:
            cmd = [self.odas_binary, "-c", self.config_file]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._running = True
            logger.info(f"ODAS started: {' '.join(cmd)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ODAS: {e}")
            return False
    
    def stop(self):
        """停止 ODAS 进程"""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except:
                self._process.kill()
        self._running = False
        logger.info("ODAS stopped")
    
    def is_running(self) -> bool:
        """检查 ODAS 是否在运行"""
        if self._process:
            return self._process.poll() is None
        return False


# ============================================================
# 便捷函数
# ============================================================

def create_odas_client(
    sst_port: int = 9000,
    sss_port: int = 9001,
) -> ODASClient:
    """创建 ODAS 客户端的便捷函数"""
    return ODASClient(sst_port=sst_port, sss_port=sss_port)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    def on_source_active(source: TrackedSource):
        print(f"🎤 Source {source.id} active at {source.azimuth:.1f}°")
    
    def on_doa_update(source: TrackedSource):
        print(f"🎯 DOA updated: {source.azimuth:.1f}°")
    
    client = create_odas_client()
    client.on("on_source_active", on_source_active)
    client.on("on_doa_update", on_doa_update)
    
    print("Starting ODAS client...")
    print("Make sure 'odaslive' is running with the correct config")
    
    client.start()
    
    try:
        for i in range(100):
            time.sleep(0.5)
            sources = client.get_tracked_sources()
            if sources:
                print(f"Active sources: {[s.to_dict() for s in sources]}")
            print(f"Stats: {client.get_stats()}")
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
