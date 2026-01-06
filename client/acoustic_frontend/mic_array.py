"""
麦克风阵列核心模块 (Microphone Array Core)

ReSpeaker 6-Mic Circular Array 的高级封装
整合多通道采集、DOA、波束成形和AEC为统一接口

硬件规格:
- 6个全向麦克风 (MSM321A3729H9CP, -22dBFS, SNR 59dB)
- 环形排列，直径约7cm
- 2个回声参考通道 (用于AEC)
- 采样率: 16kHz (支持8kHz-48kHz)
"""

import numpy as np
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class MicArrayState(Enum):
    """麦克风阵列状态"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class AudioFrame:
    """
    音频帧数据结构 - 包含所有声学信息
    
    这是声学前端的核心输出单元，整合了多通道音频、
    DOA估计、波束成形输出等信息
    """
    # 原始多通道数据
    raw_channels: np.ndarray      # shape: (samples, 8)
    timestamp: float              # 采集时间戳
    
    # 分离后的通道
    mic_channels: np.ndarray = None       # shape: (samples, 6) 麦克风
    echo_channels: np.ndarray = None      # shape: (samples, 2) 回声参考
    
    # 处理后输出
    enhanced_audio: np.ndarray = None     # shape: (samples,) 波束增强后
    clean_audio: np.ndarray = None        # shape: (samples,) AEC处理后
    
    # 声学特征
    doa_angle: float = None               # 声源方向 (0-360度)
    doa_confidence: float = 0.0           # DOA置信度
    energy: float = 0.0                   # 音频能量
    is_speech: bool = False               # VAD检测结果
    
    # 元数据
    sample_rate: int = 16000
    
    def get_mono(self) -> np.ndarray:
        """获取单通道音频 (优先返回处理后的音频)"""
        if self.clean_audio is not None:
            return self.clean_audio
        if self.enhanced_audio is not None:
            return self.enhanced_audio
        if self.mic_channels is not None:
            return self.mic_channels[:, 0]  # 返回第一个麦克风
        return self.raw_channels[:, 0]
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典 (用于事件传递)"""
        return {
            "timestamp": self.timestamp,
            "doa_angle": self.doa_angle,
            "doa_confidence": self.doa_confidence,
            "energy": self.energy,
            "is_speech": self.is_speech,
            "sample_rate": self.sample_rate,
            "samples": len(self.raw_channels) if self.raw_channels is not None else 0,
        }


@dataclass 
class MicArrayConfig:
    """麦克风阵列配置"""
    # 采样参数
    sample_rate: int = 16000
    chunk_duration_ms: int = 30       # 每帧时长(ms)
    
    # 通道配置 (ReSpeaker 6-Mic 固定配置)
    total_channels: int = 8
    mic_channel_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    echo_channel_indices: List[int] = field(default_factory=lambda: [6, 7])
    
    # 阵列几何 (6麦克风环形阵列，角度位置)
    mic_angles_deg: List[float] = field(default_factory=lambda: [0, 60, 120, 180, 240, 300])
    array_radius_m: float = 0.035     # 阵列半径 (约3.5cm)
    
    # 功能开关
    enable_doa: bool = True
    enable_beamforming: bool = True  
    enable_aec: bool = True
    enable_vad: bool = True
    
    # VAD参数
    vad_threshold: float = 0.02       # 能量阈值
    vad_min_speech_ms: int = 100      # 最小语音长度
    
    @property
    def chunk_samples(self) -> int:
        """每帧采样点数"""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


class MicrophoneArray:
    """
    麦克风阵列核心类
    
    整合 ReSpeaker 6-Mic 的所有声学处理功能:
    - 8通道音频采集 (6 mic + 2 echo)
    - 实时 DOA 声源定位
    - 波束成形信号增强
    - 回声消除 (AEC)
    - 简单VAD检测
    
    使用示例:
    ```python
    mic = MicrophoneArray()
    mic.start()
    
    while True:
        frame = mic.read()
        if frame:
            print(f"DOA: {frame.doa_angle}°, Energy: {frame.energy:.4f}")
            # 使用 frame.clean_audio 进行后续处理
    
    mic.stop()
    ```
    """
    
    def __init__(self, config: MicArrayConfig = None):
        self.config = config or MicArrayConfig()
        self.state = MicArrayState.IDLE
        
        # 内部组件 (延迟初始化)
        self._driver = None
        self._doa = None
        self._beamformer = None
        self._aec = None
        
        # 状态
        self._is_running = False
        self._lock = threading.Lock()
        
        # 回调
        self._callbacks: Dict[str, List[Callable]] = {
            "on_audio": [],        # 音频帧回调
            "on_doa_update": [],   # DOA更新回调
            "on_speech_start": [], # 语音开始
            "on_speech_end": [],   # 语音结束
        }
        
        # VAD状态
        self._speech_active = False
        self._speech_frames_count = 0
        
        # 统计信息
        self._frame_count = 0
        self._last_doa = None
        
        logger.info(f"MicrophoneArray initialized: {self.config.sample_rate}Hz, "
                   f"{self.config.chunk_duration_ms}ms chunks")
    
    def _init_components(self):
        """初始化内部组件"""
        # 导入驱动
        from .respeaker_driver import ReSpeakerDriver, ReSpeakerConfig
        
        driver_config = ReSpeakerConfig(
            sample_rate=self.config.sample_rate,
            channels=self.config.total_channels,
            chunk_size=self.config.chunk_samples,
        )
        self._driver = ReSpeakerDriver(driver_config)
        
        # DOA估计器
        if self.config.enable_doa:
            try:
                from .doa import DOAEstimator
                self._doa = DOAEstimator(
                    sample_rate=self.config.sample_rate,
                    mic_angles=self.config.mic_angles_deg,
                    array_radius=self.config.array_radius_m,
                )
                logger.info("DOA estimator enabled")
            except Exception as e:
                logger.warning(f"DOA initialization failed: {e}")
                self._doa = None
        
        # 波束成形器
        if self.config.enable_beamforming:
            try:
                from .beamformer import Beamformer
                self._beamformer = Beamformer(
                    sample_rate=self.config.sample_rate,
                    mic_angles=self.config.mic_angles_deg,
                    array_radius=self.config.array_radius_m,
                )
                logger.info("Beamformer enabled")
            except Exception as e:
                logger.warning(f"Beamformer initialization failed: {e}")
                self._beamformer = None
        
        # 回声消除器
        if self.config.enable_aec:
            try:
                from .aec import AcousticEchoCanceller
                self._aec = AcousticEchoCanceller(
                    sample_rate=self.config.sample_rate,
                )
                logger.info("AEC enabled")
            except Exception as e:
                logger.warning(f"AEC initialization failed: {e}")
                self._aec = None
    
    def start(self):
        """启动麦克风阵列"""
        with self._lock:
            if self._is_running:
                logger.warning("MicrophoneArray already running")
                return
            
            self._init_components()
            
            if self._driver:
                self._driver.start()
                self._is_running = True
                self.state = MicArrayState.LISTENING
                logger.info("MicrophoneArray started")
            else:
                self.state = MicArrayState.ERROR
                logger.error("Failed to start: driver not available")
    
    def stop(self):
        """停止麦克风阵列"""
        with self._lock:
            self._is_running = False
            if self._driver:
                self._driver.stop()
            self.state = MicArrayState.IDLE
            logger.info("MicrophoneArray stopped")
    
    def read(self, timeout: float = 1.0) -> Optional[AudioFrame]:
        """
        读取一帧处理后的音频
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            AudioFrame 对象，包含所有声学信息
        """
        if not self._is_running or not self._driver:
            return None
        
        # 从驱动读取原始音频
        raw_audio = self._driver.read(timeout=timeout)
        if raw_audio is None:
            return None
        
        # 创建音频帧
        frame = AudioFrame(
            raw_channels=raw_audio,
            timestamp=time.time(),
            sample_rate=self.config.sample_rate,
        )
        
        # 处理音频帧
        self._process_frame(frame)
        
        # 更新统计
        self._frame_count += 1
        
        # 触发回调
        self._emit("on_audio", frame)
        
        return frame
    
    def _process_frame(self, frame: AudioFrame):
        """处理音频帧 - 执行DOA、波束成形、AEC"""
        
        # 1. 分离通道
        frame.mic_channels = frame.raw_channels[:, self.config.mic_channel_indices]
        frame.echo_channels = frame.raw_channels[:, self.config.echo_channel_indices]
        
        # 2. 计算能量
        frame.energy = np.sqrt(np.mean(frame.mic_channels.astype(np.float32) ** 2))
        
        # 3. DOA估计
        if self._doa:
            doa_angle, confidence = self._doa.estimate(frame.mic_channels)
            frame.doa_angle = doa_angle
            frame.doa_confidence = confidence
            
            # DOA变化时触发回调
            if self._last_doa is None or abs(doa_angle - self._last_doa) > 10:
                self._emit("on_doa_update", frame)
                self._last_doa = doa_angle
        
        # 4. 波束成形 (指向DOA方向)
        if self._beamformer and frame.doa_angle is not None:
            frame.enhanced_audio = self._beamformer.process(
                frame.mic_channels, 
                target_angle=frame.doa_angle
            )
        else:
            # 降级: 使用第一个麦克风
            frame.enhanced_audio = frame.mic_channels[:, 0].astype(np.float32)
        
        # 5. 回声消除
        if self._aec:
            # 使用回声通道作为参考
            echo_ref = frame.echo_channels[:, 0].astype(np.float32)
            frame.clean_audio = self._aec.process(frame.enhanced_audio, echo_ref)
        else:
            frame.clean_audio = frame.enhanced_audio
        
        # 6. VAD检测
        if self.config.enable_vad:
            frame.is_speech = frame.energy > self.config.vad_threshold
            self._update_vad_state(frame)
    
    def _update_vad_state(self, frame: AudioFrame):
        """更新VAD状态机"""
        min_frames = int(self.config.vad_min_speech_ms / self.config.chunk_duration_ms)
        
        if frame.is_speech:
            self._speech_frames_count += 1
            if not self._speech_active and self._speech_frames_count >= min_frames:
                self._speech_active = True
                self._emit("on_speech_start", frame)
        else:
            if self._speech_active:
                self._speech_active = False
                self._speech_frames_count = 0
                self._emit("on_speech_end", frame)
            else:
                self._speech_frames_count = 0
    
    def on(self, event: str, callback: Callable):
        """
        注册事件回调
        
        Args:
            event: 事件名称 (on_audio, on_doa_update, on_speech_start, on_speech_end)
            callback: 回调函数，接收 AudioFrame 参数
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def off(self, event: str, callback: Callable):
        """取消事件回调"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    def _emit(self, event: str, data: Any):
        """触发事件"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def set_reference_audio(self, audio: np.ndarray):
        """
        设置AEC参考信号 (TTS播放时调用)
        
        Args:
            audio: TTS输出的音频数据
        """
        if self._aec:
            self._aec.set_reference(audio)
    
    def get_current_doa(self) -> Optional[float]:
        """获取当前声源方向"""
        return self._last_doa
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self.state.value,
            "is_running": self._is_running,
            "frame_count": self._frame_count,
            "current_doa": self._last_doa,
            "speech_active": self._speech_active,
            "components": {
                "doa": self._doa is not None,
                "beamformer": self._beamformer is not None,
                "aec": self._aec is not None,
            }
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# ============================================================
# 便捷函数
# ============================================================

def create_mic_array(
    sample_rate: int = 16000,
    enable_doa: bool = True,
    enable_beamforming: bool = True,
    enable_aec: bool = True,
) -> MicrophoneArray:
    """
    创建麦克风阵列实例的便捷函数
    
    Args:
        sample_rate: 采样率
        enable_doa: 是否启用DOA
        enable_beamforming: 是否启用波束成形
        enable_aec: 是否启用回声消除
        
    Returns:
        配置好的 MicrophoneArray 实例
    """
    config = MicArrayConfig(
        sample_rate=sample_rate,
        enable_doa=enable_doa,
        enable_beamforming=enable_beamforming,
        enable_aec=enable_aec,
    )
    return MicrophoneArray(config)


def create_from_config(config_path: str = None) -> MicrophoneArray:
    """
    从配置文件创建麦克风阵列
    
    Args:
        config_path: 配置文件路径，默认为 config/config.yaml
        
    Returns:
        配置好的 MicrophoneArray 实例
    """
    import os
    import yaml
    
    # 默认配置路径
    if config_path is None:
        # 尝试多个可能的路径
        possible_paths = [
            "config/config.yaml",
            "../config/config.yaml",
            "../../config/config.yaml",
            os.path.join(os.path.dirname(__file__), "../../config/config.yaml"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        logger.warning(f"Config file not found, using defaults")
        return create_mic_array()
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    
    af_config = full_config.get('acoustic_frontend', {})
    
    if not af_config.get('enabled', True):
        logger.info("acoustic_frontend disabled in config")
        return None
    
    # 解析配置
    device_cfg = af_config.get('device', {})
    array_cfg = af_config.get('array', {})
    doa_cfg = af_config.get('doa', {})
    bf_cfg = af_config.get('beamforming', {})
    aec_cfg = af_config.get('aec', {})
    vad_cfg = af_config.get('vad', {})
    
    # 构建 MicArrayConfig
    config = MicArrayConfig(
        sample_rate=device_cfg.get('sample_rate', 16000),
        chunk_duration_ms=device_cfg.get('chunk_duration_ms', 30),
        total_channels=device_cfg.get('total_channels', 8),
        mic_channel_indices=device_cfg.get('mic_channels', [0, 1, 2, 3, 4, 5]),
        echo_channel_indices=device_cfg.get('echo_channels', [6, 7]),
        mic_angles_deg=array_cfg.get('mic_angles_deg', [0, 60, 120, 180, 240, 300]),
        array_radius_m=array_cfg.get('radius_m', 0.035),
        enable_doa=doa_cfg.get('enabled', True),
        enable_beamforming=bf_cfg.get('enabled', True),
        enable_aec=aec_cfg.get('enabled', True),
        enable_vad=vad_cfg.get('enabled', True),
        vad_threshold=vad_cfg.get('energy_threshold', 0.02),
        vad_min_speech_ms=vad_cfg.get('min_speech_duration_ms', 100),
    )
    
    logger.info(f"Loaded acoustic_frontend config from {config_path}")
    return MicrophoneArray(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    def on_doa(frame: AudioFrame):
        print(f"DOA updated: {frame.doa_angle:.1f}° (confidence: {frame.doa_confidence:.2f})")
    
    def on_speech_start(frame: AudioFrame):
        print("🎤 Speech started")
    
    def on_speech_end(frame: AudioFrame):
        print("🔇 Speech ended")
    
    mic = create_mic_array()
    mic.on("on_doa_update", on_doa)
    mic.on("on_speech_start", on_speech_start)
    mic.on("on_speech_end", on_speech_end)
    
    with mic:
        print("Listening for 10 seconds...")
        print("Speak and move around to test DOA!")
        
        for i in range(int(10 * 1000 / 30)):  # 10秒
            frame = mic.read()
            if frame and i % 10 == 0:  # 每300ms打印一次
                print(f"Energy: {frame.energy:.4f}, DOA: {frame.doa_angle or 'N/A'}")
    
    print("Stats:", mic.get_stats())
