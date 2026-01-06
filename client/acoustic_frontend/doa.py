"""
增强的 DOA (Direction of Arrival) 声源定位模块

支持两种模式:
1. 内置 GCC-PHAT: 轻量级，无外部依赖
2. ODAS 集成: SRP-PHAT + 卡尔曼滤波跟踪 (精度更高)

在有 ODAS 运行时自动使用 ODAS 数据，否则降级到内置算法
"""

import numpy as np
import logging
import threading
import time
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# 声速 (m/s)
SPEED_OF_SOUND = 343.0


class DOABackend(Enum):
    """DOA 后端类型"""
    BUILTIN_GCC_PHAT = "gcc_phat"      # 内置 GCC-PHAT
    ODAS_SRP_PHAT = "odas"             # ODAS SRP-PHAT


@dataclass
class DOAConfig:
    """DOA估计器配置"""
    sample_rate: int = 16000
    
    # 阵列几何 (6麦克风环形排列)
    mic_angles_deg: List[float] = None
    array_radius_m: float = 0.0463     # ReSpeaker 6-Mic 半径
    
    # 算法参数
    fft_size: int = 512
    num_directions: int = 360
    
    # 麦克风对选择 (相对位置的麦克风对效果最好)
    mic_pairs: List[Tuple[int, int]] = None
    
    # 后端选择
    backend: DOABackend = DOABackend.BUILTIN_GCC_PHAT
    
    # ODAS 配置
    odas_sst_port: int = 9000
    odas_auto_fallback: bool = True    # ODAS 不可用时自动降级
    
    def __post_init__(self):
        if self.mic_angles_deg is None:
            self.mic_angles_deg = [0, 60, 120, 180, 240, 300]
        
        if self.mic_pairs is None:
            self.mic_pairs = [(0, 3), (1, 4), (2, 5)]


@dataclass 
class DOAResult:
    """DOA 估计结果"""
    angle: float                  # 方位角 (0-360度)
    confidence: float             # 置信度 (0-1)
    energy: float = 0.0           # 声源能量
    source_id: int = 0            # 声源 ID (ODAS 分配)
    backend: DOABackend = DOABackend.BUILTIN_GCC_PHAT
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        return {
            "angle": round(self.angle, 1),
            "confidence": round(self.confidence, 3),
            "energy": round(self.energy, 4),
            "source_id": self.source_id,
            "backend": self.backend.value,
        }


class EnhancedDOAEstimator:
    """
    增强的 DOA 声源定位估计器
    
    支持 GCC-PHAT 内置算法和 ODAS SRP-PHAT 外部引擎
    
    使用示例:
    ```python
    # 自动选择最佳后端
    doa = EnhancedDOAEstimator(backend="auto")
    
    # 强制使用 ODAS
    doa = EnhancedDOAEstimator(backend=DOABackend.ODAS_SRP_PHAT)
    
    # 估计声源方向
    result = doa.estimate(mic_data)
    print(f"Sound from {result.angle}° (backend: {result.backend})")
    ```
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mic_angles: List[float] = None,
        array_radius: float = 0.0463,
        backend: str = "auto",
    ):
        """
        初始化增强DOA估计器
        
        Args:
            sample_rate: 采样率
            mic_angles: 各麦克风的角度位置 (度)
            array_radius: 阵列半径 (米)
            backend: 后端选择 ("auto", "gcc_phat", "odas")
        """
        self.config = DOAConfig(
            sample_rate=sample_rate,
            mic_angles_deg=mic_angles or [0, 60, 120, 180, 240, 300],
            array_radius_m=array_radius,
        )
        
        # 后端选择
        self._backend = self._select_backend(backend)
        self._odas_client = None
        self._odas_available = False
        
        # 内置 GCC-PHAT 组件
        self._mic_positions = self._compute_mic_positions()
        self._delays_table = self._precompute_delays()
        
        # 历史平滑
        self._angle_history = []
        self._history_size = 5
        
        # 初始化 ODAS 客户端 (如果需要)
        if self._backend == DOABackend.ODAS_SRP_PHAT or backend == "auto":
            self._init_odas_client()
        
        logger.info(f"EnhancedDOAEstimator initialized: backend={self._backend.value}, "
                   f"odas_available={self._odas_available}")
    
    def _select_backend(self, backend_str: str) -> DOABackend:
        """选择后端"""
        if backend_str == "auto":
            return DOABackend.ODAS_SRP_PHAT  # 优先尝试 ODAS
        elif backend_str == "odas":
            return DOABackend.ODAS_SRP_PHAT
        else:
            return DOABackend.BUILTIN_GCC_PHAT
    
    def _init_odas_client(self):
        """初始化 ODAS 客户端"""
        try:
            from .odas_client import ODASClient
            
            self._odas_client = ODASClient(
                sst_port=self.config.odas_sst_port
            )
            self._odas_client.start()
            
            # 等待连接
            time.sleep(0.5)
            self._odas_available = self._odas_client.is_connected()
            
            if self._odas_available:
                logger.info("ODAS backend connected")
            else:
                logger.warning("ODAS not available, using fallback")
                
        except ImportError:
            logger.warning("ODAS client not found")
            self._odas_available = False
        except Exception as e:
            logger.warning(f"ODAS init failed: {e}")
            self._odas_available = False
    
    def _compute_mic_positions(self) -> np.ndarray:
        """计算麦克风笛卡尔坐标"""
        positions = []
        for angle_deg in self.config.mic_angles_deg:
            angle_rad = np.radians(angle_deg)
            x = self.config.array_radius_m * np.cos(angle_rad)
            y = self.config.array_radius_m * np.sin(angle_rad)
            positions.append([x, y])
        return np.array(positions)
    
    def _precompute_delays(self) -> np.ndarray:
        """预计算各方向各麦克风对的理论时延"""
        num_directions = self.config.num_directions
        num_pairs = len(self.config.mic_pairs)
        
        delays = np.zeros((num_directions, num_pairs))
        
        for dir_idx in range(num_directions):
            angle_rad = np.radians(dir_idx)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            for pair_idx, (mic_i, mic_j) in enumerate(self.config.mic_pairs):
                pos_diff = self._mic_positions[mic_i] - self._mic_positions[mic_j]
                tau = np.dot(pos_diff, direction) / SPEED_OF_SOUND
                delays[dir_idx, pair_idx] = tau * self.config.sample_rate
        
        return delays
    
    def estimate(self, mic_data: np.ndarray = None) -> DOAResult:
        """
        估计声源方向
        
        Args:
            mic_data: 多通道麦克风数据, shape=(samples, 6)
                     如果使用 ODAS 后端, 可以为 None
            
        Returns:
            DOAResult 对象
        """
        # 优先使用 ODAS
        if self._odas_available and self._odas_client:
            result = self._estimate_from_odas()
            if result is not None:
                return result
            
            # ODAS 无数据时降级
            if not self.config.odas_auto_fallback:
                return DOAResult(angle=0, confidence=0, backend=DOABackend.ODAS_SRP_PHAT)
        
        # 使用内置 GCC-PHAT
        if mic_data is None:
            return DOAResult(angle=0, confidence=0, backend=DOABackend.BUILTIN_GCC_PHAT)
        
        return self._estimate_gcc_phat(mic_data)
    
    def _estimate_from_odas(self) -> Optional[DOAResult]:
        """从 ODAS 获取 DOA 估计"""
        if not self._odas_client:
            return None
        
        sources = self._odas_client.get_tracked_sources(active_only=True)
        
        if not sources:
            return None
        
        # 返回能量最强的声源
        primary = sources[0]
        
        return DOAResult(
            angle=primary.azimuth,
            confidence=primary.activity,
            energy=primary.energy,
            source_id=primary.id,
            backend=DOABackend.ODAS_SRP_PHAT,
        )
    
    def _estimate_gcc_phat(self, mic_data: np.ndarray) -> DOAResult:
        """使用 GCC-PHAT 估计 DOA"""
        if mic_data.shape[1] < 6:
            logger.warning(f"Expected 6 channels, got {mic_data.shape[1]}")
            return DOAResult(angle=0, confidence=0, backend=DOABackend.BUILTIN_GCC_PHAT)
        
        mic_data = mic_data.astype(np.float32)
        
        # 检查信号能量
        energy = np.sqrt(np.mean(mic_data ** 2))
        if energy < 100:
            return DOAResult(
                angle=self._get_smoothed_angle(None),
                confidence=0,
                energy=energy / 32768,
                backend=DOABackend.BUILTIN_GCC_PHAT
            )
        
        # GCC-PHAT 计算
        gcc_sum = np.zeros(self.config.num_directions)
        
        for pair_idx, (mic_i, mic_j) in enumerate(self.config.mic_pairs):
            tau, peak_value = self._gcc_phat(mic_data[:, mic_i], mic_data[:, mic_j])
            
            for dir_idx in range(self.config.num_directions):
                expected_tau = self._delays_table[dir_idx, pair_idx]
                error = abs(tau - expected_tau)
                score = np.exp(-error ** 2 / 2) * peak_value
                gcc_sum[dir_idx] += score
        
        best_dir = np.argmax(gcc_sum)
        best_score = gcc_sum[best_dir]
        
        confidence = min(1.0, best_score / (len(self.config.mic_pairs) * 0.8))
        smoothed_angle = self._get_smoothed_angle(best_dir)
        
        return DOAResult(
            angle=smoothed_angle,
            confidence=confidence,
            energy=energy / 32768,
            backend=DOABackend.BUILTIN_GCC_PHAT,
        )
    
    def _gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray) -> Tuple[float, float]:
        """GCC-PHAT 时延估计"""
        n = len(sig1) + len(sig2)
        
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        
        R = SIG1 * np.conj(SIG2)
        magnitude = np.abs(R)
        magnitude[magnitude < 1e-6] = 1e-6
        R_phat = R / magnitude
        
        cc = np.fft.irfft(R_phat, n=n)
        
        max_shift = int(self.config.sample_rate * self.config.array_radius_m * 2 / SPEED_OF_SOUND) + 1
        cc_valid = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
        
        peak_idx = np.argmax(np.abs(cc_valid))
        peak_value = np.abs(cc_valid[peak_idx])
        tau = peak_idx - max_shift
        
        return tau, peak_value
    
    def _get_smoothed_angle(self, new_angle: Optional[float]) -> float:
        """时间平滑"""
        if new_angle is not None:
            self._angle_history.append(new_angle)
            if len(self._angle_history) > self._history_size:
                self._angle_history.pop(0)
        
        if not self._angle_history:
            return 0.0
        
        angles_rad = np.radians(self._angle_history)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        mean_angle = np.degrees(np.arctan2(mean_sin, mean_cos))
        
        if mean_angle < 0:
            mean_angle += 360
        
        return mean_angle
    
    def get_all_sources(self) -> List[DOAResult]:
        """
        获取所有检测到的声源
        
        仅在 ODAS 后端可用时返回多个声源
        """
        if self._odas_available and self._odas_client:
            sources = self._odas_client.get_tracked_sources(active_only=True)
            return [
                DOAResult(
                    angle=s.azimuth,
                    confidence=s.activity,
                    energy=s.energy,
                    source_id=s.id,
                    backend=DOABackend.ODAS_SRP_PHAT,
                )
                for s in sources
            ]
        return []
    
    def get_backend(self) -> DOABackend:
        """获取当前使用的后端"""
        if self._odas_available:
            return DOABackend.ODAS_SRP_PHAT
        return DOABackend.BUILTIN_GCC_PHAT
    
    def is_odas_available(self) -> bool:
        """检查 ODAS 是否可用"""
        return self._odas_available
    
    def reset(self):
        """重置历史状态"""
        self._angle_history = []
    
    def stop(self):
        """停止估计器"""
        if self._odas_client:
            self._odas_client.stop()


# ============================================================
# 兼容性别名
# ============================================================

# 保持与旧版 DOAEstimator 的兼容性
DOAEstimator = EnhancedDOAEstimator


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Enhanced DOA Estimator...")
    
    # 测试自动后端选择
    doa = EnhancedDOAEstimator(backend="auto")
    print(f"Backend: {doa.get_backend().value}")
    print(f"ODAS available: {doa.is_odas_available()}")
    
    # 模拟测试数据
    duration = 0.1
    samples = int(16000 * duration)
    t = np.arange(samples) / 16000
    signal = np.sin(2 * np.pi * 1000 * t)
    mic_data = np.column_stack([signal] * 6)
    
    result = doa.estimate(mic_data)
    print(f"Result: {result.to_dict()}")
    
    doa.stop()
