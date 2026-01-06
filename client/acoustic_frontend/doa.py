"""
DOA (Direction of Arrival) 声源定位模块

基于 GCC-PHAT (Generalized Cross-Correlation with Phase Transform) 算法
利用 ReSpeaker 6-Mic 环形阵列实现 360° 声源方向估计

原理:
1. 计算麦克风对之间的时延 (TDOA)
2. 根据时延和阵列几何关系估计声源方向
3. 多麦克风对结果融合提高精度

阵列布局 (俯视图):
            0° (Mic 0)
              ●
         ●         ●  
       Mic 5      Mic 1
                    
     ●               ● 
    Mic 4         Mic 2
         ●
        Mic 3
           180°
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 声速 (m/s)
SPEED_OF_SOUND = 343.0


@dataclass
class DOAConfig:
    """DOA估计器配置"""
    sample_rate: int = 16000
    
    # 阵列几何 (6麦克风环形排列)
    mic_angles_deg: List[float] = None    # 各麦克风角度位置
    array_radius_m: float = 0.035         # 阵列半径 (米)
    
    # 算法参数
    fft_size: int = 512                   # FFT窗口大小
    num_directions: int = 360             # 搜索方向数 (分辨率)
    
    # 麦克风对选择 (相对位置的麦克风对效果最好)
    mic_pairs: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.mic_angles_deg is None:
            self.mic_angles_deg = [0, 60, 120, 180, 240, 300]
        
        # 默认使用对角麦克风对 (0-3, 1-4, 2-5)
        if self.mic_pairs is None:
            self.mic_pairs = [(0, 3), (1, 4), (2, 5)]


class DOAEstimator:
    """
    DOA 声源定位估计器
    
    使用 GCC-PHAT 算法估计声源到达方向
    
    使用示例:
    ```python
    doa = DOAEstimator(sample_rate=16000)
    
    # mic_data shape: (samples, 6)
    angle, confidence = doa.estimate(mic_data)
    print(f"Sound from {angle}° with confidence {confidence:.2f}")
    ```
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mic_angles: List[float] = None,
        array_radius: float = 0.035,
    ):
        """
        初始化DOA估计器
        
        Args:
            sample_rate: 采样率
            mic_angles: 各麦克风的角度位置 (度)
            array_radius: 阵列半径 (米)
        """
        self.config = DOAConfig(
            sample_rate=sample_rate,
            mic_angles_deg=mic_angles or [0, 60, 120, 180, 240, 300],
            array_radius_m=array_radius,
        )
        
        # 预计算麦克风位置 (笛卡尔坐标)
        self._mic_positions = self._compute_mic_positions()
        
        # 预计算各方向的理论时延
        self._delays_table = self._precompute_delays()
        
        # 历史平滑
        self._angle_history = []
        self._history_size = 5
        
        logger.info(f"DOAEstimator initialized: {len(self.config.mic_angles_deg)} mics, "
                   f"radius={self.config.array_radius_m*100:.1f}cm")
    
    def _compute_mic_positions(self) -> np.ndarray:
        """计算麦克风笛卡尔坐标 (x, y)"""
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
            # 声源方向单位向量
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            for pair_idx, (mic_i, mic_j) in enumerate(self.config.mic_pairs):
                # 两麦克风位置差
                pos_diff = self._mic_positions[mic_i] - self._mic_positions[mic_j]
                # 理论时延 (秒)
                tau = np.dot(pos_diff, direction) / SPEED_OF_SOUND
                # 转换为采样点
                delays[dir_idx, pair_idx] = tau * self.config.sample_rate
        
        return delays
    
    def estimate(self, mic_data: np.ndarray) -> Tuple[float, float]:
        """
        估计声源方向
        
        Args:
            mic_data: 多通道麦克风数据, shape=(samples, 6)
            
        Returns:
            (angle, confidence): 声源方向角度(0-360)和置信度(0-1)
        """
        if mic_data.shape[1] < 6:
            logger.warning(f"Expected 6 channels, got {mic_data.shape[1]}")
            return 0.0, 0.0
        
        # 转换为float32
        mic_data = mic_data.astype(np.float32)
        
        # 检查是否有有效信号
        energy = np.sqrt(np.mean(mic_data ** 2))
        if energy < 100:  # 信号太弱
            return self._get_smoothed_angle(None), 0.0
        
        # 对各麦克风对计算 GCC-PHAT
        gcc_sum = np.zeros(self.config.num_directions)
        
        for pair_idx, (mic_i, mic_j) in enumerate(self.config.mic_pairs):
            tau, peak_value = self._gcc_phat(mic_data[:, mic_i], mic_data[:, mic_j])
            
            # 将测量的时延与理论时延对比，找到最匹配的方向
            for dir_idx in range(self.config.num_directions):
                expected_tau = self._delays_table[dir_idx, pair_idx]
                # 时延匹配得分 (高斯权重)
                error = abs(tau - expected_tau)
                score = np.exp(-error ** 2 / 2) * peak_value
                gcc_sum[dir_idx] += score
        
        # 找到最佳方向
        best_dir = np.argmax(gcc_sum)
        best_score = gcc_sum[best_dir]
        
        # 归一化置信度
        confidence = min(1.0, best_score / (len(self.config.mic_pairs) * 0.8))
        
        # 平滑处理
        smoothed_angle = self._get_smoothed_angle(best_dir)
        
        return smoothed_angle, confidence
    
    def _gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray) -> Tuple[float, float]:
        """
        GCC-PHAT 时延估计
        
        Args:
            sig1, sig2: 两个麦克风的信号
            
        Returns:
            (tau, peak_value): 时延(采样点)和互相关峰值
        """
        n = len(sig1) + len(sig2)
        
        # FFT
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        
        # 互功率谱
        R = SIG1 * np.conj(SIG2)
        
        # PHAT加权 (相位变换)
        magnitude = np.abs(R)
        magnitude[magnitude < 1e-6] = 1e-6  # 避免除零
        R_phat = R / magnitude
        
        # IFFT得到互相关
        cc = np.fft.irfft(R_phat, n=n)
        
        # 找到峰值
        max_shift = int(self.config.sample_rate * self.config.array_radius_m * 2 / SPEED_OF_SOUND) + 1
        
        # 只在有效范围内搜索
        cc_valid = np.concatenate([cc[-max_shift:], cc[:max_shift+1]])
        
        peak_idx = np.argmax(np.abs(cc_valid))
        peak_value = np.abs(cc_valid[peak_idx])
        
        # 转换为时延
        tau = peak_idx - max_shift
        
        return tau, peak_value
    
    def _get_smoothed_angle(self, new_angle: Optional[float]) -> float:
        """时间平滑，减少抖动"""
        if new_angle is not None:
            self._angle_history.append(new_angle)
            if len(self._angle_history) > self._history_size:
                self._angle_history.pop(0)
        
        if not self._angle_history:
            return 0.0
        
        # 使用圆形均值 (处理0/360度边界)
        angles_rad = np.radians(self._angle_history)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        mean_angle = np.degrees(np.arctan2(mean_sin, mean_cos))
        
        if mean_angle < 0:
            mean_angle += 360
        
        return mean_angle
    
    def get_direction_vector(self, angle_deg: float) -> Tuple[float, float]:
        """
        将角度转换为方向向量
        
        Args:
            angle_deg: 方向角度
            
        Returns:
            (x, y) 单位方向向量
        """
        angle_rad = np.radians(angle_deg)
        return np.cos(angle_rad), np.sin(angle_rad)
    
    def angle_to_mic_index(self, angle_deg: float) -> int:
        """
        找到最接近给定角度的麦克风
        
        Args:
            angle_deg: 目标角度
            
        Returns:
            麦克风索引 (0-5)
        """
        min_diff = float('inf')
        best_idx = 0
        
        for idx, mic_angle in enumerate(self.config.mic_angles_deg):
            diff = abs(angle_deg - mic_angle)
            diff = min(diff, 360 - diff)  # 处理环绕
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        
        return best_idx
    
    def reset(self):
        """重置历史状态"""
        self._angle_history = []


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    doa = DOAEstimator(sample_rate=16000)
    
    # 模拟从某个方向来的信号
    print("Testing DOA with simulated signals...")
    
    # 生成测试信号 (模拟90度方向的声源)
    duration = 0.1  # 100ms
    samples = int(16000 * duration)
    
    # 简单正弦波
    t = np.arange(samples) / 16000
    signal = np.sin(2 * np.pi * 1000 * t)  # 1kHz
    
    # 模拟6通道 (实际应该有时延差异)
    mic_data = np.column_stack([signal] * 6)
    
    angle, confidence = doa.estimate(mic_data)
    print(f"Estimated DOA: {angle:.1f}° (confidence: {confidence:.2f})")
