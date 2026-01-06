"""
波束成形模块 (Beamformer)

基于延时求和 (Delay-and-Sum, DAS) 算法
利用 ReSpeaker 6-Mic 环形阵列增强目标方向的信号

原理:
1. 根据目标方向计算各麦克风的理论时延
2. 对各通道应用补偿延时
3. 求和得到增强信号

特点:
- 增强目标方向的信号 (约 6dB 增益)
- 抑制其他方向的干扰
- 可动态调整波束指向
"""

import numpy as np
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# 声速 (m/s)
SPEED_OF_SOUND = 343.0


@dataclass
class BeamformerConfig:
    """波束成形器配置"""
    sample_rate: int = 16000
    
    # 阵列几何
    mic_angles_deg: List[float] = None
    array_radius_m: float = 0.035
    
    # 算法参数
    interpolation_factor: int = 4     # 分数延时插值倍率
    use_weights: bool = True          # 是否使用窗函数加权
    
    def __post_init__(self):
        if self.mic_angles_deg is None:
            self.mic_angles_deg = [0, 60, 120, 180, 240, 300]


class Beamformer:
    """
    延时求和波束成形器 (Delay-and-Sum Beamformer)
    
    将6路麦克风信号进行时延补偿后求和，
    增强来自指定方向的声音信号。
    
    使用示例:
    ```python
    bf = Beamformer(sample_rate=16000)
    
    # mic_data shape: (samples, 6)
    # 增强90度方向的信号
    enhanced = bf.process(mic_data, target_angle=90.0)
    ```
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mic_angles: List[float] = None,
        array_radius: float = 0.035,
    ):
        """
        初始化波束成形器
        
        Args:
            sample_rate: 采样率
            mic_angles: 各麦克风的角度位置 (度)
            array_radius: 阵列半径 (米)
        """
        self.config = BeamformerConfig(
            sample_rate=sample_rate,
            mic_angles_deg=mic_angles or [0, 60, 120, 180, 240, 300],
            array_radius_m=array_radius,
        )
        
        # 麦克风位置 (笛卡尔坐标)
        self._mic_positions = self._compute_mic_positions()
        
        # 当前波束指向
        self._current_angle = 0.0
        self._current_delays = None
        
        # 上一帧的尾部 (用于跨帧连续性)
        self._last_tail = None
        
        logger.info(f"Beamformer initialized: {len(self.config.mic_angles_deg)} mics, "
                   f"radius={self.config.array_radius_m*100:.1f}cm")
    
    def _compute_mic_positions(self) -> np.ndarray:
        """计算麦克风笛卡尔坐标"""
        positions = []
        for angle_deg in self.config.mic_angles_deg:
            angle_rad = np.radians(angle_deg)
            x = self.config.array_radius_m * np.cos(angle_rad)
            y = self.config.array_radius_m * np.sin(angle_rad)
            positions.append([x, y])
        return np.array(positions)
    
    def _compute_delays(self, target_angle_deg: float) -> np.ndarray:
        """
        计算各麦克风相对于目标方向的时延
        
        Args:
            target_angle_deg: 目标方向 (度)
            
        Returns:
            各麦克风的时延 (采样点，浮点数)
        """
        angle_rad = np.radians(target_angle_deg)
        
        # 声源方向单位向量 (指向阵列中心)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        delays = []
        for pos in self._mic_positions:
            # 投影距离 (麦克风在声源方向上的投影)
            proj = np.dot(pos, direction)
            # 时延 (秒)
            tau = proj / SPEED_OF_SOUND
            # 转换为采样点
            delay_samples = tau * self.config.sample_rate
            delays.append(delay_samples)
        
        # 归一化: 使最小延时为0
        delays = np.array(delays)
        delays = delays - np.min(delays)
        
        return delays
    
    def steer(self, target_angle: float):
        """
        设置波束指向方向
        
        Args:
            target_angle: 目标方向 (度)
        """
        if abs(target_angle - self._current_angle) < 1.0:
            return  # 角度变化太小，不更新
        
        self._current_angle = target_angle
        self._current_delays = self._compute_delays(target_angle)
        logger.debug(f"Beam steered to {target_angle:.1f}°")
    
    def process(
        self, 
        mic_data: np.ndarray, 
        target_angle: Optional[float] = None
    ) -> np.ndarray:
        """
        执行波束成形
        
        Args:
            mic_data: 多通道麦克风数据, shape=(samples, 6)
            target_angle: 目标方向 (度)，如果为None则使用当前方向
            
        Returns:
            增强后的单通道音频, shape=(samples,)
        """
        if target_angle is not None:
            self.steer(target_angle)
        
        if self._current_delays is None:
            self._current_delays = self._compute_delays(self._current_angle)
        
        num_samples, num_channels = mic_data.shape
        
        # 转换为float32
        mic_data = mic_data.astype(np.float32)
        
        # 对各通道应用延时补偿并求和
        output = np.zeros(num_samples, dtype=np.float32)
        
        for ch in range(min(num_channels, len(self._current_delays))):
            delay = self._current_delays[ch]
            
            # 分数延时处理
            delayed = self._fractional_delay(mic_data[:, ch], delay)
            
            # 加权求和
            if self.config.use_weights:
                # 均匀加权
                weight = 1.0 / num_channels
            else:
                weight = 1.0
            
            output += delayed * weight
        
        # 归一化 (避免削波)
        max_val = np.max(np.abs(output))
        if max_val > 32767:
            output = output * (32767 / max_val)
        
        return output
    
    def _fractional_delay(self, signal: np.ndarray, delay: float) -> np.ndarray:
        """
        分数延时
        
        使用线性插值实现亚采样级延时
        
        Args:
            signal: 输入信号
            delay: 延时量 (采样点，可以是小数)
            
        Returns:
            延时后的信号
        """
        if abs(delay) < 0.01:
            return signal
        
        n = len(signal)
        
        # 整数部分和小数部分
        int_delay = int(np.floor(delay))
        frac_delay = delay - int_delay
        
        # 创建输出
        output = np.zeros_like(signal)
        
        if int_delay >= 0:
            # 正延时
            if int_delay < n:
                if frac_delay < 0.01:
                    # 无需插值
                    output[int_delay:] = signal[:n-int_delay]
                else:
                    # 线性插值
                    for i in range(int_delay + 1, n):
                        idx = i - int_delay
                        output[i] = (1 - frac_delay) * signal[idx] + frac_delay * signal[idx - 1]
        else:
            # 负延时 (提前)
            int_delay = -int_delay
            if int_delay < n:
                output[:n-int_delay] = signal[int_delay:]
        
        return output
    
    def process_adaptive(
        self,
        mic_data: np.ndarray,
        doa_angle: float,
        smoothing: float = 0.3,
    ) -> np.ndarray:
        """
        自适应波束成形 - 平滑跟踪声源
        
        Args:
            mic_data: 多通道麦克风数据
            doa_angle: DOA估计的声源方向
            smoothing: 平滑系数 (0-1)，越大越平滑
            
        Returns:
            增强后的音频
        """
        # 平滑角度变化
        if self._current_delays is None:
            target = doa_angle
        else:
            # 处理角度环绕
            diff = doa_angle - self._current_angle
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            target = self._current_angle + diff * (1 - smoothing)
            if target < 0:
                target += 360
            elif target >= 360:
                target -= 360
        
        return self.process(mic_data, target_angle=target)
    
    def get_beam_pattern(self, num_points: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前波束的方向图
        
        Args:
            num_points: 采样点数
            
        Returns:
            (angles, gains): 角度数组和对应的增益 (dB)
        """
        angles = np.linspace(0, 360, num_points, endpoint=False)
        gains = []
        
        for angle in angles:
            delays = self._compute_delays(angle)
            # 相对于当前指向的增益
            phase_diff = self._current_delays - delays if self._current_delays is not None else delays
            gain = np.abs(np.sum(np.exp(1j * 2 * np.pi * 1000 / self.config.sample_rate * phase_diff)))
            gains.append(gain)
        
        gains = np.array(gains)
        gains_db = 20 * np.log10(gains / np.max(gains) + 1e-10)
        
        return angles, gains_db
    
    def reset(self):
        """重置状态"""
        self._current_angle = 0.0
        self._current_delays = None
        self._last_tail = None


class MVDR_Beamformer(Beamformer):
    """
    MVDR (Minimum Variance Distortionless Response) 波束成形器
    
    相比DAS，MVDR能更好地抑制干扰，但计算量更大
    适用于多声源环境
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_cov = None
        self._adaptation_rate = 0.01
    
    def update_noise_covariance(self, noise_data: np.ndarray):
        """
        更新噪声协方差矩阵 (在静音期间调用)
        
        Args:
            noise_data: 噪声数据, shape=(samples, 6)
        """
        num_channels = noise_data.shape[1]
        
        # 估计协方差
        noise_cov = np.zeros((num_channels, num_channels), dtype=np.complex128)
        for i in range(num_channels):
            for j in range(num_channels):
                noise_cov[i, j] = np.mean(noise_data[:, i] * np.conj(noise_data[:, j]))
        
        # 自适应更新
        if self._noise_cov is None:
            self._noise_cov = noise_cov
        else:
            self._noise_cov = (1 - self._adaptation_rate) * self._noise_cov + \
                             self._adaptation_rate * noise_cov
    
    def process(
        self,
        mic_data: np.ndarray,
        target_angle: Optional[float] = None,
    ) -> np.ndarray:
        """
        MVDR波束成形 (如果没有噪声估计则降级为DAS)
        """
        if self._noise_cov is None:
            # 降级为DAS
            return super().process(mic_data, target_angle)
        
        # TODO: 实现完整MVDR
        # 目前先使用DAS
        return super().process(mic_data, target_angle)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    bf = Beamformer(sample_rate=16000)
    
    # 模拟测试
    print("Testing beamformer...")
    
    # 生成测试信号
    duration = 0.1  # 100ms
    samples = int(16000 * duration)
    t = np.arange(samples) / 16000
    
    # 模拟从90度方向来的信号 (各麦克风有微小延时差)
    base_signal = np.sin(2 * np.pi * 1000 * t)  # 1kHz
    
    mic_data = np.zeros((samples, 6), dtype=np.float32)
    delays = bf._compute_delays(90.0)
    
    for ch in range(6):
        delay = int(delays[ch])
        if delay < samples:
            mic_data[delay:, ch] = base_signal[:samples-delay]
    
    # 波束成形
    enhanced = bf.process(mic_data, target_angle=90.0)
    
    print(f"Input shape: {mic_data.shape}")
    print(f"Output shape: {enhanced.shape}")
    print(f"Input energy: {np.sqrt(np.mean(mic_data**2)):.2f}")
    print(f"Output energy: {np.sqrt(np.mean(enhanced**2)):.2f}")
    
    # 测试波束图
    angles, gains = bf.get_beam_pattern()
    main_lobe = angles[np.argmax(gains)]
    print(f"Main lobe at: {main_lobe:.1f}° (expected: 90°)")
