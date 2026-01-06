"""
声学回声消除模块 (Acoustic Echo Cancellation, AEC)

利用 ReSpeaker 6-Mic 的回声参考通道 (通道6-7) 消除扬声器回声

原理:
1. TTS播放时的音频同时被回声参考通道捕获
2. 使用自适应滤波器估计回声路径
3. 从麦克风信号中减去估计的回声

ReSpeaker 6-Mic 的优势:
- 硬件级回声参考信号 (无需软件回环)
- 通道6-7直接连接DAC输出
- 低延迟同步采集
"""

import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AECConfig:
    """回声消除配置"""
    sample_rate: int = 16000
    
    # NLMS自适应滤波器参数
    filter_length: int = 256          # 滤波器长度 (采样点)
    step_size: float = 0.1            # 步长 (学习率)
    regularization: float = 1e-6      # 正则化项
    
    # 参考信号延迟补偿 (毫秒)
    reference_delay_ms: int = 0
    
    # 双讲检测参数
    doubletalk_threshold: float = 0.6
    
    # 残余回声抑制
    enable_residual_suppression: bool = True
    suppression_factor: float = 0.3


class AcousticEchoCanceller:
    """
    声学回声消除器
    
    使用 NLMS (Normalized Least Mean Squares) 自适应滤波算法
    
    使用示例:
    ```python
    aec = AcousticEchoCanceller(sample_rate=16000)
    
    # 在TTS播放期间设置参考信号
    aec.set_reference(tts_audio)
    
    # 处理麦克风输入
    # mic_audio: 包含回声的麦克风信号
    # echo_ref: 回声参考通道信号 (来自ReSpeaker通道6或7)
    clean_audio = aec.process(mic_audio, echo_ref)
    ```
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        filter_length: int = 256,
    ):
        """
        初始化回声消除器
        
        Args:
            sample_rate: 采样率
            filter_length: 自适应滤波器长度
        """
        self.config = AECConfig(
            sample_rate=sample_rate,
            filter_length=filter_length,
        )
        
        # NLMS自适应滤波器权重
        self._weights = np.zeros(filter_length, dtype=np.float32)
        
        # 参考信号缓冲
        self._ref_buffer = np.zeros(filter_length, dtype=np.float32)
        
        # 外部参考信号队列 (TTS播放时设置)
        self._external_ref = deque(maxlen=int(sample_rate * 2))  # 最多2秒
        
        # 状态
        self._is_playing = False          # 是否正在播放TTS
        self._adaptation_enabled = True   # 是否启用自适应
        
        # 性能统计
        self._erle_history = deque(maxlen=100)  # 回声抑制比历史
        
        logger.info(f"AEC initialized: filter_length={filter_length}, "
                   f"sample_rate={sample_rate}")
    
    def set_reference(self, audio: np.ndarray):
        """
        设置外部参考信号 (TTS输出)
        
        在播放TTS时调用，提供播放的音频作为回声预测的参考
        
        Args:
            audio: TTS输出音频
        """
        # 转换为float32并归一化
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0
        
        for sample in audio:
            self._external_ref.append(sample)
        
        self._is_playing = True
        logger.debug(f"Reference set: {len(audio)} samples")
    
    def clear_reference(self):
        """清除参考信号 (TTS播放结束时调用)"""
        self._external_ref.clear()
        self._is_playing = False
    
    def process(
        self,
        mic_signal: np.ndarray,
        echo_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        处理麦克风信号，消除回声
        
        Args:
            mic_signal: 麦克风信号 (包含回声)
            echo_ref: 回声参考信号 (来自ReSpeaker回声通道)
                     如果为None，使用外部设置的参考
                     
        Returns:
            消除回声后的干净语音
        """
        # 归一化
        mic_signal = mic_signal.astype(np.float32)
        if np.max(np.abs(mic_signal)) > 1.0:
            mic_signal = mic_signal / 32768.0
        
        # 确定参考信号
        if echo_ref is not None:
            # 使用硬件回声通道
            ref = echo_ref.astype(np.float32)
            if np.max(np.abs(ref)) > 1.0:
                ref = ref / 32768.0
        elif len(self._external_ref) > 0:
            # 使用外部设置的参考
            ref_len = min(len(mic_signal), len(self._external_ref))
            ref = np.array([self._external_ref.popleft() for _ in range(ref_len)])
            if len(ref) < len(mic_signal):
                ref = np.pad(ref, (0, len(mic_signal) - len(ref)))
        else:
            # 无参考信号，直接返回
            return mic_signal * 32768 if np.max(np.abs(mic_signal)) <= 1.0 else mic_signal
        
        # NLMS回声消除
        output = self._nlms_process(mic_signal, ref)
        
        # 残余回声抑制
        if self.config.enable_residual_suppression:
            output = self._residual_suppression(output, mic_signal, ref)
        
        # 反归一化
        return output * 32768
    
    def _nlms_process(
        self,
        mic: np.ndarray,
        ref: np.ndarray,
    ) -> np.ndarray:
        """
        NLMS自适应滤波
        
        Args:
            mic: 麦克风信号 (期望信号 d)
            ref: 参考信号 (输入信号 x)
            
        Returns:
            误差信号 e = d - y (即消除回声后的信号)
        """
        n_samples = len(mic)
        output = np.zeros(n_samples, dtype=np.float32)
        
        mu = self.config.step_size
        eps = self.config.regularization
        filter_len = self.config.filter_length
        
        for i in range(n_samples):
            # 更新参考缓冲
            self._ref_buffer = np.roll(self._ref_buffer, 1)
            self._ref_buffer[0] = ref[i]
            
            # 估计回声 y = w^T * x
            echo_estimate = np.dot(self._weights, self._ref_buffer)
            
            # 误差信号 e = d - y
            error = mic[i] - echo_estimate
            output[i] = error
            
            # 双讲检测
            if self._is_doubletalk(mic[i], error, ref[i]):
                continue  # 双讲时不更新权重
            
            # NLMS权重更新 w = w + mu * e * x / (||x||^2 + eps)
            if self._adaptation_enabled:
                power = np.dot(self._ref_buffer, self._ref_buffer) + eps
                update = (mu * error / power) * self._ref_buffer
                self._weights += update
        
        return output
    
    def _is_doubletalk(self, d: float, e: float, x: float) -> bool:
        """
        双讲检测
        
        当近端说话人和远端说话人同时发声时，
        应该暂停自适应以避免滤波器发散
        
        Args:
            d: 麦克风样本
            e: 误差样本  
            x: 参考样本
            
        Returns:
            True表示检测到双讲
        """
        # 简单的能量比较法
        threshold = self.config.doubletalk_threshold
        
        if abs(x) < 0.001:
            return False  # 没有远端信号
        
        ratio = abs(e) / (abs(d) + 1e-10)
        
        return ratio > threshold
    
    def _residual_suppression(
        self,
        output: np.ndarray,
        mic: np.ndarray,
        ref: np.ndarray,
    ) -> np.ndarray:
        """
        残余回声抑制
        
        使用频域方法进一步抑制残留回声
        """
        # 简单的谱减法
        ref_energy = np.sqrt(np.mean(ref ** 2))
        
        if ref_energy > 0.01:
            # 有明显参考信号时，温和地抑制
            suppression = 1.0 - self.config.suppression_factor * ref_energy
            suppression = max(0.3, suppression)
            output = output * suppression
        
        return output
    
    def get_erle(self) -> float:
        """
        获取回声抑制比 (ERLE)
        
        Returns:
            ERLE in dB (越高越好，典型值 10-30 dB)
        """
        if len(self._erle_history) == 0:
            return 0.0
        return np.mean(self._erle_history)
    
    def reset(self):
        """重置滤波器"""
        self._weights.fill(0)
        self._ref_buffer.fill(0)
        self._external_ref.clear()
        self._erle_history.clear()
        self._is_playing = False
    
    def enable_adaptation(self, enabled: bool = True):
        """启用/禁用自适应"""
        self._adaptation_enabled = enabled
    
    def get_stats(self) -> dict:
        """获取状态信息"""
        return {
            "filter_length": self.config.filter_length,
            "is_playing": self._is_playing,
            "adaptation_enabled": self._adaptation_enabled,
            "erle_db": self.get_erle(),
            "weights_norm": np.linalg.norm(self._weights),
            "ref_buffer_len": len(self._external_ref),
        }


class HardwareAEC:
    """
    硬件回声消除封装
    
    专门针对 ReSpeaker 6-Mic 的硬件回声通道优化
    利用通道6-7作为天然的回声参考
    """
    
    def __init__(self, sample_rate: int = 16000):
        self._aec = AcousticEchoCanceller(
            sample_rate=sample_rate,
            filter_length=512,  # 更长的滤波器处理房间混响
        )
        self._sample_rate = sample_rate
    
    def process_frame(
        self,
        mic_channels: np.ndarray,
        echo_channels: np.ndarray,
    ) -> np.ndarray:
        """
        处理一帧音频
        
        Args:
            mic_channels: 麦克风通道, shape=(samples, 6)
            echo_channels: 回声通道, shape=(samples, 2)
            
        Returns:
            消除回声后的音频, shape=(samples,)
        """
        # 使用第一个麦克风和第一个回声通道
        mic = mic_channels[:, 0]
        ref = echo_channels[:, 0]
        
        return self._aec.process(mic, ref)
    
    def set_playback_reference(self, audio: np.ndarray):
        """设置TTS播放参考 (可选，硬件通道更准确)"""
        self._aec.set_reference(audio)
    
    def reset(self):
        """重置"""
        self._aec.reset()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    aec = AcousticEchoCanceller(sample_rate=16000, filter_length=256)
    
    # 模拟测试
    print("Testing AEC with simulated signals...")
    
    duration = 0.5  # 500ms
    samples = int(16000 * duration)
    t = np.arange(samples) / 16000
    
    # 远端信号 (TTS播放)
    far_end = np.sin(2 * np.pi * 500 * t) * 0.3
    
    # 回声 (远端信号经过房间后) 
    echo = np.roll(far_end, 20) * 0.5
    echo += np.roll(far_end, 80) * 0.2  # 混响
    
    # 近端信号 (用户说话)
    near_end = np.sin(2 * np.pi * 300 * t) * 0.5
    near_end[samples//2:] *= 0.2  # 只说一半
    
    # 麦克风信号 = 近端 + 回声
    mic_signal = near_end + echo
    
    # 回声消除
    clean = aec.process(mic_signal * 32768, far_end * 32768)
    
    # 评估
    echo_energy = np.sqrt(np.mean(echo ** 2))
    mic_energy = np.sqrt(np.mean(mic_signal ** 2))
    clean_energy = np.sqrt(np.mean((clean / 32768) ** 2))
    near_energy = np.sqrt(np.mean(near_end ** 2))
    
    print(f"Echo energy: {echo_energy:.4f}")
    print(f"Mic (echo + near) energy: {mic_energy:.4f}")
    print(f"Clean output energy: {clean_energy:.4f}")
    print(f"Near-end energy: {near_energy:.4f}")
    print(f"AEC stats: {aec.get_stats()}")
