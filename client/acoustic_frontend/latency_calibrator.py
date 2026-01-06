"""
延迟校准模块 (Latency Calibrator)

AEC 性能与参考信号和麦克风信号的时间对齐密切相关。
此模块用于测量和校准系统的输入输出延迟。

方法:
1. 播放 Chirp 测试信号
2. 同时录制麦克风输入
3. 通过互相关计算延迟
4. 应用延迟补偿
"""

import numpy as np
import logging
import time
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """校准结果"""
    delay_samples: int           # 延迟 (采样点)
    delay_ms: float              # 延迟 (毫秒)
    confidence: float            # 置信度 (0-1)
    correlation_peak: float      # 相关峰值
    snr_db: float               # 信噪比
    success: bool               # 是否成功
    message: str = ""           # 描述信息
    
    def to_dict(self):
        return {
            "delay_samples": self.delay_samples,
            "delay_ms": round(self.delay_ms, 2),
            "confidence": round(self.confidence, 3),
            "snr_db": round(self.snr_db, 1),
            "success": self.success,
            "message": self.message,
        }


class LatencyCalibrator:
    """
    系统延迟校准器
    
    使用 Chirp 信号测量音频输入输出链路的总延迟
    
    使用示例:
    ```python
    calibrator = LatencyCalibrator(sample_rate=16000)
    
    # 获取测试信号
    test_signal = calibrator.generate_test_signal()
    
    # 播放 test_signal 并同时录制
    # recorded = play_and_record(test_signal)
    
    # 计算延迟
    result = calibrator.measure_delay(recorded)
    print(f"Delay: {result.delay_ms} ms")
    ```
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        test_duration: float = 0.5,
    ):
        """
        初始化校准器
        
        Args:
            sample_rate: 采样率
            test_duration: 测试信号时长 (秒)
        """
        self.sample_rate = sample_rate
        self.test_duration = test_duration
        
        # 生成测试信号
        self._test_signal = self._generate_chirp()
        
        logger.info(f"LatencyCalibrator initialized: {sample_rate}Hz, {test_duration}s")
    
    def _generate_chirp(self) -> np.ndarray:
        """
        生成 Chirp (线性扫频) 测试信号
        
        Chirp 信号的优势:
        - 宽带覆盖，更准确的延迟估计
        - 互相关峰值更尖锐
        - 对噪声鲁棒
        """
        n_samples = int(self.sample_rate * self.test_duration)
        t = np.arange(n_samples) / self.sample_rate
        
        # 频率从 200Hz 扫到 4000Hz
        f0 = 200
        f1 = 4000
        
        # 线性扫频
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * self.test_duration))
        chirp = np.sin(phase).astype(np.float32)
        
        # 窗函数 (避免边缘不连续)
        window = np.hanning(n_samples)
        chirp *= window
        
        # 归一化
        chirp = chirp / np.max(np.abs(chirp)) * 0.8
        
        return chirp
    
    def generate_test_signal(self, amplitude: float = 0.8) -> np.ndarray:
        """
        获取测试信号用于播放
        
        Args:
            amplitude: 振幅 (0-1)
            
        Returns:
            测试音频信号
        """
        # 前后添加静音
        silence = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
        signal = np.concatenate([silence, self._test_signal * amplitude, silence])
        
        return signal
    
    def measure_delay(
        self,
        recorded_signal: np.ndarray,
        max_delay_ms: float = 500,
    ) -> CalibrationResult:
        """
        从录制信号中测量延迟
        
        Args:
            recorded_signal: 播放测试信号时录制的音频
            max_delay_ms: 最大搜索延迟 (毫秒)
            
        Returns:
            CalibrationResult
        """
        # 归一化
        recorded = recorded_signal.astype(np.float32)
        if np.max(np.abs(recorded)) > 1.0:
            recorded = recorded / 32768.0
        
        # 信噪比估计
        snr = self._estimate_snr(recorded)
        
        # 计算互相关
        max_lag = int(max_delay_ms * self.sample_rate / 1000)
        
        try:
            delay, peak_value, confidence = self._cross_correlate(
                self._test_signal, recorded, max_lag
            )
        except Exception as e:
            logger.error(f"Cross-correlation failed: {e}")
            return CalibrationResult(
                delay_samples=0,
                delay_ms=0,
                confidence=0,
                correlation_peak=0,
                snr_db=snr,
                success=False,
                message=f"Correlation failed: {e}"
            )
        
        delay_ms = delay * 1000 / self.sample_rate
        
        # 判断是否成功
        success = confidence > 0.3 and 0 < delay < max_lag
        message = "OK" if success else "Low confidence or out of range"
        
        result = CalibrationResult(
            delay_samples=delay,
            delay_ms=delay_ms,
            confidence=confidence,
            correlation_peak=peak_value,
            snr_db=snr,
            success=success,
            message=message,
        )
        
        if success:
            logger.info(f"Calibration successful: {delay_ms:.1f}ms delay")
        else:
            logger.warning(f"Calibration uncertain: {result.message}")
        
        return result
    
    def _cross_correlate(
        self,
        reference: np.ndarray,
        recorded: np.ndarray,
        max_lag: int,
    ) -> Tuple[int, float, float]:
        """
        计算互相关并找到峰值
        
        Returns:
            (delay_samples, peak_value, confidence)
        """
        # 使用FFT加速互相关
        n = len(reference) + len(recorded) - 1
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        
        REF = np.fft.rfft(reference, n_fft)
        REC = np.fft.rfft(recorded, n_fft)
        
        # GCC-PHAT (相位变换)
        cross_power = REF * np.conj(REC)
        magnitude = np.abs(cross_power)
        magnitude[magnitude < 1e-10] = 1e-10
        
        # PHAT 加权
        cross_phat = cross_power / magnitude
        
        # 逆变换
        cc = np.fft.irfft(cross_phat, n_fft)
        
        # 只搜索正延迟 (录制应该晚于播放)
        cc_positive = cc[:max_lag]
        
        # 找峰值
        peak_idx = np.argmax(np.abs(cc_positive))
        peak_value = np.abs(cc_positive[peak_idx])
        
        # 置信度: 峰值与平均值的比率
        mean_value = np.mean(np.abs(cc_positive))
        confidence = min(1.0, (peak_value - mean_value) / (mean_value + 1e-10))
        
        return peak_idx, peak_value, confidence
    
    def _estimate_snr(self, signal: np.ndarray) -> float:
        """估计信噪比 (dB)"""
        # 简单方法: 比较信号段和静音段的能量
        n = len(signal)
        
        # 假设前后 10% 是静音
        noise_samples = int(n * 0.1)
        noise = np.concatenate([signal[:noise_samples], signal[-noise_samples:]])
        
        signal_energy = np.mean(signal ** 2)
        noise_energy = np.mean(noise ** 2) + 1e-10
        
        snr_db = 10 * np.log10(signal_energy / noise_energy)
        return snr_db


class AutoCalibrator:
    """
    自动校准器
    
    自动完成播放、录制、测量的完整流程
    需要提供 play 和 record 回调函数
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        play_callback: Callable[[np.ndarray], None] = None,
        record_callback: Callable[[float], np.ndarray] = None,
    ):
        """
        初始化自动校准器
        
        Args:
            sample_rate: 采样率
            play_callback: 播放函数 (audio) -> None
            record_callback: 录制函数 (duration) -> audio
        """
        self.calibrator = LatencyCalibrator(sample_rate=sample_rate)
        self.play = play_callback
        self.record = record_callback
        self._cached_delay: Optional[int] = None
    
    def run_calibration(self) -> CalibrationResult:
        """
        运行自动校准
        
        Returns:
            CalibrationResult
        """
        if self.play is None or self.record is None:
            return CalibrationResult(
                delay_samples=0, delay_ms=0, confidence=0,
                correlation_peak=0, snr_db=0, success=False,
                message="Play/record callbacks not set"
            )
        
        test_signal = self.calibrator.generate_test_signal()
        record_duration = len(test_signal) / self.calibrator.sample_rate + 0.5
        
        # 同时播放和录制
        # 注意: 实际实现需要线程同步
        import threading
        recorded = [None]
        
        def record_thread():
            recorded[0] = self.record(record_duration)
        
        rec_thread = threading.Thread(target=record_thread)
        rec_thread.start()
        
        time.sleep(0.1)  # 确保录制已开始
        self.play(test_signal)
        
        rec_thread.join()
        
        if recorded[0] is None:
            return CalibrationResult(
                delay_samples=0, delay_ms=0, confidence=0,
                correlation_peak=0, snr_db=0, success=False,
                message="Recording failed"
            )
        
        result = self.calibrator.measure_delay(recorded[0])
        
        if result.success:
            self._cached_delay = result.delay_samples
        
        return result
    
    def get_cached_delay(self) -> Optional[int]:
        """获取缓存的延迟值"""
        return self._cached_delay


def compensate_delay(
    signal: np.ndarray,
    reference: np.ndarray,
    delay_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    延迟补偿: 对齐信号和参考
    
    Args:
        signal: 麦克风信号
        reference: 参考信号
        delay_samples: 延迟采样点数 (正数表示信号滞后于参考)
        
    Returns:
        对齐后的 (signal, reference)
    """
    if delay_samples == 0:
        min_len = min(len(signal), len(reference))
        return signal[:min_len], reference[:min_len]
    
    if delay_samples > 0:
        # 信号滞后，裁剪信号前面，或补齐参考后面
        signal_aligned = signal[delay_samples:]
        reference_aligned = reference[:len(signal_aligned)]
    else:
        # 参考滞后
        delay_samples = abs(delay_samples)
        reference_aligned = reference[delay_samples:]
        signal_aligned = signal[:len(reference_aligned)]
    
    min_len = min(len(signal_aligned), len(reference_aligned))
    return signal_aligned[:min_len], reference_aligned[:min_len]


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Latency Calibrator...")
    
    calibrator = LatencyCalibrator(sample_rate=16000, test_duration=0.5)
    
    # 生成测试信号
    test_signal = calibrator.generate_test_signal()
    print(f"Test signal length: {len(test_signal)} samples")
    
    # 模拟延迟录制
    true_delay = 150  # 150 samples = 9.375ms
    recorded = np.roll(test_signal, true_delay) * 0.7
    recorded += np.random.randn(len(recorded)) * 0.05  # 添加噪声
    
    # 测量延迟
    result = calibrator.measure_delay(recorded)
    print(f"Result: {result.to_dict()}")
    print(f"True delay: {true_delay} samples, Measured: {result.delay_samples} samples")
    print(f"Error: {abs(true_delay - result.delay_samples)} samples")
