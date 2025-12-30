"""
音频采集模块
用于树莓派麦克风音频采集
"""

import pyaudio
import wave
import numpy as np
import logging
from pathlib import Path
import webrtcvad
from collections import deque
import time

logger = logging.getLogger(__name__)


class AudioCapture:
    """音频采集类"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_size: int = 1024, format=pyaudio.paInt16,
                 energy_threshold: int = 1500):  # 提高默认阈值，减少环境噪音干扰
        """
        初始化音频采集
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            chunk_size: 缓冲区大小
            format: 音频格式
            energy_threshold: 音量能量阈值（用于语音检测，可根据环境调整）
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        self.energy_threshold = energy_threshold  # 音量能量阈值
        
        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()
        
        # VAD（语音活动检测）- 模式3最激进，有效过滤背景噪音
        self.vad = webrtcvad.Vad(3)  # 0-3，3最激进，对非语音过滤最严格
        
        # 用于存储音频数据
        self.frames = []
        
        logger.info(f"Audio capture initialized: {sample_rate}Hz, {channels} channels, energy_threshold={energy_threshold}")
    
    def list_devices(self):
        """列出所有音频设备"""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        devices = []
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': int(device_info.get('defaultSampleRate'))
                })
        
        return devices
    
    def record(self, duration: float, output_path: str = None, 
               device_index: int = None) -> np.ndarray:
        """
        录制固定时长的音频
        
        Args:
            duration: 录制时长（秒）
            output_path: 保存路径（可选）
            device_index: 音频设备索引
        
        Returns:
            音频数据
        """
        try:
            logger.info(f"Recording for {duration} seconds...")
            
            # 打开音频流
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * duration)
            
            for i in range(num_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            # 停止并关闭流
            stream.stop_stream()
            stream.close()
            
            logger.info("Recording completed")
            
            # 转换为numpy数组
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 保存到文件（如果指定了路径）
            if output_path:
                self.save_wav(audio_array, output_path)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            raise
    
    def record_with_vad(self, max_duration: float = 30.0, 
                       silence_duration: float = 2.0,
                       output_path: str = None,
                       device_index: int = None) -> np.ndarray:
        """
        使用VAD自动检测语音并录制
        
        Args:
            max_duration: 最大录制时长
            silence_duration: 静音持续时长（触发停止）
            output_path: 保存路径
            device_index: 音频设备索引
        
        Returns:
            音频数据
        """
        try:
            logger.info("Recording with VAD...")
            
            # 打开音频流
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            # 先采样环境噪音来校准阈值（0.3秒）
            calibration_chunks = int(0.3 * self.sample_rate / self.chunk_size)
            noise_energies = []
            for _ in range(calibration_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                noise_energies.append(np.abs(audio_array).mean())
            
            # 设置动态阈值：环境噪音平均值的 4.5 倍，但不低于默认值
            avg_noise = np.mean(noise_energies) if noise_energies else 0
            dynamic_threshold = max(self.energy_threshold, int(avg_noise * 4.5))
            self.energy_threshold = dynamic_threshold
            logger.info(f"Calibrated energy threshold: {dynamic_threshold} (noise avg: {avg_noise:.1f})")
            
            frames = []
            speech_detected = False
            silence_chunks = 0
            silence_threshold = int(silence_duration * self.sample_rate / self.chunk_size)
            max_chunks = int(max_duration * self.sample_rate / self.chunk_size)
            
            logger.info("Listening for speech...")
            
            for i in range(max_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                
                # VAD检测（需要特定的帧大小，如320字节对应20ms@16kHz）
                # 调整chunk用于VAD检测
                is_speech = self._is_speech(data)
                
                if is_speech:
                    speech_detected = True
                    silence_chunks = 0
                    frames.append(data)
                    logger.debug("Speech detected")
                elif speech_detected:
                    frames.append(data)
                    silence_chunks += 1
                    
                    # 如果静音时间超过阈值，停止录制
                    if silence_chunks > silence_threshold:
                        logger.info("Silence detected, stopping recording")
                        break
            
            # 停止并关闭流
            stream.stop_stream()
            stream.close()
            
            if not frames:
                logger.warning("No speech detected")
                return np.array([], dtype=np.int16)
            
            logger.info(f"Recording completed: {len(frames)} chunks")
            
            # 转换为numpy数组
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 保存到文件
            if output_path:
                self.save_wav(audio_array, output_path)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"VAD recording failed: {e}")
            raise
    
    def _is_speech(self, frame: bytes) -> bool:
        """使用音量能量 + VAD 双重检测是否为语音"""
        try:
            # 将音频数据转换为numpy数组计算能量
            audio_array = np.frombuffer(frame, dtype=np.int16)
            energy = np.abs(audio_array).mean()
            
            # 能量阈值检测（可根据环境调整，默认 300-500）
            # 如果能量太低，直接判断为非语音
            if energy < self.energy_threshold:
                return False
            
            # VAD需要特定长度的帧（10, 20, 或 30ms）
            # 对于16kHz: 320字节(20ms), 160字节(10ms), 480字节(30ms)
            frame_length = len(frame)
            
            # 如果帧长度不匹配，进行调整
            if frame_length < 320:
                frame = frame + b'\x00' * (320 - frame_length)
            elif frame_length > 320:
                frame = frame[:320]
            
            # 使用 webrtcvad 确认是否是语音（而非单纯噪音）
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            # VAD失败时，仅使用能量阈值
            audio_array = np.frombuffer(frame, dtype=np.int16)
            energy = np.abs(audio_array).mean()
            return energy > self.energy_threshold
    
    def save_wav(self, audio_array: np.ndarray, output_path: str):
        """保存音频为WAV文件"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_array.tobytes())
            
            logger.info(f"Audio saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise
    
    def stream_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数"""
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_streaming(self, device_index: int = None):
        """开始流式录音"""
        self.frames = []
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.stream_callback
        )
        
        self.stream.start_stream()
        logger.info("Streaming started")
    
    def stop_streaming(self) -> np.ndarray:
        """停止流式录音并返回数据"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
            
            # 转换为numpy数组
            audio_data = b''.join(self.frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            logger.info("Streaming stopped")
            return audio_array
        
        return np.array([], dtype=np.int16)
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'audio'):
            self.audio.terminate()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    capture = AudioCapture()
    
    # 列出设备
    print("Available devices:")
    for device in capture.list_devices():
        print(f"  {device['index']}: {device['name']}")
    
    # 测试录音
    # audio = capture.record(duration=3.0, output_path="test_recording.wav")
    # print(f"Recorded {len(audio)} samples")
