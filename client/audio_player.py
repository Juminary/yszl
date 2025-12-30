"""
音频播放模块
用于树莓派扬声器音频播放
"""

import pyaudio
import wave
import numpy as np
import logging
from pathlib import Path
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioPlayer:
    """音频播放类"""
    
    def __init__(self):
        """初始化音频播放器"""
        self.audio = pyaudio.PyAudio()
        logger.info("Audio player initialized")
    
    def list_devices(self):
        """列出所有音频输出设备"""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        devices = []
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxOutputChannels') > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxOutputChannels'),
                    'sample_rate': int(device_info.get('defaultSampleRate'))
                })
        
        return devices
    
    def play_file(self, audio_path: str, device_index: int = None):
        """
        播放音频文件
        
        Args:
            audio_path: 音频文件路径
            device_index: 输出设备索引
        """
        import subprocess
        import platform
        
        try:
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return
            
            logger.info(f"Playing audio: {audio_path}")
            
            # 在 macOS 上优先使用 afplay（更可靠）
            if platform.system() == 'Darwin':
                try:
                    result = subprocess.run(
                        ['afplay', str(audio_path)],
                        check=True,
                        capture_output=True,
                        timeout=60
                    )
                    logger.info("Playback completed (afplay)")
                    return
                except subprocess.CalledProcessError as e:
                    logger.warning(f"afplay failed: {e}, trying PyAudio")
                except FileNotFoundError:
                    logger.warning("afplay not found, trying PyAudio")
            
            # 使用soundfile读取（支持更多格式）
            try:
                audio_data, sample_rate = sf.read(audio_path, dtype='int16')
                
                # 处理立体声/单声道
                if len(audio_data.shape) > 1:
                    channels = audio_data.shape[1]
                else:
                    channels = 1
                    audio_data = audio_data.reshape(-1, 1)
                
                # 播放
                self._play_array(audio_data, sample_rate, channels, device_index)
                
            except Exception as e:
                # 如果soundfile失败，尝试使用wave
                logger.warning(f"soundfile failed, trying wave: {e}")
                self._play_wav(str(audio_path), device_index)
            
            logger.info("Playback completed")
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            raise
    
    def _play_wav(self, wav_path: str, device_index: int = None):
        """播放WAV文件（使用wave库）"""
        with wave.open(wav_path, 'rb') as wf:
            # 打开音频流
            stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=device_index
            )
            
            # 读取并播放数据
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            # 停止并关闭流
            stream.stop_stream()
            stream.close()
    
    def _play_array(self, audio_data: np.ndarray, sample_rate: int,
                   channels: int, device_index: int = None):
        """播放numpy数组"""
        # 确保数据类型正确
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # 打开音频流
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
            output_device_index=device_index
        )
        
        # 播放
        stream.write(audio_data.tobytes())
        
        # 停止并关闭流
        stream.stop_stream()
        stream.close()
    
    def play_array(self, audio_array: np.ndarray, sample_rate: int = 16000,
                   device_index: int = None):
        """
        播放音频数组
        
        Args:
            audio_array: 音频数据
            sample_rate: 采样率
            device_index: 输出设备索引
        """
        try:
            logger.info(f"Playing audio array: {len(audio_array)} samples @ {sample_rate}Hz")
            
            # 判断是单声道还是立体声
            if len(audio_array.shape) > 1:
                channels = audio_array.shape[1]
            else:
                channels = 1
                audio_array = audio_array.reshape(-1, 1)
            
            self._play_array(audio_array, sample_rate, channels, device_index)
            
            logger.info("Playback completed")
            
        except Exception as e:
            logger.error(f"Failed to play array: {e}")
            raise
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'audio'):
            self.audio.terminate()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    player = AudioPlayer()
    
    # 列出设备
    print("Available output devices:")
    for device in player.list_devices():
        print(f"  {device['index']}: {device['name']}")
    
    # 测试播放
    # player.play_file("test_audio.wav")
