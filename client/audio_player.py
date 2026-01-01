"""
音频播放模块
用于树莓派扬声器音频播放

增强功能:
- 支持打断(Barge-in)
- 事件总线集成
- TTS参考信号输出(用于AEC)
"""

import pyaudio
import wave
import numpy as np
import logging
from pathlib import Path
import soundfile as sf
import threading
from typing import Optional, Callable, Generator

# 尝试导入事件总线
try:
    from event_bus import EventBus, EventType, get_event_bus
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


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
    
    def create_streaming_player(self, sample_rate: int = 22050, channels: int = 1):
        """
        创建流式播放器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
            
        Returns:
            StreamingAudioPlayer 实例
        """
        return StreamingAudioPlayer(self.audio, sample_rate, channels)
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'audio'):
            self.audio.terminate()


class StreamingAudioPlayer:
    """
    流式音频播放器
    使用后台线程和队列实现边接收边播放
    """
    
    def __init__(self, audio: pyaudio.PyAudio, sample_rate: int = 22050, channels: int = 1):
        """
        初始化流式播放器
        
        Args:
            audio: PyAudio 实例
            sample_rate: 采样率
            channels: 声道数
        """
        import threading
        import queue
        
        self.audio = audio
        self.sample_rate = sample_rate
        self.channels = channels
        self.queue = queue.Queue()
        self.stream = None
        self.playing = False
        self.thread = None
        self._stop_event = threading.Event()
        self._started = False
        
        logger.info(f"[StreamingPlayer] Initialized: {sample_rate}Hz, {channels}ch")
    
    def start(self):
        """开始播放（创建后台播放线程）"""
        import threading
        
        if self._started:
            return
        
        self._started = True
        self._stop_event.clear()
        
        # 创建音频流
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=4096
        )
        
        # 启动播放线程
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playing = True
        self.thread.start()
        
        logger.info("[StreamingPlayer] Started")
    
    def _playback_loop(self):
        """后台播放循环"""
        while self.playing and not self._stop_event.is_set():
            try:
                # 从队列获取数据块（带超时）
                chunk = self.queue.get(timeout=0.1)
                
                if chunk is None:  # 结束信号
                    break
                
                # 播放数据块
                self.stream.write(chunk)
                
            except Exception:
                # 队列超时，继续等待
                continue
        
        logger.info("[StreamingPlayer] Playback loop ended")
    
    def feed(self, audio_bytes: bytes):
        """
        向播放器提供音频数据
        
        Args:
            audio_bytes: PCM 音频数据（16-bit）
        """
        if not self._started:
            self.start()
        
        self.queue.put(audio_bytes)
    
    def stop(self):
        """停止播放"""
        self.playing = False
        self._stop_event.set()
        
        # 发送结束信号
        self.queue.put(None)
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        
        logger.info("[StreamingPlayer] Stopped")
    
    def wait_until_done(self):
        """等待所有音频播放完成"""
        # 发送结束信号
        self.queue.put(None)
        
        # 等待队列清空
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=60.0)
        
        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        
        self.playing = False
        logger.info("[StreamingPlayer] Playback completed")


logger = logging.getLogger(__name__)


class InterruptibleAudioPlayer:
    """
    可中断音频播放器
    
    支持:
    - 打断播放(Barge-in)
    - 事件总线集成
    - TTS参考信号回调(用于AEC)
    """
    
    def __init__(
        self,
        event_bus: Optional['EventBus'] = None,
        sample_rate: int = 22050,
        channels: int = 1,
        chunk_size: int = 2048
    ):
        """
        初始化可中断播放器
        
        Args:
            event_bus: 事件总线实例
            sample_rate: 默认采样率
            channels: 默认声道数
            chunk_size: 播放块大小
        """
        self.event_bus = event_bus
        if event_bus is None and EVENT_BUS_AVAILABLE:
            self.event_bus = get_event_bus()
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        self.audio = pyaudio.PyAudio()
        self._playing = False
        self._interrupted = False
        self._play_lock = threading.Lock()
        self._current_text = ""  # 当前播放的文本（用于日志）
        
        # 回调函数
        self._on_reference_chunk: Optional[Callable] = None
        self._on_playback_start: Optional[Callable] = None
        self._on_playback_end: Optional[Callable] = None
        
        # 订阅打断事件
        if self.event_bus and EVENT_BUS_AVAILABLE:
            self.event_bus.subscribe(EventType.BARGE_IN, self._on_barge_in)
        
        logger.info(f"InterruptibleAudioPlayer initialized: {sample_rate}Hz, chunk_size={chunk_size}")
    
    def _on_barge_in(self, event):
        """处理打断事件"""
        if self._playing:
            self._interrupted = True
            logger.info(f"Playback interrupted by barge-in (was playing: {self._current_text[:30]}...)")
    
    @property
    def is_playing(self) -> bool:
        """是否正在播放"""
        return self._playing
    
    @property
    def was_interrupted(self) -> bool:
        """上次播放是否被中断"""
        return self._interrupted
    
    def play(
        self,
        audio_data: np.ndarray,
        sample_rate: int = None,
        text: str = "",
        emit_events: bool = True
    ) -> bool:
        """
        播放音频，支持打断
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率（默认使用初始化时的设置）
            text: 对应的文本（用于日志）
            emit_events: 是否发布事件
            
        Returns:
            bool: True=播放完成, False=被中断
        """
        sample_rate = sample_rate or self.sample_rate
        
        with self._play_lock:
            self._playing = True
            self._interrupted = False
            self._current_text = text
            
            # 发布TTS开始事件
            if emit_events and self.event_bus and EVENT_BUS_AVAILABLE:
                self.event_bus.emit(
                    EventType.TTS_START,
                    data={"text": text[:100]},
                    source="interruptible_player"
                )
            
            # 回调
            if self._on_playback_start:
                try:
                    self._on_playback_start(text)
                except Exception as e:
                    logger.error(f"Playback start callback error: {e}")
            
            try:
                # 确保数据格式正确
                if audio_data.dtype != np.int16:
                    if np.abs(audio_data).max() <= 1.0:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_data = audio_data.astype(np.int16)
                
                # 打开音频流
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=self.chunk_size
                )
                
                # 分块播放
                total_samples = len(audio_data)
                played_samples = 0
                
                for i in range(0, total_samples, self.chunk_size):
                    # 检查是否被打断
                    if self._interrupted:
                        logger.info(f"Playback interrupted at {played_samples}/{total_samples} samples")
                        break
                    
                    chunk = audio_data[i:i + self.chunk_size]
                    
                    # 回调参考信号（用于AEC）
                    if self._on_reference_chunk:
                        try:
                            self._on_reference_chunk(chunk)
                        except Exception:
                            pass
                    
                    stream.write(chunk.tobytes())
                    played_samples += len(chunk)
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Playback error: {e}")
            finally:
                self._playing = False
                
                # 发布TTS结束事件
                if emit_events and self.event_bus and EVENT_BUS_AVAILABLE:
                    if self._interrupted:
                        self.event_bus.emit(
                            EventType.TTS_INTERRUPTED,
                            data={"text": text[:100]},
                            source="interruptible_player"
                        )
                    else:
                        self.event_bus.emit(
                            EventType.TTS_END,
                            data={"text": text[:100]},
                            source="interruptible_player"
                        )
                
                # 回调
                if self._on_playback_end:
                    try:
                        self._on_playback_end(self._interrupted)
                    except Exception as e:
                        logger.error(f"Playback end callback error: {e}")
            
            return not self._interrupted
    
    def play_file(self, audio_path: str, text: str = "") -> bool:
        """
        播放音频文件，支持打断
        
        Args:
            audio_path: 音频文件路径
            text: 对应的文本（用于日志）
            
        Returns:
            bool: True=播放完成, False=被中断
        """
        try:
            audio_data, sample_rate = sf.read(audio_path, dtype='int16')
            
            # 处理立体声
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # 取第一个通道
            
            return self.play(audio_data, sample_rate, text)
            
        except Exception as e:
            logger.error(f"Failed to play file {audio_path}: {e}")
            return False
    
    def stop(self):
        """停止播放"""
        self._interrupted = True
    
    def play_stream(
        self,
        audio_stream,
        sample_rate: int = 22050,
        text: str = "",
        emit_events: bool = True
    ) -> bool:
        """
        流式播放音频，边接收边播放
        
        Args:
            audio_stream: 可迭代的音频数据流（每次yield一块bytes）
            sample_rate: 采样率
            text: 对应的文本（用于日志）
            emit_events: 是否发布事件
            
        Returns:
            bool: True=播放完成, False=被中断
        """
        with self._play_lock:
            self._playing = True
            self._interrupted = False
            self._current_text = text
            
            # 发布TTS开始事件
            if emit_events and self.event_bus and EVENT_BUS_AVAILABLE:
                self.event_bus.emit(
                    EventType.TTS_START,
                    data={"text": text[:100], "streaming": True},
                    source="interruptible_player"
                )
            
            # 回调
            if self._on_playback_start:
                try:
                    self._on_playback_start(text)
                except Exception as e:
                    logger.error(f"Playback start callback error: {e}")
            
            try:
                # 打开音频流
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=self.chunk_size
                )
                
                chunk_count = 0
                first_chunk_received = False
                
                for chunk_data in audio_stream:
                    # 检查是否被打断
                    if self._interrupted:
                        logger.info(f"Streaming playback interrupted at chunk {chunk_count}")
                        break
                    
                    if not first_chunk_received:
                        first_chunk_received = True
                        logger.info("First audio chunk received, starting playback...")
                    
                    # 转换bytes到numpy
                    if isinstance(chunk_data, bytes):
                        audio_chunk = np.frombuffer(chunk_data, dtype=np.int16)
                    else:
                        audio_chunk = chunk_data
                    
                    # 回调参考信号（用于AEC）
                    if self._on_reference_chunk:
                        try:
                            self._on_reference_chunk(audio_chunk)
                        except Exception:
                            pass
                    
                    # 播放
                    stream.write(audio_chunk.tobytes())
                    chunk_count += 1
                
                stream.stop_stream()
                stream.close()
                logger.info(f"Streaming playback finished: {chunk_count} chunks")
                
            except Exception as e:
                logger.error(f"Streaming playback error: {e}")
            finally:
                self._playing = False
                
                # 发布TTS结束事件
                if emit_events and self.event_bus and EVENT_BUS_AVAILABLE:
                    if self._interrupted:
                        self.event_bus.emit(
                            EventType.TTS_INTERRUPTED,
                            data={"text": text[:100], "streaming": True},
                            source="interruptible_player"
                        )
                    else:
                        self.event_bus.emit(
                            EventType.TTS_END,
                            data={"text": text[:100], "streaming": True},
                            source="interruptible_player"
                        )
                
                # 回调
                if self._on_playback_end:
                    try:
                        self._on_playback_end(self._interrupted)
                    except Exception as e:
                        logger.error(f"Playback end callback error: {e}")
            
            return not self._interrupted
    
    def on_reference_chunk(self, callback: Callable):
        """
        设置参考信号回调（用于AEC）
        
        Args:
            callback: (audio_chunk: np.ndarray) -> None
        """
        self._on_reference_chunk = callback
    
    def on_playback_start(self, callback: Callable):
        """
        设置播放开始回调
        
        Args:
            callback: (text: str) -> None
        """
        self._on_playback_start = callback
    
    def on_playback_end(self, callback: Callable):
        """
        设置播放结束回调
        
        Args:
            callback: (was_interrupted: bool) -> None
        """
        self._on_playback_end = callback
    
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
    
    # 测试可中断播放器
    print("\n--- Testing InterruptibleAudioPlayer ---")
    interruptible_player = InterruptibleAudioPlayer()
    
    def on_ref_chunk(chunk):
        pass  # AEC参考信号回调位置
    
    interruptible_player.on_reference_chunk(on_ref_chunk)
    
    # 生成测试音频（正弦波）
    import math
    duration = 3.0
    freq = 440.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), False)
    test_audio = (np.sin(2 * math.pi * freq * t) * 16000).astype(np.int16)
    
    print("Playing test audio (interrupt by Ctrl+C)...")
    try:
        completed = interruptible_player.play(test_audio, sr, "Test audio")
        print(f"Playback {'completed' if completed else 'interrupted'}")
    except KeyboardInterrupt:
        interruptible_player.stop()
        print("Stopped by user")
