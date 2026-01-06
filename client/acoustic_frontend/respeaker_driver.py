"""
ReSpeaker 6-Mic Circular Array 驱动模块

硬件特性:
- 2 x AC108 ADC (4通道/片，共8通道)
- 6个全向麦克风 (通道0-5)
- 2个回声通道 (通道6-7，用于AEC)
- 1 x AC101 DAC (音频输出)
- 12 x RGB LED (状态指示)
- 1个用户按钮
"""

import logging
import numpy as np
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)

# 尝试导入树莓派特定库
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available (not on Raspberry Pi)")


class DeviceType(Enum):
    """设备类型"""
    RESPEAKER_6MIC = "seeed-8mic-voicecard"
    RESPEAKER_4MIC = "seeed-4mic-voicecard"
    USB_MICROPHONE = "usb"
    DEFAULT = "default"


@dataclass
class ReSpeakerConfig:
    """ReSpeaker配置"""
    sample_rate: int = 16000
    channels: int = 8  # 6 mic + 2 echo
    chunk_size: int = 1024
    format_width: int = 2  # 16-bit
    mic_channels: List[int] = None  # [0,1,2,3,4,5]
    echo_channels: List[int] = None  # [6,7]
    button_gpio: int = 26  # 按钮GPIO引脚
    
    def __post_init__(self):
        if self.mic_channels is None:
            self.mic_channels = list(range(6))
        if self.echo_channels is None:
            self.echo_channels = [6, 7]


class ReSpeakerDriver:
    """
    ReSpeaker 6-Mic Circular Array 驱动
    
    功能:
    - 8通道音频采集
    - 通道分离 (麦克风/回声)
    - 按钮事件检测
    - 设备自动发现
    """
    
    def __init__(self, config: ReSpeakerConfig = None):
        """
        初始化驱动
        
        Args:
            config: ReSpeaker配置
        """
        self.config = config or ReSpeakerConfig()
        self.device_index = None
        self.stream = None
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.capture_thread = None
        
        # 按钮回调
        self.button_callback = None
        
        # 初始化PyAudio
        if PYAUDIO_AVAILABLE:
            self.pa = pyaudio.PyAudio()
            self._find_device()
        else:
            self.pa = None
            logger.error("PyAudio not available, ReSpeaker driver disabled")
    
    def _find_device(self) -> Optional[int]:
        """自动发现ReSpeaker设备"""
        if not self.pa:
            return None
        
        info = self.pa.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount', 0)
        
        for i in range(num_devices):
            device_info = self.pa.get_device_info_by_host_api_device_index(0, i)
            device_name = device_info.get('name', '')
            max_input_channels = device_info.get('maxInputChannels', 0)
            
            # 检查是否是ReSpeaker设备
            if 'seeed' in device_name.lower() and max_input_channels >= 8:
                self.device_index = i
                logger.info(f"Found ReSpeaker device: {device_name} (index={i})")
                return i
            
            # 备用: 检查8通道输入设备
            if max_input_channels >= 8 and self.device_index is None:
                self.device_index = i
                logger.info(f"Found 8-channel device: {device_name} (index={i})")
        
        if self.device_index is None:
            logger.warning("No ReSpeaker device found, using default")
        
        return self.device_index
    
    def start(self):
        """开始音频采集"""
        if not self.pa or self.is_running:
            return
        
        try:
            self.stream = self.pa.open(
                rate=self.config.sample_rate,
                format=self.pa.get_format_from_width(self.config.format_width),
                channels=self.config.channels,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.config.chunk_size,
            )
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # 初始化按钮监听
            if GPIO_AVAILABLE:
                self._setup_button()
            
            logger.info(f"ReSpeaker started: {self.config.sample_rate}Hz, {self.config.channels}ch")
            
        except Exception as e:
            logger.error(f"Failed to start ReSpeaker: {e}")
            self.is_running = False
    
    def stop(self):
        """停止音频采集"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except:
                pass
        
        logger.info("ReSpeaker stopped")
    
    def _capture_loop(self):
        """音频采集循环"""
        while self.is_running:
            try:
                data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
                
                # 转换为numpy数组 (samples, channels)
                audio = np.frombuffer(data, dtype=np.int16)
                audio = audio.reshape(-1, self.config.channels)
                
                # 放入队列
                try:
                    self.audio_queue.put_nowait(audio)
                except queue.Full:
                    # 丢弃最旧的数据
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(audio)
                    except:
                        pass
                        
            except Exception as e:
                if self.is_running:
                    logger.error(f"Capture error: {e}")
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        读取一帧音频
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            音频数据 shape=(samples, 8)
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def read_separated(self, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取并分离麦克风和回声通道
        
        Returns:
            (mic_data, echo_data): 麦克风数据(6ch)和回声数据(2ch)
        """
        audio = self.read(timeout)
        if audio is None:
            return None, None
        
        mic_data = audio[:, self.config.mic_channels]
        echo_data = audio[:, self.config.echo_channels]
        
        return mic_data, echo_data
    
    def _setup_button(self):
        """设置按钮GPIO"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.button_gpio, GPIO.IN)
            GPIO.add_event_detect(
                self.config.button_gpio, 
                GPIO.FALLING, 
                callback=self._on_button_press,
                bouncetime=200
            )
            logger.info(f"Button configured on GPIO {self.config.button_gpio}")
        except Exception as e:
            logger.warning(f"Failed to setup button: {e}")
    
    def _on_button_press(self, channel):
        """按钮按下回调"""
        logger.debug("Button pressed")
        if self.button_callback:
            self.button_callback()
    
    def set_button_callback(self, callback: Callable):
        """设置按钮回调函数"""
        self.button_callback = callback
    
    def get_device_info(self) -> dict:
        """获取设备信息"""
        return {
            "device_index": self.device_index,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "mic_channels": self.config.mic_channels,
            "echo_channels": self.config.echo_channels,
            "is_running": self.is_running,
            "pyaudio_available": PYAUDIO_AVAILABLE,
            "gpio_available": GPIO_AVAILABLE,
        }
    
    def __del__(self):
        """析构函数"""
        self.stop()
        if self.pa:
            self.pa.terminate()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    driver = ReSpeakerDriver()
    print("Device Info:", driver.get_device_info())
    
    if driver.device_index is not None:
        driver.start()
        
        print("Recording 3 seconds...")
        frames = []
        for _ in range(int(3 * 16000 / 1024)):
            audio = driver.read(timeout=1.0)
            if audio is not None:
                frames.append(audio)
        
        driver.stop()
        
        if frames:
            all_audio = np.vstack(frames)
            print(f"Captured {len(all_audio)} samples, shape: {all_audio.shape}")
