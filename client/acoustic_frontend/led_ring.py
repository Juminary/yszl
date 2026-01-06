"""
LED环形灯控制模块

ReSpeaker 6-Mic Circular Array 配备 12 颗 RGB LED
可用于状态指示、声源方向指示等

硬件特性:
- 12 x APA102 RGB LED (SPI控制)
- 环形排列，与麦克风阵列同心
- 支持独立颜色控制
- 亮度可调

LED布局 (俯视图):
           0° 
          [0]
      [11]    [1]
    [10]        [2]
     [9]        [3]
       [8]    [4]
         [7][6][5]
           180°
"""

import numpy as np
import logging
import time
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# 尝试导入SPI库
try:
    from spidev import SpiDev
    SPI_AVAILABLE = True
except ImportError:
    SPI_AVAILABLE = False
    logger.warning("spidev not available (not on Raspberry Pi)")

# 尝试导入pixel_ring库某些功能由它实现
try:
    from pixel_ring import pixel_ring
    PIXEL_RING_AVAILABLE = True
except ImportError:
    PIXEL_RING_AVAILABLE = False


class LEDPattern(Enum):
    """预设LED模式"""
    OFF = "off"
    SOLID = "solid"                    # 全亮单色
    BREATHING = "breathing"            # 呼吸灯效果
    SPIN = "spin"                      # 旋转效果
    DOA_INDICATOR = "doa_indicator"    # 声源方向指示
    LISTENING = "listening"            # 监听状态
    SPEAKING = "speaking"              # 说话状态
    THINKING = "thinking"              # 思考/处理状态
    SUCCESS = "success"                # 成功
    ERROR = "error"                    # 错误


@dataclass
class Color:
    """RGB颜色"""
    r: int
    g: int
    b: int
    brightness: int = 31  # 0-31
    
    def to_apa102(self) -> Tuple[int, int, int, int]:
        """转换为APA102格式 (brightness, b, g, r)"""
        return (0xE0 | (self.brightness & 0x1F), self.b, self.g, self.r)

    @classmethod
    def from_hex(cls, hex_color: str, brightness: int = 31) -> 'Color':
        """从十六进制创建颜色"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return cls(r, g, b, brightness)


# 预定义颜色
class Colors:
    OFF = Color(0, 0, 0, 0)
    WHITE = Color(255, 255, 255)
    RED = Color(255, 0, 0)
    GREEN = Color(0, 255, 0)
    BLUE = Color(0, 0, 255)
    CYAN = Color(0, 255, 255)
    YELLOW = Color(255, 255, 0)
    MAGENTA = Color(255, 0, 255)
    ORANGE = Color(255, 128, 0)
    PURPLE = Color(128, 0, 255)
    
    # 医疗主题色
    MEDICAL_BLUE = Color(0, 150, 255)
    MEDICAL_GREEN = Color(0, 200, 100)
    CALM_BLUE = Color(100, 180, 255)


class LEDRing:
    """
    ReSpeaker LED环形灯控制器
    
    使用示例:
    ```python
    led = LEDRing()
    led.start()
    
    # 显示声源方向
    led.show_doa(90)  # 90度方向亮起
    
    # 使用预设模式
    led.set_pattern(LEDPattern.LISTENING)
    
    # 自定义颜色
    led.fill(Colors.MEDICAL_BLUE)
    
    led.stop()
    ```
    """
    
    NUM_LEDS = 12
    
    def __init__(self, use_pixel_ring: bool = True):
        """
        初始化LED控制器
        
        Args:
            use_pixel_ring: 是否使用pixel_ring库 (更简单但依赖较多)
        """
        self._use_pixel_ring = use_pixel_ring and PIXEL_RING_AVAILABLE
        self._use_spi = False
        
        # LED状态缓冲
        self._buffer: List[Color] = [Colors.OFF for _ in range(self.NUM_LEDS)]
        
        # 动画状态
        self._current_pattern = LEDPattern.OFF
        self._animation_thread: Optional[threading.Thread] = None
        self._animation_running = False
        self._lock = threading.Lock()
        
        # DOA相关
        self._current_doa = 0.0
        
        # 初始化硬件
        self._init_hardware()
        
        logger.info(f"LEDRing initialized: pixel_ring={self._use_pixel_ring}, spi={self._use_spi}")
    
    def _init_hardware(self):
        """初始化硬件接口"""
        if self._use_pixel_ring:
            try:
                pixel_ring.set_brightness(10)
                pixel_ring.off()
                return
            except Exception as e:
                logger.warning(f"pixel_ring init failed: {e}")
                self._use_pixel_ring = False
        
        # 回退到直接SPI控制
        if SPI_AVAILABLE:
            try:
                self._spi = SpiDev()
                self._spi.open(0, 1)  # SPI0, CE1
                self._spi.max_speed_hz = 8000000
                self._use_spi = True
            except Exception as e:
                logger.warning(f"SPI init failed: {e}")
                self._use_spi = False
    
    def start(self):
        """启动LED控制 (开启动画线程)"""
        if self._animation_running:
            return
        
        self._animation_running = True
        self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self._animation_thread.start()
    
    def stop(self):
        """停止LED控制"""
        self._animation_running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=1.0)
        self.off()
    
    def _animation_loop(self):
        """动画主循环"""
        frame = 0
        while self._animation_running:
            with self._lock:
                pattern = self._current_pattern
            
            if pattern == LEDPattern.BREATHING:
                self._animate_breathing(frame)
            elif pattern == LEDPattern.SPIN:
                self._animate_spin(frame)
            elif pattern == LEDPattern.LISTENING:
                self._animate_listening(frame)
            elif pattern == LEDPattern.SPEAKING:
                self._animate_speaking(frame)
            elif pattern == LEDPattern.THINKING:
                self._animate_thinking(frame)
            elif pattern == LEDPattern.DOA_INDICATOR:
                self._animate_doa(self._current_doa)
            
            frame += 1
            time.sleep(0.05)  # 20 FPS
    
    def set_pattern(self, pattern: LEDPattern, color: Color = None):
        """
        设置LED模式
        
        Args:
            pattern: LED模式
            color: 可选的主色调
        """
        with self._lock:
            self._current_pattern = pattern
        
        if pattern == LEDPattern.OFF:
            self.off()
        elif pattern == LEDPattern.SOLID:
            self.fill(color or Colors.WHITE)
        elif pattern == LEDPattern.SUCCESS:
            self.fill(Colors.MEDICAL_GREEN)
        elif pattern == LEDPattern.ERROR:
            self.fill(Colors.RED)
    
    def show_doa(self, angle: float, color: Color = None):
        """
        显示声源方向
        
        Args:
            angle: DOA角度 (0-360度)
            color: 指示颜色
        """
        self._current_doa = angle
        with self._lock:
            self._current_pattern = LEDPattern.DOA_INDICATOR
    
    # 说话人角色颜色映射
    SPEAKER_ROLE_COLORS = {
        "doctor": Color(0, 150, 255, 31),      # 医疗蓝
        "patient": Color(0, 255, 120, 31),     # 健康绿
        "family": Color(255, 180, 0, 31),      # 温暖橙
        "unknown": Color(200, 200, 200, 25),   # 灰白
    }
    
    def show_speaker_doa(self, angle: float, speaker_role: str = "unknown"):
        """
        显示带说话人颜色的DOA指示
        
        不同说话人使用不同颜色:
        - doctor: 蓝色
        - patient: 绿色
        - family: 橙色
        
        Args:
            angle: DOA角度
            speaker_role: 说话人角色
        """
        # 获取角色颜色
        color = self.SPEAKER_ROLE_COLORS.get(
            speaker_role.lower(), 
            self.SPEAKER_ROLE_COLORS["unknown"]
        )
        
        # 每个LED覆盖30度 (360/12)
        led_index = int((angle % 360) / 30) % self.NUM_LEDS
        
        # 清空所有LED
        self._buffer = [Colors.OFF for _ in range(self.NUM_LEDS)]
        
        # 主方向LED使用角色颜色
        self._buffer[led_index] = color
        
        # 相邻LED稍暗（渐变效果）
        left = (led_index - 1) % self.NUM_LEDS
        right = (led_index + 1) % self.NUM_LEDS
        dim_color = Color(color.r // 2, color.g // 2, color.b // 2, color.brightness // 2)
        self._buffer[left] = dim_color
        self._buffer[right] = dim_color
        
        self._flush()
    
    def show_multi_speakers(self, speakers: list):
        """
        同时显示多个说话人的DOA
        
        Args:
            speakers: 列表，每个元素为 {"angle": float, "role": str}
        """
        self._buffer = [Colors.OFF for _ in range(self.NUM_LEDS)]
        
        for speaker in speakers:
            angle = speaker.get("angle", 0)
            role = speaker.get("role", "unknown")
            
            color = self.SPEAKER_ROLE_COLORS.get(
                role.lower(),
                self.SPEAKER_ROLE_COLORS["unknown"]
            )
            
            led_index = int((angle % 360) / 30) % self.NUM_LEDS
            self._buffer[led_index] = color
        
        self._flush()

    def _animate_doa(self, angle: float):
        """渲染DOA指示"""
        # 每个LED覆盖30度 (360/12)
        led_index = int((angle % 360) / 30) % self.NUM_LEDS
        
        # 清空所有LED
        self._buffer = [Colors.OFF for _ in range(self.NUM_LEDS)]
        
        # 主方向LED最亮
        self._buffer[led_index] = Color(0, 200, 255, 31)
        
        # 相邻LED稍暗（渐变效果）
        left = (led_index - 1) % self.NUM_LEDS
        right = (led_index + 1) % self.NUM_LEDS
        self._buffer[left] = Color(0, 100, 200, 15)
        self._buffer[right] = Color(0, 100, 200, 15)
        
        self._flush()
    
    def _animate_listening(self, frame: int):
        """监听状态动画 - 柔和的蓝色脉冲"""
        brightness = int(15 + 10 * np.sin(frame * 0.1))
        color = Color(100, 180, 255, brightness)
        self._buffer = [color for _ in range(self.NUM_LEDS)]
        self._flush()
    
    def _animate_speaking(self, frame: int):
        """说话状态动画 - 绿色环形波"""
        for i in range(self.NUM_LEDS):
            phase = (frame + i * 3) % 30
            if phase < 15:
                brightness = int(5 + 20 * (phase / 15))
            else:
                brightness = int(25 - 20 * ((phase - 15) / 15))
            self._buffer[i] = Color(0, 200, 100, brightness)
        self._flush()
    
    def _animate_thinking(self, frame: int):
        """思考状态动画 - 旋转的点"""
        self._buffer = [Colors.OFF for _ in range(self.NUM_LEDS)]
        pos = frame % self.NUM_LEDS
        self._buffer[pos] = Color(255, 200, 0, 31)
        self._buffer[(pos - 1) % self.NUM_LEDS] = Color(255, 200, 0, 15)
        self._buffer[(pos - 2) % self.NUM_LEDS] = Color(255, 200, 0, 5)
        self._flush()
    
    def _animate_breathing(self, frame: int):
        """呼吸灯效果"""
        brightness = int(5 + 26 * (1 + np.sin(frame * 0.05)) / 2)
        color = Color(100, 180, 255, brightness)
        self._buffer = [color for _ in range(self.NUM_LEDS)]
        self._flush()
    
    def _animate_spin(self, frame: int):
        """旋转效果"""
        for i in range(self.NUM_LEDS):
            hue = ((frame * 5 + i * 30) % 360) / 360
            r, g, b = self._hsv_to_rgb(hue, 1.0, 1.0)
            self._buffer[i] = Color(int(r*255), int(g*255), int(b*255), 20)
        self._flush()
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """HSV转RGB"""
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)
    
    def fill(self, color: Color):
        """填充所有LED为同一颜色"""
        with self._lock:
            self._current_pattern = LEDPattern.SOLID
        self._buffer = [color for _ in range(self.NUM_LEDS)]
        self._flush()
    
    def set_led(self, index: int, color: Color):
        """设置单个LED颜色"""
        if 0 <= index < self.NUM_LEDS:
            self._buffer[index] = color
            self._flush()
    
    def off(self):
        """关闭所有LED"""
        with self._lock:
            self._current_pattern = LEDPattern.OFF
        
        if self._use_pixel_ring:
            try:
                pixel_ring.off()
                return
            except:
                pass
        
        self._buffer = [Colors.OFF for _ in range(self.NUM_LEDS)]
        self._flush()
    
    def _flush(self):
        """将缓冲区刷新到硬件"""
        if self._use_pixel_ring:
            try:
                # pixel_ring.set_color更接近原生实现
                # 这里简化处理
                return
            except:
                pass
        
        if self._use_spi:
            self._spi_write()
    
    def _spi_write(self):
        """通过SPI写入LED数据 (APA102协议)"""
        if not self._use_spi:
            return
        
        # 开始帧 (32个0)
        data = [0x00] * 4
        
        # LED数据
        for color in self._buffer:
            frame = color.to_apa102()
            data.extend(frame)
        
        # 结束帧
        data.extend([0xFF] * 4)
        
        try:
            self._spi.xfer2(data)
        except Exception as e:
            logger.error(f"SPI write error: {e}")
    
    def set_brightness(self, brightness: int):
        """
        设置全局亮度
        
        Args:
            brightness: 亮度值 (0-31)
        """
        brightness = max(0, min(31, brightness))
        
        if self._use_pixel_ring:
            try:
                pixel_ring.set_brightness(brightness)
                return
            except:
                pass
        
        for color in self._buffer:
            color.brightness = brightness
        self._flush()
    
    def __del__(self):
        self.stop()


# ============================================================
# 便捷函数
# ============================================================

def create_led_ring() -> LEDRing:
    """创建LED环控制器"""
    return LEDRing()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("Testing LED Ring...")
    led = LEDRing()
    led.start()
    
    print("1. Showing DOA indicator at 90°")
    led.show_doa(90)
    time.sleep(2)
    
    print("2. Listening pattern")
    led.set_pattern(LEDPattern.LISTENING)
    time.sleep(2)
    
    print("3. Speaking pattern")
    led.set_pattern(LEDPattern.SPEAKING)
    time.sleep(2)
    
    print("4. Thinking pattern")
    led.set_pattern(LEDPattern.THINKING)
    time.sleep(2)
    
    print("5. Success")
    led.set_pattern(LEDPattern.SUCCESS)
    time.sleep(1)
    
    print("6. Off")
    led.stop()
    
    print("Test complete!")
