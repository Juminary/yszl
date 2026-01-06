"""
声学前端增强模块 (Acoustic Frontend Enhancement)

基于 ReSpeaker 6-Mic Circular Array Kit for Raspberry Pi

硬件规格:
- 2 x AC108 ADC (8通道: 6麦克风 + 2回声参考)
- 6 x MSM321A3729H9CP 全向麦克风 (SNR 59dB)
- 1 x AC101 DAC (音频输出)
- 12 x APA102 RGB LED (状态指示)
- 1 x 用户按钮 (GPIO 26)

核心功能:
- mic_array: 统一的麦克风阵列接口，整合所有声学处理
- doa: GCC-PHAT 声源定位，360度方向估计
- beamformer: 延时求和波束成形，增强目标方向信号
- aec: NLMS自适应回声消除，利用硬件回声参考通道
- led_ring: LED环形灯控制，声源方向可视化
- respeaker_driver: 底层ReSpeaker硬件驱动

使用示例:
```python
from acoustic_frontend import MicrophoneArray, LEDRing

# 创建麦克风阵列
mic = MicrophoneArray()
led = LEDRing()

mic.start()
led.start()

# 读取处理后的音频帧
while True:
    frame = mic.read()
    if frame:
        # 显示声源方向
        if frame.doa_angle:
            led.show_doa(frame.doa_angle)
        
        # 使用消除回声后的音频
        clean_audio = frame.clean_audio
        # ... 后续处理 (ASR等)

mic.stop()
led.stop()
```
"""

# 核心模块
from .mic_array import MicrophoneArray, MicArrayConfig, AudioFrame, create_mic_array, create_from_config

# 声学处理组件
from .doa import DOAEstimator, DOAConfig
from .beamformer import Beamformer, BeamformerConfig, MVDR_Beamformer
from .aec import AcousticEchoCanceller, AECConfig, HardwareAEC

# LED控制
from .led_ring import LEDRing, LEDPattern, Color, Colors

# 底层驱动
from .respeaker_driver import ReSpeakerDriver, ReSpeakerConfig, DeviceType

__all__ = [
    # 主要接口
    'MicrophoneArray',
    'MicArrayConfig', 
    'AudioFrame',
    'create_mic_array',
    'create_from_config',
    
    # DOA
    'DOAEstimator',
    'DOAConfig',
    
    # 波束成形
    'Beamformer',
    'BeamformerConfig',
    'MVDR_Beamformer',
    
    # 回声消除
    'AcousticEchoCanceller',
    'AECConfig',
    'HardwareAEC',
    
    # LED
    'LEDRing',
    'LEDPattern',
    'Color',
    'Colors',
    
    # 驱动
    'ReSpeakerDriver',
    'ReSpeakerConfig',
    'DeviceType',
]

__version__ = '2.0.0'
__author__ = 'YiShengZhiLian Team'
