"""
声学前端增强模块 (Acoustic Frontend Enhancement)

基于 ReSpeaker 6-Mic Circular Array Kit for Raspberry Pi

核心功能:
- 声源定位 (DOA): SRP-PHAT + ODAS后端
- 波束成形: 延时求和增强
- 回声消除: NLMS自适应滤波
- 说话人分离: 空间-声纹融合 (Spatio-Spectral Fusion)
- 延迟校准: Chirp信号测量
- 情感解析: SenseVoice标签提取
- LED控制: 多说话人方向可视化
"""

# 核心模块
from .mic_array import MicrophoneArray, MicArrayConfig, AudioFrame, create_mic_array, create_from_config

# DOA (支持ODAS后端)
from .doa import EnhancedDOAEstimator, DOABackend, DOAResult, DOAConfig

# 波束成形
from .beamformer import Beamformer, BeamformerConfig, MVDR_Beamformer

# 回声消除
from .aec import AcousticEchoCanceller, AECConfig, HardwareAEC

# 延迟校准
from .latency_calibrator import LatencyCalibrator, CalibrationResult, compensate_delay

# ODAS 客户端
from .odas_client import ODASClient, ODASManager, TrackedSource, ODASSourceState

# 声纹提取
from .speaker_embedder import SpeakerEmbedder, SpeakerEmbedding, EmbeddingConfig

# 空间-声纹融合
from .spatio_spectral_fusion import SpatioSpectralFusion, SpeakerCluster, FusionConfig

# 情感解析
from .emotion_parser import SenseVoiceEmotionParser, EmotionResult, EmotionTracker

# LED控制
from .led_ring import LEDRing, LEDPattern, Color, Colors

# 底层驱动
from .respeaker_driver import ReSpeakerDriver, ReSpeakerConfig, DeviceType

# 兼容性别名
DOAEstimator = EnhancedDOAEstimator

__all__ = [
    # 主要接口
    'MicrophoneArray', 'MicArrayConfig', 'AudioFrame',
    'create_mic_array', 'create_from_config',
    
    # DOA
    'DOAEstimator', 'EnhancedDOAEstimator', 'DOABackend', 'DOAResult', 'DOAConfig',
    
    # ODAS
    'ODASClient', 'ODASManager', 'TrackedSource', 'ODASSourceState',
    
    # 声纹嵌入
    'SpeakerEmbedder', 'SpeakerEmbedding', 'EmbeddingConfig',
    
    # 空间-声纹融合
    'SpatioSpectralFusion', 'SpeakerCluster', 'FusionConfig',
    
    # 波束成形
    'Beamformer', 'BeamformerConfig', 'MVDR_Beamformer',
    
    # 回声消除 + 校准
    'AcousticEchoCanceller', 'AECConfig', 'HardwareAEC',
    'LatencyCalibrator', 'CalibrationResult', 'compensate_delay',
    
    # 情感解析
    'SenseVoiceEmotionParser', 'EmotionResult', 'EmotionTracker',
    
    # LED
    'LEDRing', 'LEDPattern', 'Color', 'Colors',
    
    # 驱动
    'ReSpeakerDriver', 'ReSpeakerConfig', 'DeviceType',
]

__version__ = '2.4.0'
