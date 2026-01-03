# Audio modules - 音频分析模块
from .emotion import EmotionModule
from .speaker import SpeakerModule
from .paralinguistic import ParalinguisticAnalyzer
from .sound_event import SoundEventDetector

__all__ = [
    'EmotionModule',
    'SpeakerModule',
    'ParalinguisticAnalyzer',
    'SoundEventDetector'
]
