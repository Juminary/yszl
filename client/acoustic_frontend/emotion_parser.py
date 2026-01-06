"""
SenseVoice 情感标签解析器

SenseVoice 是阿里达摩院开发的多任务语音模型，能够在ASR转录的同时
输出情感标签和音频事件标签。

输出格式示例:
"我今天心情不太好<|SAD|>，有点累了<|TIRED|>"

支持的情感标签:
- <|HAPPY|>   - 开心
- <|SAD|>     - 悲伤
- <|ANGRY|>   - 愤怒
- <|NEUTRAL|> - 中性
- <|FEAR|>    - 恐惧
- <|SURPRISE|> - 惊讶

音频事件标签:
- <|LAUGHTER|> - 笑声
- <|COUGH|>    - 咳嗽
- <|SIGH|>     - 叹气
- <|NOISE|>    - 噪音
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """情感类型枚举"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    UNKNOWN = "unknown"


class AudioEventType(Enum):
    """音频事件类型"""
    LAUGHTER = "laughter"
    COUGH = "cough"
    SIGH = "sigh"
    NOISE = "noise"
    SPEECH = "speech"


@dataclass
class EmotionResult:
    """情感解析结果"""
    clean_text: str                      # 移除标签后的纯文本
    emotions: List[str]                  # 检测到的情感列表
    dominant_emotion: str                # 主导情感
    audio_events: List[str]              # 音频事件
    confidence: float = 1.0              # 置信度
    raw_text: str = ""                   # 原始文本
    
    def to_dict(self) -> Dict:
        return {
            "clean_text": self.clean_text,
            "emotions": self.emotions,
            "dominant_emotion": self.dominant_emotion,
            "audio_events": self.audio_events,
            "confidence": self.confidence,
        }
    
    def has_emotion(self, emotion: str) -> bool:
        """检查是否包含特定情感"""
        return emotion.upper() in [e.upper() for e in self.emotions]
    
    def is_negative(self) -> bool:
        """是否为负面情感"""
        negative = {"sad", "angry", "fear"}
        return self.dominant_emotion.lower() in negative


class SenseVoiceEmotionParser:
    """
    SenseVoice 情感标签解析器
    
    从ASR输出文本中提取情感和事件标签
    
    使用示例:
    ```python
    parser = SenseVoiceEmotionParser()
    
    # 解析带标签的文本
    result = parser.parse("今天很开心<|HAPPY|>")
    
    print(result.clean_text)       # "今天很开心"
    print(result.dominant_emotion)  # "happy"
    ```
    """
    
    # 情感标签正则
    EMOTION_PATTERN = re.compile(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEAR|SURPRISE)\|>", re.IGNORECASE)
    
    # 音频事件标签正则
    EVENT_PATTERN = re.compile(r"<\|(LAUGHTER|COUGH|SIGH|NOISE|SPEECH)\|>", re.IGNORECASE)
    
    # 所有标签正则 (用于清理)
    ALL_TAGS_PATTERN = re.compile(r"<\|[A-Z_]+\|>", re.IGNORECASE)
    
    # 情感优先级 (越高越重要)
    EMOTION_PRIORITY = {
        "angry": 5,
        "fear": 4,
        "sad": 3,
        "surprise": 2,
        "happy": 1,
        "neutral": 0,
    }
    
    def __init__(self):
        """初始化解析器"""
        logger.info("SenseVoiceEmotionParser initialized")
    
    def parse(self, text: str) -> EmotionResult:
        """
        解析带标签的文本
        
        Args:
            text: SenseVoice输出的带标签文本
            
        Returns:
            EmotionResult 对象
        """
        if not text:
            return EmotionResult(
                clean_text="",
                emotions=[],
                dominant_emotion="neutral",
                audio_events=[],
                raw_text=text,
            )
        
        # 提取情感标签
        emotions = self._extract_emotions(text)
        
        # 提取音频事件
        audio_events = self._extract_events(text)
        
        # 清理文本
        clean_text = self._clean_text(text)
        
        # 确定主导情感
        dominant = self._get_dominant_emotion(emotions)
        
        return EmotionResult(
            clean_text=clean_text.strip(),
            emotions=emotions,
            dominant_emotion=dominant,
            audio_events=audio_events,
            raw_text=text,
        )
    
    def _extract_emotions(self, text: str) -> List[str]:
        """提取情感标签"""
        matches = self.EMOTION_PATTERN.findall(text)
        return [m.lower() for m in matches]
    
    def _extract_events(self, text: str) -> List[str]:
        """提取音频事件标签"""
        matches = self.EVENT_PATTERN.findall(text)
        return [m.lower() for m in matches]
    
    def _clean_text(self, text: str) -> str:
        """移除所有标签"""
        return self.ALL_TAGS_PATTERN.sub("", text)
    
    def _get_dominant_emotion(self, emotions: List[str]) -> str:
        """
        确定主导情感
        
        规则:
        1. 如果没有情感，返回neutral
        2. 按优先级选择最重要的情感
        3. 相同优先级取最后出现的
        """
        if not emotions:
            return "neutral"
        
        # 按优先级排序
        sorted_emotions = sorted(
            emotions,
            key=lambda e: self.EMOTION_PRIORITY.get(e.lower(), 0),
            reverse=True
        )
        
        return sorted_emotions[0]
    
    def extract_for_tts(self, text: str) -> Tuple[str, str]:
        """
        为TTS准备：提取干净文本和情感指令
        
        Returns:
            (clean_text, emotion_instruction)
        """
        result = self.parse(text)
        
        # 根据主导情感生成TTS指令
        emotion_map = {
            "happy": "用欢快、明亮的语气",
            "sad": "用温柔、安慰的语气",
            "angry": "用平静、舒缓的语气",  # 对愤怒用户要冷静回应
            "fear": "用镇定、关怀的语气",
            "neutral": "",
        }
        
        instruction = emotion_map.get(result.dominant_emotion, "")
        
        return result.clean_text, instruction


class EmotionTracker:
    """
    情感跟踪器
    
    跟踪会话中的情感变化趋势
    """
    
    def __init__(self, window_size: int = 10):
        """
        初始化跟踪器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self._history: List[str] = []
        self._parser = SenseVoiceEmotionParser()
    
    def add(self, text: str) -> EmotionResult:
        """添加并解析一条文本"""
        result = self._parser.parse(text)
        
        if result.dominant_emotion != "neutral":
            self._history.append(result.dominant_emotion)
            if len(self._history) > self.window_size:
                self._history.pop(0)
        
        return result
    
    def get_trend(self) -> Dict[str, float]:
        """获取情感趋势（各情感占比）"""
        if not self._history:
            return {"neutral": 1.0}
        
        counts = {}
        for emotion in self._history:
            counts[emotion] = counts.get(emotion, 0) + 1
        
        total = len(self._history)
        return {k: v / total for k, v in counts.items()}
    
    def is_escalating_negative(self) -> bool:
        """检测负面情感是否在升级"""
        if len(self._history) < 3:
            return False
        
        negative = {"sad", "angry", "fear"}
        recent = self._history[-3:]
        
        return all(e in negative for e in recent)
    
    def reset(self):
        """重置历史"""
        self._history.clear()


# 便捷函数
def parse_emotion(text: str) -> EmotionResult:
    """快速解析情感"""
    parser = SenseVoiceEmotionParser()
    return parser.parse(text)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    parser = SenseVoiceEmotionParser()
    
    # 测试用例
    test_cases = [
        "今天天气真好<|HAPPY|>",
        "我头很疼<|SAD|>，有点难受",
        "你怎么回事<|ANGRY|>！",
        "啊<|SURPRISE|>？真的吗",
        "好的，明白了<|NEUTRAL|>",
        "<|COUGH|>咳咳<|COUGH|>",
        "普通文本没有标签",
    ]
    
    print("=== SenseVoice Emotion Parser Test ===\n")
    
    for text in test_cases:
        result = parser.parse(text)
        print(f"Input:    {text}")
        print(f"Clean:    {result.clean_text}")
        print(f"Emotion:  {result.dominant_emotion}")
        print(f"Events:   {result.audio_events}")
        print()
    
    # 测试TTS准备
    print("=== TTS Preparation ===\n")
    text, instruction = parser.extract_for_tts("我今天很难过<|SAD|>")
    print(f"Text: {text}")
    print(f"Instruction: {instruction}")
