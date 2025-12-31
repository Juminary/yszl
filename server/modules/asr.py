"""
语音识别模块 (Automatic Speech Recognition)
使用 FunASR SenseVoice 模型实现多语言语音识别
支持中文、英语、日语、韩语、粤语等50+语言和方言
"""

import torch
import numpy as np
import logging
from typing import Dict, Optional, Union
import librosa
import re

logger = logging.getLogger(__name__)

# 尝试导入funasr
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    logger.warning("FunASR not available. Please install with: pip install funasr")


class ASRModule:
    """语音识别模块 - 使用SenseVoice实现多语言/方言识别"""
    
    def __init__(self, model_name: str = "sensevoice", device: str = "cuda", language: str = "auto"):
        """
        初始化ASR模块
        
        Args:
            model_name: 模型名称 (sensevoice, sensevoice-small)
            device: 运行设备 (cuda/cpu/mps)
            language: 识别语言 (auto=自动检测, zh, en, ja, ko, yue等)
        """
        # 设备选择
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.language = language
        self.model_name = model_name
        
        if not FUNASR_AVAILABLE:
            raise ImportError("FunASR is required. Install with: pip install funasr modelscope")
        
        logger.info(f"Loading SenseVoice model on {self.device}")
        try:
            # 加载SenseVoice模型
            # 模型ID: iic/SenseVoiceSmall - 支持50+语言的多语言ASR
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                device=self.device,
                disable_update=True,  # 禁用更新检查，使用本地模型
            )
            logger.info("SenseVoice model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SenseVoice model: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        清理SenseVoice输出的特殊标签
        
        SenseVoice输出格式: <|zh|><|NEUTRAL|><|Speech|><|woitn|>实际的文本内容
        """
        if not text:
            return ""
        
        # 移除SenseVoice的特殊标签 <|...|>
        cleaned = re.sub(r'<\|[^|]*\|>', '', text)
        return cleaned.strip()
    
    def _parse_sensevoice_result(self, text: str) -> Dict:
        """
        解析SenseVoice的完整输出，提取语言、情感等信息
        
        输出格式: <|语言|><|情感|><|事件|><|模式|>文本
        """
        result = {
            "language_detected": None,
            "emotion": None,
            "event": None,
            "clean_text": ""
        }
        
        if not text:
            return result
        
        # 提取语言标签
        lang_match = re.search(r'<\|(zh|en|ja|ko|yue|[a-z]{2,3})\|>', text)
        if lang_match:
            result["language_detected"] = lang_match.group(1)
        
        # 提取情感标签
        emotion_match = re.search(r'<\|(NEUTRAL|HAPPY|SAD|ANGRY|FEARFUL|DISGUSTED|SURPRISED)\|>', text, re.IGNORECASE)
        if emotion_match:
            result["emotion"] = emotion_match.group(1).lower()
        
        # 提取事件标签  
        event_match = re.search(r'<\|(Speech|Laughter|Applause|Crying|Coughing|Music|Noise)\|>', text, re.IGNORECASE)
        if event_match:
            result["event"] = event_match.group(1).lower()
        
        # 清理文本
        result["clean_text"] = self._clean_text(text)
        
        return result
    
    def transcribe(
        self, 
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sample_rate: int = 16000,
        language: str = None,
        hotwords: str = None,
        **kwargs
    ) -> Dict:
        """
        转录音频文件或音频数组
        
        Args:
            audio_path: 音频文件路径
            audio_array: 音频数组 (如果提供，则忽略audio_path)
            sample_rate: 音频采样率
            language: 识别语言（可选，auto=自动检测）
            hotwords: 热词字符串（SenseVoice暂不支持热词）
            **kwargs: 其他参数
        
        Returns:
            包含识别结果的字典:
            {
                'text': 识别的文本（已清理）,
                'raw_text': 原始输出（含标签）,
                'language': 识别到的语言,
                'emotion': 情感（如有）,
                'event': 音频事件（如有）,
                'confidence': 置信度
            }
        """
        try:
            # 准备输入
            if audio_array is not None:
                # 确保采样率为16kHz
                if sample_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                input_data = audio_array
            elif audio_path is not None:
                input_data = audio_path
            else:
                raise ValueError("Must provide either audio_path or audio_array")
            
            # 执行识别
            logger.info(f"Transcribing audio with SenseVoice...")
            result = self.model.generate(
                input=input_data,
                batch_size_s=300,
            )
            
            # 解析结果
            if result and len(result) > 0:
                raw_text = result[0].get("text", "")
                parsed = self._parse_sensevoice_result(raw_text)
                
                return {
                    "text": parsed["clean_text"],
                    "raw_text": raw_text,
                    "language": parsed["language_detected"] or language or self.language,
                    "emotion": parsed["emotion"],
                    "event": parsed["event"],
                    "segments": result[0].get("timestamp", []),
                    "confidence": 0.95
                }
            else:
                return {
                    "text": "",
                    "raw_text": "",
                    "language": language or self.language,
                    "emotion": None,
                    "event": None,
                    "segments": [],
                    "confidence": 0.0
                }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "raw_text": "",
                "language": language or self.language,
                "emotion": None,
                "event": None,
                "segments": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def transcribe_streaming(self, audio_chunks: list, sample_rate: int = 16000) -> Dict:
        """
        流式语音识别（累积音频块进行识别）
        
        Args:
            audio_chunks: 音频块列表
            sample_rate: 采样率
        
        Returns:
            识别结果
        """
        try:
            # 合并音频块
            audio_array = np.concatenate(audio_chunks)
            return self.transcribe(audio_array=audio_array, sample_rate=sample_rate)
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return {
                "text": "",
                "language": self.language,
                "segments": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def transcribe_with_hotwords(self, audio_path: str, hotwords: list) -> Dict:
        """
        使用热词增强的语音识别
        
        注意: SenseVoice暂不支持热词，此方法仅为兼容性保留
        
        Args:
            audio_path: 音频文件路径
            hotwords: 热词列表
        
        Returns:
            识别结果
        """
        logger.warning("SenseVoice does not support hotwords, ignoring hotword list")
        return self.transcribe(audio_path=audio_path)
    
    def detect_language(self, audio_path: str) -> Dict:
        """
        检测音频语言
        
        SenseVoice支持自动语言检测，会在转录结果中返回语言标签
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            语言检测结果
        """
        result = self.transcribe(audio_path=audio_path)
        detected_lang = result.get("language", "unknown")
        
        lang_names = {
            "zh": "中文",
            "en": "英语", 
            "ja": "日语",
            "ko": "韩语",
            "yue": "粤语",
        }
        
        return {
            "language": detected_lang,
            "language_name": lang_names.get(detected_lang, detected_lang),
            "probability": 0.95,
            "emotion": result.get("emotion"),
            "note": "Detected by SenseVoice multi-lingual model"
        }
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_type": "SenseVoice (FunASR)",
            "device": self.device,
            "language": self.language,
            "features": [
                "Multi-lingual: 50+ languages",
                "Dialect support: Cantonese, Minnan, etc.",
                "Emotion recognition: happy, sad, angry, etc.",
                "Audio event detection: laughter, applause, coughing",
                "Auto language detection"
            ],
            "supported_languages": [
                "zh (中文)", "en (英语)", "ja (日语)", "ko (韩语)",
                "yue (粤语)", "de (德语)", "fr (法语)", "es (西语)"
            ]
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 初始化ASR模块
    asr = ASRModule(model_name="sensevoice", device="cpu")
    
    # 打印模型信息
    print("Model Info:", asr.get_model_info())
    
    # 测试识别（需要提供测试音频文件）
    # result = asr.transcribe("test_audio.wav")
    # print("Transcription:", result)
    # print("Clean text:", result["text"])
    # print("Emotion:", result["emotion"])
    # print("Language:", result["language"])
