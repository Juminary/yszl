"""
副语言信息感知模块 - 情感识别
使用 SenseVoice 实现多任务语音情感识别
"""

import torch
import numpy as np
import librosa
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入FunASR (SenseVoice)
try:
    from funasr import AutoModel
    SENSEVOICE_AVAILABLE = True
except ImportError:
    SENSEVOICE_AVAILABLE = False
    logger.warning("FunASR not available. Please install with: pip install funasr")


class EmotionModule:
    """情感识别模块 - 使用SenseVoice"""
    
    # 情感类别映射
    EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise"]
    EMOTIONS_ZH = ["中性", "开心", "悲伤", "愤怒", "恐惧", "惊讶"]
    
    # SenseVoice输出的情感标签映射
    SENSEVOICE_EMOTION_MAP = {
        "NEUTRAL": "neutral",
        "HAPPY": "happy", 
        "SAD": "sad",
        "ANGRY": "angry",
        "FEARFUL": "fear",
        "SURPRISED": "surprise",
        "DISGUSTED": "angry",  # 映射到angry
        "<|NEUTRAL|>": "neutral",
        "<|HAPPY|>": "happy",
        "<|SAD|>": "sad",
        "<|ANGRY|>": "angry",
    }
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        初始化情感识别模块
        
        Args:
            model_path: 模型路径（未使用，保持API兼容）
            device: 运行设备
        """
        # 设备选择
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.num_classes = len(self.EMOTIONS)
        
        if not SENSEVOICE_AVAILABLE:
            logger.warning("SenseVoice not available, using fallback mode")
            self.model = None
            return
        
        logger.info("Loading SenseVoice model...")
        try:
            # 加载SenseVoice模型
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                device=self.device,
                disable_update=True  # 禁用更新检查，使用本地模型
            )
            logger.info("SenseVoice model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SenseVoice model: {e}")
            self.model = None
    
    def predict(self, audio_path: str = None, audio_array: np.ndarray = None,
                sample_rate: int = 16000) -> Dict:
        """
        预测音频的情感
        
        Args:
            audio_path: 音频文件路径
            audio_array: 音频数组
            sample_rate: 采样率
        
        Returns:
            情感预测结果
        """
        try:
            if self.model is None:
                # 使用声学特征后备方案
                return self._predict_fallback(audio_path, audio_array, sample_rate)
            
            # 准备输入
            if audio_array is not None:
                # 保存临时文件（SenseVoice需要文件路径）
                import tempfile
                temp_path = tempfile.mktemp(suffix=".wav")
                import soundfile as sf
                if sample_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                sf.write(temp_path, audio_array, 16000)
                input_path = temp_path
            else:
                input_path = audio_path
            
            # 调用SenseVoice
            result = self.model.generate(
                input=input_path,
                language="auto",  # 自动检测语言
                use_itn=True
            )
            
            # 解析结果
            if result and len(result) > 0:
                text_result = result[0].get("text", "")
                
                # SenseVoice在文本中嵌入情感标签，如 "<|HAPPY|>你好"
                emotion = self._parse_emotion_from_text(text_result)
                emotion_zh = self.EMOTIONS_ZH[self.EMOTIONS.index(emotion)]
                
                # 提取韵律特征
                prosody = self._extract_prosody(audio_path, audio_array, sample_rate)
                
                return {
                    "emotion": emotion,
                    "emotion_zh": emotion_zh,
                    "confidence": 0.85,  # SenseVoice通常有较高准确率
                    "probabilities": {e: 0.1 for e in self.EMOTIONS},
                    "prosody": prosody,
                    "raw_text": text_result
                }
            else:
                return self._default_result()
                
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return self._default_result(error=str(e))
    
    def _parse_emotion_from_text(self, text: str) -> str:
        """从SenseVoice输出文本中解析情感标签"""
        text_upper = text.upper()
        
        for sv_label, emotion in self.SENSEVOICE_EMOTION_MAP.items():
            if sv_label in text_upper:
                return emotion
        
        return "neutral"
    
    def _predict_fallback(self, audio_path: str = None, audio_array: np.ndarray = None,
                          sample_rate: int = 16000) -> Dict:
        """后备方案：基于声学特征的简单情感判断"""
        try:
            if audio_array is not None:
                y = audio_array
                sr = sample_rate
            elif audio_path is not None:
                y, sr = librosa.load(audio_path, sr=sample_rate)
            else:
                return self._default_result()
            
            # 提取简单特征
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # 简单规则判断
            if energy_mean > 0.1 and energy_std > 0.05:
                emotion = "angry"
            elif energy_mean < 0.02:
                emotion = "sad"
            elif zcr_mean > 0.15:
                emotion = "happy"
            else:
                emotion = "neutral"
            
            emotion_idx = self.EMOTIONS.index(emotion)
            prosody = self._extract_prosody(audio_path, audio_array, sample_rate)
            
            return {
                "emotion": emotion,
                "emotion_zh": self.EMOTIONS_ZH[emotion_idx],
                "confidence": 0.6,
                "probabilities": {e: 0.1 for e in self.EMOTIONS},
                "prosody": prosody,
                "method": "fallback"
            }
            
        except Exception as e:
            return self._default_result(error=str(e))
    
    def _default_result(self, error: str = None) -> Dict:
        """返回默认结果"""
        result = {
            "emotion": "neutral",
            "emotion_zh": "中性",
            "confidence": 0.0,
            "probabilities": {emotion: 0.0 for emotion in self.EMOTIONS},
            "prosody": {}
        }
        if error:
            result["error"] = error
        return result
    
    def _extract_prosody(self, audio_path: str = None, audio_array: np.ndarray = None,
                        sample_rate: int = 16000) -> Dict:
        """提取韵律特征"""
        try:
            if audio_array is not None:
                y = audio_array
                sr = sample_rate
            elif audio_path is not None:
                y, sr = librosa.load(audio_path, sr=sample_rate)
            else:
                return {}
            
            prosody = {}
            
            # 音高特征
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = pitches[pitches > 0]
                if len(pitch_values) > 0:
                    prosody["pitch_mean"] = float(np.mean(pitch_values))
                    prosody["pitch_std"] = float(np.std(pitch_values))
                    prosody["pitch_range"] = float(np.max(pitch_values) - np.min(pitch_values))
            except:
                pass
            
            # 能量特征
            rms = librosa.feature.rms(y=y)[0]
            prosody["energy_mean"] = float(np.mean(rms))
            prosody["energy_std"] = float(np.std(rms))
            
            # 语速估计
            prosody["duration"] = float(len(y) / sr)
            
            return prosody
            
        except Exception as e:
            logger.error(f"Prosody extraction failed: {e}")
            return {}
    
    def analyze_batch(self, audio_files: List[str]) -> List[Dict]:
        """批量分析多个音频文件"""
        results = []
        for audio_file in audio_files:
            result = self.predict(audio_path=audio_file)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_type": "SenseVoice" if self.model else "Fallback (acoustic features)",
            "device": self.device,
            "emotions": self.EMOTIONS,
            "features": [
                "Multi-task: ASR + Emotion + Language detection",
                "Direct emotion label output",
                "Low latency",
                "Integrated with ASR pipeline"
            ]
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    emotion_module = EmotionModule(device="cpu")
    print("Model Info:", emotion_module.get_model_info())
    
    # 测试（需要提供测试音频）
    # result = emotion_module.predict("test_audio.wav")
    # print("Emotion:", result)
