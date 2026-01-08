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
            raise ImportError("SenseVoice not available. Please install with: pip install funasr")
        
        logger.info("Loading SenseVoice model...")
        # 加载SenseVoice模型 - 优先使用本地，不存在则下载到 server/models
        from modelscope import snapshot_download
        
        models_dir = Path(__file__).parent.parent.parent / "models" / "asr"
        model_path = models_dir / "SenseVoiceSmall"
        
        if model_path.exists():
            logger.info(f"Loading local model from: {model_path}")
            model_to_load = str(model_path)
        else:
            models_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading SenseVoice from ModelScope...")
            model_to_load = snapshot_download("iic/SenseVoiceSmall", cache_dir=str(models_dir))
            logger.info(f"Downloaded to: {model_to_load}")
        
        self.model = AutoModel(
            model=model_to_load,
            trust_remote_code=True,
            device=self.device,
            disable_update=True
        )
        logger.info("SenseVoice model loaded successfully")
    
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
                raw_emotion = result[0].get("emotion") or result[0].get("pred_emotion")

                # SenseVoice在文本中嵌入情感标签，如 "<|HAPPY|>你好"
                emotion_from_tag = self._parse_emotion_from_text(text_result)
                emotion = self.SENSEVOICE_EMOTION_MAP.get((raw_emotion or '').upper(), emotion_from_tag)
                emotion = emotion or "neutral"

                # 提取韵律特征
                prosody = self._extract_prosody(audio_path, audio_array, sample_rate)

                # 文本语义启发：明显的“开心/不开心/生气”词汇覆盖一档
                text_heur = self._heuristic_emotion_from_text(text_result)
                if text_heur:
                    emotion = text_heur["emotion"]
                    confidence = text_heur["confidence"]
                else:
                    confidence = 0.85 if emotion != "neutral" else 0.5

                # 基于韵律的辅助判别：若模型输出为neutral且韵律特征明显偏低/高，做轻微偏移
                heuristic = self._heuristic_emotion_from_prosody(prosody)
                if emotion == "neutral" and heuristic:
                    emotion = heuristic["emotion"]
                    confidence = heuristic["confidence"]

                emotion_zh = self.EMOTIONS_ZH[self.EMOTIONS.index(emotion)]
                return {
                    "emotion": emotion,
                    "emotion_zh": emotion_zh,
                    "confidence": confidence,
                    "probabilities": {e: (heuristic.get("probabilities", {}).get(e, 0.0) if heuristic else 0.0) for e in self.EMOTIONS},
                    "prosody": prosody,
                    "raw_text": text_result,
                    "raw_emotion": raw_emotion
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

    def _heuristic_emotion_from_prosody(self, prosody: Dict) -> Dict:
        """基于简单韵律阈值的补充情感判断，缓解过度输出neutral。
        能量/音高阈值是经验值，可按实际音频再微调。
        """
        try:
            energy = prosody.get("energy_mean", 0.0)
            pitch_mean = prosody.get("pitch_mean", 0.0)
            pitch_std = prosody.get("pitch_std", 0.0)

            # 经验阈值：低能量低音高 -> 悲伤；高能量高音高 -> 开心/愤怒。
            if energy < 0.012 and pitch_mean < 150:
                return {
                    "emotion": "sad",
                    "confidence": 0.6,
                    "probabilities": {"sad": 0.6, "neutral": 0.4}
                }
            if energy > 0.05 and pitch_mean > 200:
                # 区分开心/愤怒：音高波动较大视为开心，波动较小视为愤怒。
                if pitch_std >= 60:
                    return {
                        "emotion": "happy",
                        "confidence": 0.6,
                        "probabilities": {"happy": 0.6, "neutral": 0.3, "angry": 0.1}
                    }
                else:
                    return {
                        "emotion": "angry",
                        "confidence": 0.55,
                        "probabilities": {"angry": 0.55, "neutral": 0.35, "happy": 0.1}
                    }
            return {}
        except Exception as e:
            logger.debug(f"Heuristic emotion failed: {e}")
            return {}

    def _heuristic_emotion_from_text(self, text: str) -> Dict:
        """基于中文情感关键词的简单覆盖，优先解决明显的“不开心/生气/难过”等情况。"""
        try:
            t = (text or "").lower()
            # 悲伤/不开心
            sad_keys = ["不开心", "难过", "伤心", "沮丧", "低落", "压抑", "烦", "心情不好", "不高兴"]
            if any(k in t for k in sad_keys):
                return {"emotion": "sad", "confidence": 0.7, "probabilities": {"sad": 0.7, "neutral": 0.3}}

            # 开心
            happy_keys = ["开心", "高兴", "愉快", "不错", "好开心", "很爽"]
            if any(k in t for k in happy_keys):
                return {"emotion": "happy", "confidence": 0.7, "probabilities": {"happy": 0.7, "neutral": 0.3}}

            # 生气/烦躁
            angry_keys = ["生气", "火大", "气死", "很烦", "恼火"]
            if any(k in t for k in angry_keys):
                return {"emotion": "angry", "confidence": 0.65, "probabilities": {"angry": 0.65, "neutral": 0.35}}

            return {}
        except Exception as e:
            logger.debug(f"Text heuristic emotion failed: {e}")
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
            "model_type": "SenseVoice",
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
