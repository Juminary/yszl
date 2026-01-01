"""
声学事件检测模块 (Sound Event Detection)
使用 YAMNet 模型检测咳嗽、喘息、呻吟等医疗相关声音事件

支持事件类型:
- 咳嗽 (Cough) - 干咳/湿咳分类
- 喘息 (Wheeze/Stridor)  
- 呻吟 (Groan/Moan)
- 打喷嚏 (Sneeze)
- 清嗓 (Throat clearing)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入TensorFlow Lite
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Sound event detection disabled.")

# 尝试导入librosa用于音频处理
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available.")


class SoundEventDetector:
    """
    声学事件检测器 - 使用YAMNet模型
    
    YAMNet是Google开发的音频分类模型，支持521种音频类别
    包括咳嗽、喘息、呻吟等医疗相关声音
    """
    
    # YAMNet医疗相关类别ID映射
    MEDICAL_SOUND_CLASSES = {
        # 咳嗽相关
        "cough": [47],  # Cough
        
        # 呼吸相关
        "wheeze": [30, 31],  # Wheeze, Respiratory sounds
        "breathing": [28, 29],  # Breathing, Breath
        "snore": [32],  # Snore
        "gasp": [33],  # Gasp
        
        # 不适相关
        "groan": [24, 25],  # Groan, Grunt
        "cry": [21, 22],  # Crying, Sobbing
        "scream": [17],  # Scream
        
        # 其他
        "sneeze": [48],  # Sneeze
        "throat_clear": [49],  # Throat clearing
        "hiccup": [50],  # Hiccup
        
        # 语音相关（用于区分）
        "speech": [0, 1, 2, 3],  # Speech, Male/Female speech
    }
    
    # 中文标签映射
    LABEL_ZH = {
        "cough": "咳嗽",
        "wheeze": "喘息/哮鸣",
        "breathing": "呼吸声",
        "snore": "打鼾",
        "gasp": "喘气",
        "groan": "呻吟",
        "cry": "哭泣",
        "scream": "尖叫",
        "sneeze": "打喷嚏",
        "throat_clear": "清嗓",
        "hiccup": "打嗝",
        "speech": "语音",
    }
    
    def __init__(
        self,
        model_path: str = None,
        sample_rate: int = 16000,
        threshold: float = 0.3,
        device: str = "cpu"
    ):
        """
        初始化声学事件检测器
        
        Args:
            model_path: YAMNet模型路径（.tflite），None则使用TF Hub
            sample_rate: 采样率
            threshold: 检测阈值
            device: 运行设备
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.device = device
        self.model = None
        self.class_names = []
        
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available, SoundEventDetector disabled")
            return
        
        try:
            self._load_model(model_path)
            logger.info(f"SoundEventDetector initialized with {len(self.class_names)} classes")
        except Exception as e:
            logger.error(f"Failed to initialize SoundEventDetector: {e}")
    
    def _load_model(self, model_path: str = None):
        """加载YAMNet模型 - 优先使用本地ModelScope下载的模型"""
        import os
        
        # 优先检查本地模型路径
        local_model_paths = [
            model_path,
            os.path.join(os.path.dirname(__file__), '..', 'models', 'damo', 'audio_yamnet_tflite_zh-cn'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'yamnet'),
        ]
        
        for local_path in local_model_paths:
            if local_path and os.path.exists(local_path):
                # 尝试加载本地TFLite模型
                try:
                    tflite_path = os.path.join(local_path, 'yamnet.tflite')
                    if os.path.exists(tflite_path):
                        logger.info(f"Loading YAMNet from local: {tflite_path}")
                        self._load_tflite_model(tflite_path)
                        return
                except Exception as e:
                    logger.warning(f"Failed to load local TFLite model: {e}")
        
        # 回退到TensorFlow Hub
        try:
            import tensorflow_hub as hub
            
            logger.info("Loading YAMNet from TensorFlow Hub...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            # 加载类别名称
            class_map_path = self.model.class_map_path().numpy().decode('utf-8')
            import csv
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                self.class_names = [row['display_name'] for row in reader]
            
            logger.info(f"YAMNet loaded from TF Hub with {len(self.class_names)} classes")
            
        except ImportError:
            logger.warning("tensorflow_hub not available, using fallback detection")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YAMNet: {e}")
            self.model = None
    
    def _load_tflite_model(self, tflite_path: str):
        """加载TFLite格式的YAMNet模型"""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            self.model = interpreter
            self.model_type = "tflite"
            
            # 加载类别名称（从同目录下的label文件）
            import os
            label_path = os.path.join(os.path.dirname(tflite_path), 'yamnet_class_map.csv')
            if os.path.exists(label_path):
                import csv
                with open(label_path) as f:
                    reader = csv.DictReader(f)
                    self.class_names = [row.get('display_name', row.get('name', '')) for row in reader]
            else:
                # 使用默认的521个类别
                self.class_names = [f"class_{i}" for i in range(521)]
            
            logger.info(f"YAMNet TFLite loaded from {tflite_path}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise
    
    def detect(
        self,
        audio_array: np.ndarray = None,
        audio_path: str = None,
        sample_rate: int = None
    ) -> Dict:
        """
        检测音频中的声学事件
        
        Args:
            audio_array: 音频数组
            audio_path: 音频文件路径
            sample_rate: 采样率
            
        Returns:
            检测结果字典
        """
        if self.model is None:
            # 使用回退检测方法
            return self._fallback_detect(audio_array, audio_path, sample_rate)
        
        try:
            # 准备音频
            if audio_array is None and audio_path is not None:
                audio_array, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
            elif audio_array is not None:
                sample_rate = sample_rate or self.sample_rate
                if sample_rate != self.sample_rate:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=self.sample_rate)
            else:
                return self._empty_result()
            
            # 归一化
            audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / 32768.0  # int16 -> float
            
            # YAMNet推理
            scores, embeddings, spectrogram = self.model(audio_array)
            scores = scores.numpy()
            
            # 取平均分数
            avg_scores = np.mean(scores, axis=0)
            
            # 提取医疗相关事件
            events = {}
            for event_name, class_ids in self.MEDICAL_SOUND_CLASSES.items():
                if class_ids:
                    event_score = float(np.max([avg_scores[i] for i in class_ids if i < len(avg_scores)]))
                    if event_score >= self.threshold:
                        events[event_name] = event_score
            
            # 检测最显著的事件
            top_events = sorted(events.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                "events": dict(top_events),
                "events_zh": {self.LABEL_ZH.get(k, k): v for k, v in top_events},
                "cough_detected": events.get("cough", 0) >= self.threshold,
                "wheeze_detected": events.get("wheeze", 0) >= self.threshold,
                "respiratory_distress": self._check_respiratory_distress(events),
                "raw_top5": self._get_top5_classes(avg_scores),
            }
            
        except Exception as e:
            logger.error(f"Sound event detection failed: {e}")
            return self._empty_result(error=str(e))
    
    def _fallback_detect(
        self,
        audio_array: np.ndarray = None,
        audio_path: str = None,
        sample_rate: int = None
    ) -> Dict:
        """
        回退检测方法 - 基于频谱分析
        当YAMNet不可用时使用
        """
        try:
            # 准备音频
            if audio_array is None and audio_path is not None:
                if LIBROSA_AVAILABLE:
                    audio_array, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
                else:
                    return self._empty_result(error="No audio processing library available")
            
            if audio_array is None:
                return self._empty_result()
            
            sample_rate = sample_rate or self.sample_rate
            
            # 基于频谱的简单检测
            events = {}
            
            if LIBROSA_AVAILABLE:
                # 计算梅尔频谱
                mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
                mel_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 咳嗽检测 - 短时高能量爆发
                energy = np.mean(librosa.feature.rms(y=audio_array))
                onset_strength = librosa.onset.onset_strength(y=audio_array, sr=sample_rate)
                
                if np.max(onset_strength) > 50 and energy > 0.02:
                    # 分析高频和低频能量比来判断干咳/湿咳
                    high_freq_energy = np.mean(mel_db[64:, :])
                    low_freq_energy = np.mean(mel_db[:32, :])
                    
                    if high_freq_energy > low_freq_energy:
                        events["cough"] = 0.7
                        events["cough_type"] = "dry"
                    else:
                        events["cough"] = 0.65
                        events["cough_type"] = "wet"
                
                # 喘息检测 - 持续性高频能量
                high_freq = np.mean(mel_db[80:, :])
                if high_freq > -30:  # 高频能量较强
                    zcr = np.mean(librosa.feature.zero_crossing_rate(audio_array))
                    if zcr > 0.1:  # 高过零率
                        events["wheeze"] = 0.6
            
            return {
                "events": events,
                "events_zh": {self.LABEL_ZH.get(k, k): v for k, v in events.items() if isinstance(v, float)},
                "cough_detected": events.get("cough", 0) >= self.threshold,
                "wheeze_detected": events.get("wheeze", 0) >= self.threshold,
                "cough_type": events.get("cough_type"),
                "respiratory_distress": self._check_respiratory_distress(events),
                "method": "fallback_spectral",
            }
            
        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            return self._empty_result(error=str(e))
    
    def analyze_cough(self, audio_array: np.ndarray, sample_rate: int = None) -> Dict:
        """
        详细分析咳嗽特征
        
        Returns:
            咳嗽分析结果: {
                "detected": bool,
                "type": "dry" | "wet" | "unknown",
                "severity": "mild" | "moderate" | "severe",
                "frequency": int  # 每分钟咳嗽次数估计
            }
        """
        sample_rate = sample_rate or self.sample_rate
        
        if not LIBROSA_AVAILABLE:
            return {"detected": False, "error": "librosa not available"}
        
        try:
            # 首先检测是否有咳嗽
            detection = self.detect(audio_array=audio_array, sample_rate=sample_rate)
            
            if not detection.get("cough_detected"):
                return {"detected": False}
            
            # 计算梅尔频谱
            mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 分析高频/低频能量比 - 判断干咳/湿咳
            high_freq_energy = np.mean(mel_db[64:, :])
            low_freq_energy = np.mean(mel_db[:32, :])
            ratio = high_freq_energy / (low_freq_energy + 1e-6)
            
            if ratio > 0.8:
                cough_type = "dry"  # 干咳 - 高频能量强
            else:
                cough_type = "wet"  # 湿咳 - 低频能量强
            
            # 计算能量强度 - 判断严重程度
            rms = np.mean(librosa.feature.rms(y=audio_array))
            if rms > 0.1:
                severity = "severe"
            elif rms > 0.05:
                severity = "moderate"
            else:
                severity = "mild"
            
            # 检测咳嗽次数（基于onset检测）
            onsets = librosa.onset.onset_detect(y=audio_array, sr=sample_rate)
            duration = len(audio_array) / sample_rate
            frequency = int(len(onsets) / duration * 60) if duration > 0 else 0
            
            return {
                "detected": True,
                "type": cough_type,
                "type_zh": "干咳" if cough_type == "dry" else "湿咳",
                "severity": severity,
                "severity_zh": {"mild": "轻度", "moderate": "中度", "severe": "重度"}[severity],
                "frequency_per_min": frequency,
                "confidence": detection.get("events", {}).get("cough", 0.5),
            }
            
        except Exception as e:
            logger.error(f"Cough analysis failed: {e}")
            return {"detected": False, "error": str(e)}
    
    def is_respiratory_distress(self, audio_array: np.ndarray, sample_rate: int = None) -> Tuple[bool, Dict]:
        """
        检测是否存在呼吸窘迫迹象
        
        Returns:
            (是否呼吸窘迫, 详细信息)
        """
        detection = self.detect(audio_array=audio_array, sample_rate=sample_rate)
        
        distress_indicators = []
        
        if detection.get("wheeze_detected"):
            distress_indicators.append("哮鸣音")
        
        if detection.get("events", {}).get("gasp", 0) > self.threshold:
            distress_indicators.append("喘气")
        
        if detection.get("events", {}).get("stridor", 0) > self.threshold:
            distress_indicators.append("喘鸣")
        
        is_distress = len(distress_indicators) >= 1
        
        return is_distress, {
            "is_distress": is_distress,
            "indicators": distress_indicators,
            "priority": "urgent" if is_distress else "normal",
            "recommendation": "建议紧急就诊呼吸科" if is_distress else None,
        }
    
    def _check_respiratory_distress(self, events: Dict) -> bool:
        """检查是否有呼吸窘迫迹象"""
        distress_events = ["wheeze", "gasp", "stridor"]
        return any(events.get(e, 0) >= self.threshold for e in distress_events)
    
    def _get_top5_classes(self, scores: np.ndarray) -> List[Dict]:
        """获取前5个类别"""
        top_indices = np.argsort(scores)[-5:][::-1]
        return [
            {"class": self.class_names[i] if i < len(self.class_names) else f"class_{i}", 
             "score": float(scores[i])}
            for i in top_indices
        ]
    
    def _empty_result(self, error: str = None) -> Dict:
        """返回空结果"""
        result = {
            "events": {},
            "events_zh": {},
            "cough_detected": False,
            "wheeze_detected": False,
            "respiratory_distress": False,
        }
        if error:
            result["error"] = error
        return result
    
    def get_info(self) -> Dict:
        """获取模块信息"""
        return {
            "available": self.model is not None or LIBROSA_AVAILABLE,
            "model_type": "YAMNet" if self.model else "Fallback (Spectral)",
            "sample_rate": self.sample_rate,
            "threshold": self.threshold,
            "supported_events": list(self.MEDICAL_SOUND_CLASSES.keys()),
            "supported_events_zh": list(self.LABEL_ZH.values()),
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    detector = SoundEventDetector()
    print("Detector Info:", detector.get_info())
    
    # 如果有测试音频
    # result = detector.detect(audio_path="test_cough.wav")
    # print("Detection Result:", result)
