"""
副语言特征分析模块 (Paralinguistic Analysis)
提取语音中的非语言信息：疼痛评估、焦虑检测、认知负荷等

基于 OpenSMILE eGeMAPS 特征集和声学分析
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# 尝试导入librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available")

# 尝试导入opensmile
try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    logger.warning("OpenSMILE not available, using fallback features")


class PainLevel(Enum):
    """疼痛等级"""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    EXTREME = 4


class AnxietyLevel(Enum):
    """焦虑等级"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class ProsodyFeatures:
    """韵律特征"""
    pitch_mean: float = 0.0      # 平均音高
    pitch_std: float = 0.0       # 音高标准差
    pitch_range: float = 0.0     # 音高范围
    jitter: float = 0.0          # 基频微扰
    shimmer: float = 0.0         # 振幅微扰
    hnr: float = 0.0             # 谐波噪声比
    speech_rate: float = 0.0     # 语速(音节/秒)
    pause_ratio: float = 0.0     # 停顿比例
    energy_mean: float = 0.0     # 平均能量
    energy_std: float = 0.0      # 能量标准差


class ParalinguisticAnalyzer:
    """
    副语言特征分析器
    
    功能:
    - 疼痛等级声学评估
    - 焦虑水平检测
    - 语义-声学不一致检测
    - 认知负荷评估
    - 情绪声学特征提取
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        初始化分析器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
        self.smile = None
        
        if OPENSMILE_AVAILABLE:
            try:
                # 使用eGeMAPS特征集 - 最适合情感/副语言分析
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
                logger.info("OpenSMILE initialized with eGeMAPS feature set")
            except Exception as e:
                logger.error(f"Failed to initialize OpenSMILE: {e}")
                self.smile = None
        
        logger.info(f"ParalinguisticAnalyzer initialized (OpenSMILE: {self.smile is not None})")
    
    def extract_prosody(
        self,
        audio_array: np.ndarray,
        sample_rate: int = None
    ) -> ProsodyFeatures:
        """
        提取韵律特征
        
        Args:
            audio_array: 音频数组
            sample_rate: 采样率
            
        Returns:
            ProsodyFeatures 韵律特征对象
        """
        sample_rate = sample_rate or self.sample_rate
        features = ProsodyFeatures()
        
        if not LIBROSA_AVAILABLE:
            return features
        
        try:
            # 确保音频是float格式
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / 32768.0
            
            # 1. 音高特征 (F0)
            pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sample_rate)
            pitch_values = pitches[pitches > 0]
            
            if len(pitch_values) > 0:
                features.pitch_mean = float(np.mean(pitch_values))
                features.pitch_std = float(np.std(pitch_values))
                features.pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
                
                # Jitter - 基频周期变化
                if len(pitch_values) > 1:
                    pitch_diff = np.diff(pitch_values)
                    features.jitter = float(np.mean(np.abs(pitch_diff)) / np.mean(pitch_values))
            
            # 2. 能量特征
            rms = librosa.feature.rms(y=audio_array)[0]
            features.energy_mean = float(np.mean(rms))
            features.energy_std = float(np.std(rms))
            
            # Shimmer - 振幅周期变化
            if len(rms) > 1:
                rms_diff = np.diff(rms)
                features.shimmer = float(np.mean(np.abs(rms_diff)) / np.mean(rms + 1e-6))
            
            # 3. 谐波噪声比 (HNR)
            try:
                harmonic, percussive = librosa.effects.hpss(audio_array)
                harmonic_energy = np.sum(harmonic ** 2)
                noise_energy = np.sum(percussive ** 2) + 1e-10
                features.hnr = float(10 * np.log10(harmonic_energy / noise_energy))
            except:
                features.hnr = 0.0
            
            # 4. 语速估计 (基于过零率和能量)
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            duration = len(audio_array) / sample_rate
            # 估计音节数 (简化方法)
            syllables = len(librosa.onset.onset_detect(y=audio_array, sr=sample_rate))
            features.speech_rate = float(syllables / duration) if duration > 0 else 0.0
            
            # 5. 停顿比例
            silence_threshold = 0.01
            silence_frames = np.sum(rms < silence_threshold)
            features.pause_ratio = float(silence_frames / len(rms)) if len(rms) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to extract prosody features: {e}")
        
        return features
    
    def extract_egemaps(self, audio_array: np.ndarray, sample_rate: int = None) -> Dict:
        """
        提取eGeMAPS特征集
        
        Returns:
            88维eGeMAPS特征字典
        """
        if self.smile is None:
            return {"error": "OpenSMILE not available", "features": {}}
        
        sample_rate = sample_rate or self.sample_rate
        
        try:
            # OpenSMILE需要特定格式
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / 32768.0
            
            # 提取特征
            features = self.smile.process_signal(audio_array, sample_rate)
            feature_dict = features.iloc[0].to_dict()
            
            return {
                "features": feature_dict,
                "feature_count": len(feature_dict),
            }
            
        except Exception as e:
            logger.error(f"Failed to extract eGeMAPS features: {e}")
            return {"error": str(e), "features": {}}
    
    def estimate_pain_level(
        self,
        audio_array: np.ndarray,
        sample_rate: int = None
    ) -> Dict:
        """
        估计声学疼痛等级 (1-10)
        
        基于研究发现:
        - 疼痛时F0升高、抖动增加
        - Jitter和Shimmer增大
        - 语速变慢、停顿增多
        - HNR降低（声音更"粗糙"）
        
        Returns:
            疼痛评估结果
        """
        prosody = self.extract_prosody(audio_array, sample_rate)
        
        # 疼痛指标权重
        pain_score = 0.0
        indicators = []
        
        # 1. Jitter异常 (正常 < 0.01)
        if prosody.jitter > 0.02:
            pain_score += 2.5
            indicators.append(f"声带抖动异常 (jitter={prosody.jitter:.3f})")
        elif prosody.jitter > 0.015:
            pain_score += 1.5
            indicators.append(f"声带轻微抖动 (jitter={prosody.jitter:.3f})")
        
        # 2. Shimmer异常 (正常 < 0.03)
        if prosody.shimmer > 0.06:
            pain_score += 2.0
            indicators.append(f"振幅不稳定 (shimmer={prosody.shimmer:.3f})")
        elif prosody.shimmer > 0.04:
            pain_score += 1.0
            indicators.append(f"振幅轻微波动 (shimmer={prosody.shimmer:.3f})")
        
        # 3. 语速过慢 (正常 3-5 音节/秒)
        if prosody.speech_rate < 2.0 and prosody.speech_rate > 0:
            pain_score += 1.5
            indicators.append(f"语速缓慢 ({prosody.speech_rate:.1f}音节/秒)")
        
        # 4. 停顿过多 (正常 < 0.3)
        if prosody.pause_ratio > 0.5:
            pain_score += 1.5
            indicators.append(f"停顿过多 ({prosody.pause_ratio:.1%})")
        elif prosody.pause_ratio > 0.4:
            pain_score += 0.8
        
        # 5. HNR过低 (正常 > 15dB)
        if prosody.hnr < 8:
            pain_score += 1.5
            indicators.append(f"声音嘶哑 (HNR={prosody.hnr:.1f}dB)")
        elif prosody.hnr < 12:
            pain_score += 0.8
        
        # 限制在1-10范围
        pain_level = min(10, max(1, int(pain_score + 1)))
        
        # 确定等级
        if pain_level <= 2:
            level_enum = PainLevel.NONE
            level_zh = "无明显疼痛"
        elif pain_level <= 4:
            level_enum = PainLevel.MILD
            level_zh = "轻度疼痛"
        elif pain_level <= 6:
            level_enum = PainLevel.MODERATE
            level_zh = "中度疼痛"
        elif pain_level <= 8:
            level_enum = PainLevel.SEVERE
            level_zh = "重度疼痛"
        else:
            level_enum = PainLevel.EXTREME
            level_zh = "剧烈疼痛"
        
        return {
            "pain_level": pain_level,
            "pain_level_enum": level_enum.name,
            "pain_level_zh": level_zh,
            "confidence": min(0.9, 0.5 + len(indicators) * 0.1),
            "indicators": indicators,
            "prosody": {
                "jitter": prosody.jitter,
                "shimmer": prosody.shimmer,
                "speech_rate": prosody.speech_rate,
                "pause_ratio": prosody.pause_ratio,
                "hnr": prosody.hnr,
            },
            "note": "声学评估仅供参考，需结合患者自述判断",
        }
    
    def detect_anxiety(
        self,
        audio_array: np.ndarray,
        sample_rate: int = None
    ) -> Dict:
        """
        检测焦虑水平
        
        焦虑声学特征:
        - 音高升高且不稳定
        - 语速加快
        - 能量波动大
        """
        prosody = self.extract_prosody(audio_array, sample_rate)
        
        anxiety_score = 0.0
        indicators = []
        
        # 1. 音高不稳定
        if prosody.pitch_std > 50:
            anxiety_score += 30
            indicators.append("音高波动大")
        
        # 2. 语速过快 (正常 3-5)
        if prosody.speech_rate > 6:
            anxiety_score += 25
            indicators.append("语速急促")
        
        # 3. 能量波动
        if prosody.energy_std > prosody.energy_mean * 0.5:
            anxiety_score += 25
            indicators.append("声音不稳")
        
        # 4. Jitter高
        if prosody.jitter > 0.015:
            anxiety_score += 20
            indicators.append("声带紧张")
        
        # 确定等级
        if anxiety_score < 30:
            level = AnxietyLevel.LOW
            level_zh = "低焦虑"
        elif anxiety_score < 60:
            level = AnxietyLevel.MODERATE
            level_zh = "中度焦虑"
        else:
            level = AnxietyLevel.HIGH
            level_zh = "高度焦虑"
        
        return {
            "anxiety_level": level.value,
            "anxiety_level_zh": level_zh,
            "score": anxiety_score,
            "indicators": indicators,
            "recommendation": "建议安抚情绪" if level != AnxietyLevel.LOW else None,
        }
    
    def detect_semantic_acoustic_mismatch(
        self,
        audio_array: np.ndarray,
        semantic_pain_level: int,
        sample_rate: int = None
    ) -> Dict:
        """
        检测语义-声学不一致
        
        当患者口头说"不痛"但声学特征显示高疼痛时，
        可能存在掩饰疼痛的情况
        
        Args:
            audio_array: 音频数组
            semantic_pain_level: 患者自述疼痛等级 (1-10)
            sample_rate: 采样率
            
        Returns:
            不一致检测结果
        """
        acoustic_pain = self.estimate_pain_level(audio_array, sample_rate)
        acoustic_level = acoustic_pain["pain_level"]
        
        diff = acoustic_level - semantic_pain_level
        
        mismatch_detected = abs(diff) >= 3
        
        if diff >= 4:
            mismatch_type = "underreport"
            message = "声学特征显示疼痛程度高于自述，患者可能低估或掩饰疼痛"
            priority = "attention"
        elif diff >= 3:
            mismatch_type = "slight_underreport"
            message = "声学特征略高于自述"
            priority = "note"
        elif diff <= -3:
            mismatch_type = "overreport"
            message = "自述疼痛程度高于声学特征"
            priority = "note"
        else:
            mismatch_type = "consistent"
            message = "自述与声学特征一致"
            priority = "normal"
        
        return {
            "mismatch_detected": mismatch_detected,
            "mismatch_type": mismatch_type,
            "semantic_pain": semantic_pain_level,
            "acoustic_pain": acoustic_level,
            "difference": diff,
            "message": message,
            "priority": priority,
            "acoustic_details": acoustic_pain,
        }
    
    def estimate_cognitive_load(
        self,
        audio_array: np.ndarray,
        sample_rate: int = None
    ) -> Dict:
        """
        估计认知负荷
        
        用于：
        - 老年MCI（轻度认知障碍）筛查
        - 检测困惑/理解困难
        
        认知负荷声学特征:
        - 语速降低
        - 停顿增多
        - 填充词增多（需配合ASR）
        - 重复和自我纠正
        """
        prosody = self.extract_prosody(audio_array, sample_rate)
        
        load_score = 0.0
        indicators = []
        
        # 1. 语速慢
        if prosody.speech_rate < 2.5 and prosody.speech_rate > 0:
            load_score += 30
            indicators.append("语速缓慢")
        
        # 2. 停顿多
        if prosody.pause_ratio > 0.4:
            load_score += 30
            indicators.append("频繁停顿")
        
        # 3. 音高不稳定（思考时音高变化）
        if prosody.pitch_std > 60:
            load_score += 20
            indicators.append("音调犹豫")
        
        # 确定等级
        if load_score < 30:
            level = "low"
            level_zh = "认知负荷低"
        elif load_score < 60:
            level = "moderate"
            level_zh = "认知负荷中等"
        else:
            level = "high"
            level_zh = "认知负荷高"
        
        return {
            "cognitive_load": level,
            "cognitive_load_zh": level_zh,
            "score": load_score,
            "indicators": indicators,
            "recommendation": "建议使用简单语言、放慢语速" if level == "high" else None,
            "mci_screening_note": "如持续出现高认知负荷，建议进行MCI筛查" if level == "high" else None,
        }
    
    def analyze(
        self,
        audio_array: np.ndarray,
        sample_rate: int = None,
        semantic_pain_level: int = None
    ) -> Dict:
        """
        综合副语言分析
        
        Args:
            audio_array: 音频数组
            sample_rate: 采样率
            semantic_pain_level: 患者自述疼痛等级（可选）
            
        Returns:
            综合分析结果
        """
        result = {
            "prosody": self.extract_prosody(audio_array, sample_rate).__dict__,
            "pain": self.estimate_pain_level(audio_array, sample_rate),
            "anxiety": self.detect_anxiety(audio_array, sample_rate),
            "cognitive_load": self.estimate_cognitive_load(audio_array, sample_rate),
        }
        
        if semantic_pain_level is not None:
            result["semantic_acoustic_mismatch"] = self.detect_semantic_acoustic_mismatch(
                audio_array, semantic_pain_level, sample_rate
            )
        
        # 综合建议
        recommendations = []
        
        if result["pain"]["pain_level"] >= 6:
            recommendations.append("声学特征显示较高疼痛水平")
        
        if result["anxiety"]["anxiety_level"] == "high":
            recommendations.append("检测到高焦虑，建议安抚")
        
        if result["cognitive_load"]["cognitive_load"] == "high":
            recommendations.append("认知负荷高，建议简化沟通")
        
        result["recommendations"] = recommendations
        result["priority"] = "urgent" if result["pain"]["pain_level"] >= 8 else "normal"
        
        return result
    
    def get_info(self) -> Dict:
        """获取模块信息"""
        return {
            "available": True,
            "opensmile_available": OPENSMILE_AVAILABLE,
            "librosa_available": LIBROSA_AVAILABLE,
            "sample_rate": self.sample_rate,
            "features": [
                "疼痛等级评估",
                "焦虑水平检测",
                "语义-声学不一致检测",
                "认知负荷评估",
                "韵律特征提取",
            ],
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    analyzer = ParalinguisticAnalyzer()
    print("Analyzer Info:", analyzer.get_info())
    
    # 生成测试音频
    test_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    
    result = analyzer.analyze(test_audio, semantic_pain_level=3)
    print("\nAnalysis Result:")
    print(f"Pain Level: {result['pain']['pain_level']} - {result['pain']['pain_level_zh']}")
    print(f"Anxiety: {result['anxiety']['anxiety_level_zh']}")
    print(f"Cognitive Load: {result['cognitive_load']['cognitive_load_zh']}")
