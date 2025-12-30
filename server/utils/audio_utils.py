"""
音频处理工具函数
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, sample_rate: int = 16000) -> tuple:
    """
    加载音频文件
    
    Args:
        audio_path: 音频文件路径
        sample_rate: 目标采样率
    
    Returns:
        (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise


def save_audio(audio_array: np.ndarray, output_path: str, sample_rate: int = 16000):
    """
    保存音频文件
    
    Args:
        audio_array: 音频数组
        output_path: 输出路径
        sample_rate: 采样率
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio_array, sample_rate)
        logger.info(f"Audio saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    重采样音频
    
    Args:
        audio: 音频数组
        orig_sr: 原始采样率
        target_sr: 目标采样率
    
    Returns:
        重采样后的音频
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception as e:
        logger.error(f"Failed to resample audio: {e}")
        raise


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    归一化音频
    
    Args:
        audio: 音频数组
    
    Returns:
        归一化后的音频
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def remove_silence(audio: np.ndarray, sample_rate: int = 16000, 
                   top_db: int = 20) -> np.ndarray:
    """
    移除静音部分
    
    Args:
        audio: 音频数组
        sample_rate: 采样率
        top_db: 静音阈值（分贝）
    
    Returns:
        移除静音后的音频
    """
    try:
        # 检测非静音区间
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        # 拼接非静音部分
        non_silent = []
        for start, end in intervals:
            non_silent.append(audio[start:end])
        
        if non_silent:
            return np.concatenate(non_silent)
        else:
            return audio
            
    except Exception as e:
        logger.error(f"Failed to remove silence: {e}")
        return audio


def extract_audio_features(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    提取音频基本特征
    
    Args:
        audio: 音频数组
        sample_rate: 采样率
    
    Returns:
        特征字典
    """
    features = {}
    
    try:
        # 持续时间
        features['duration'] = len(audio) / sample_rate
        
        # RMS能量
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 过零率
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        
        # 频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        
        return features
        
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        return features


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    将立体声转换为单声道
    
    Args:
        audio: 音频数组
    
    Returns:
        单声道音频
    """
    if len(audio.shape) > 1:
        return np.mean(audio, axis=1)
    return audio


def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """
    添加噪声（用于数据增强）
    
    Args:
        audio: 音频数组
        noise_level: 噪声级别
    
    Returns:
        添加噪声后的音频
    """
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    时间拉伸（改变语速但不改变音高）
    
    Args:
        audio: 音频数组
        rate: 拉伸率（>1加快，<1减慢）
    
    Returns:
        拉伸后的音频
    """
    try:
        return librosa.effects.time_stretch(audio, rate=rate)
    except Exception as e:
        logger.error(f"Failed to time stretch: {e}")
        return audio


def pitch_shift(audio: np.ndarray, sample_rate: int = 16000, 
                n_steps: float = 0) -> np.ndarray:
    """
    音高变换
    
    Args:
        audio: 音频数组
        sample_rate: 采样率
        n_steps: 半音数（正数升高，负数降低）
    
    Returns:
        变换后的音频
    """
    try:
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
    except Exception as e:
        logger.error(f"Failed to pitch shift: {e}")
        return audio


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试音频处理功能
    # audio, sr = load_audio("test.wav")
    # features = extract_audio_features(audio, sr)
    # print("Features:", features)
