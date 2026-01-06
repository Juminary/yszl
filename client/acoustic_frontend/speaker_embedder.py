"""
声纹嵌入提取器 (Speaker Embedding Extractor)

使用轻量级模型提取说话人声纹特征向量，用于:
1. 说话人识别/验证
2. 与DOA空间信息融合进行说话人分离

支持的模型:
- ECAPA-TDNN (推荐): 高精度
- CAM++: ModelScope中文优化版
- 自定义ONNX模型
"""

import numpy as np
import logging
import time
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """声纹提取配置"""
    sample_rate: int = 16000
    embedding_dim: int = 192          # 输出向量维度
    min_duration_sec: float = 0.5     # 最短有效时长
    max_duration_sec: float = 10.0    # 最长处理时长
    normalize: bool = True            # 是否L2归一化
    
    # 模型配置
    model_type: str = "ecapa"         # ecapa, cam++, onnx
    model_path: Optional[str] = None  # ONNX模型路径
    device: str = "cpu"               # cpu, cuda


@dataclass
class SpeakerEmbedding:
    """说话人嵌入向量"""
    vector: np.ndarray               # 嵌入向量
    speaker_id: Optional[str] = None # 说话人ID (如已知)
    confidence: float = 1.0          # 置信度
    duration: float = 0.0            # 音频时长
    timestamp: float = None          # 提取时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def cosine_similarity(self, other: 'SpeakerEmbedding') -> float:
        """计算与另一个嵌入的余弦相似度"""
        if self.vector is None or other.vector is None:
            return 0.0
        
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        
        return float(np.dot(self.vector, other.vector) / (norm_a * norm_b))


class SpeakerEmbedder:
    """
    声纹嵌入提取器
    
    从音频中提取说话人特征向量，支持多种后端
    
    使用示例:
    ```python
    embedder = SpeakerEmbedder()
    
    # 提取单个嵌入
    embedding = embedder.extract(audio_data)
    
    # 比较两个说话人
    sim = embedding1.cosine_similarity(embedding2)
    if sim > 0.7:
        print("Same speaker")
    ```
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        model_type: str = "ecapa",
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        初始化声纹提取器
        
        Args:
            sample_rate: 采样率
            model_type: 模型类型 (ecapa, cam++, onnx, dummy)
            model_path: ONNX模型路径 (model_type=onnx时使用)
            device: 运行设备
        """
        self.config = EmbeddingConfig(
            sample_rate=sample_rate,
            model_type=model_type,
            model_path=model_path,
            device=device,
        )
        
        self._model = None
        self._session = None
        self._initialized = False
        
        # 尝试初始化模型
        self._init_model()
        
        logger.info(f"SpeakerEmbedder initialized: type={model_type}, "
                   f"dim={self.config.embedding_dim}, initialized={self._initialized}")
    
    def _init_model(self):
        """初始化模型"""
        model_type = self.config.model_type.lower()
        
        if model_type == "onnx" and self.config.model_path:
            self._init_onnx_model()
        elif model_type == "ecapa":
            self._init_ecapa_model()
        elif model_type == "cam++":
            self._init_campp_model()
        elif model_type == "dummy":
            # 测试用：随机嵌入
            self._initialized = True
            logger.info("Using dummy embedder (random vectors)")
        else:
            logger.warning(f"Unknown model type: {model_type}, using dummy")
            self._initialized = True
    
    def _init_onnx_model(self):
        """初始化ONNX模型"""
        try:
            import onnxruntime as ort
            
            model_path = self.config.model_path
            if not Path(model_path).exists():
                logger.error(f"ONNX model not found: {model_path}")
                return
            
            providers = ['CPUExecutionProvider']
            if self.config.device == "cuda":
                providers.insert(0, 'CUDAExecutionProvider')
            
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._initialized = True
            
            # 获取输出维度
            output_info = self._session.get_outputs()[0]
            self.config.embedding_dim = output_info.shape[-1]
            
            logger.info(f"ONNX model loaded: {model_path}")
            
        except ImportError:
            logger.warning("onnxruntime not installed")
        except Exception as e:
            logger.error(f"ONNX model init failed: {e}")
    
    def _init_ecapa_model(self):
        """初始化ECAPA-TDNN模型 (通过speechbrain或modelscope)"""
        try:
            # 尝试使用ModelScope的ECAPA模型
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            self._model = pipeline(
                task=Tasks.speaker_verification,
                model='iic/speech_ecapa-tdnn_sv_zh-cn_16k-common',
                device=self.config.device,
            )
            self._initialized = True
            self.config.embedding_dim = 192
            
            logger.info("ECAPA-TDNN model loaded via ModelScope")
            
        except ImportError:
            logger.warning("ModelScope not available, trying alternative")
            self._init_dummy_model()
        except Exception as e:
            logger.warning(f"ECAPA init failed: {e}")
            self._init_dummy_model()
    
    def _init_campp_model(self):
        """初始化CAM++模型"""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            self._model = pipeline(
                task=Tasks.speaker_verification,
                model='iic/speech_campplus_sv_zh-cn_16k-common',
                device=self.config.device,
            )
            self._initialized = True
            self.config.embedding_dim = 192
            
            logger.info("CAM++ model loaded via ModelScope")
            
        except ImportError:
            logger.warning("ModelScope not available")
            self._init_dummy_model()
        except Exception as e:
            logger.warning(f"CAM++ init failed: {e}")
            self._init_dummy_model()
    
    def _init_dummy_model(self):
        """初始化假模型（用于测试）"""
        self._initialized = True
        logger.info("Using dummy embedding model")
    
    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
    ) -> SpeakerEmbedding:
        """
        从音频中提取声纹嵌入
        
        Args:
            audio: 音频数据 (float32, 归一化到[-1,1])
            sample_rate: 采样率 (默认使用配置值)
            
        Returns:
            SpeakerEmbedding 对象
        """
        if not self._initialized:
            logger.warning("Embedder not initialized")
            return self._empty_embedding()
        
        sample_rate = sample_rate or self.config.sample_rate
        
        # 预处理
        audio = self._preprocess(audio, sample_rate)
        if audio is None:
            return self._empty_embedding()
        
        duration = len(audio) / sample_rate
        
        # 提取嵌入
        vector = self._extract_vector(audio)
        
        # L2归一化
        if self.config.normalize and vector is not None:
            norm = np.linalg.norm(vector)
            if norm > 1e-6:
                vector = vector / norm
        
        return SpeakerEmbedding(
            vector=vector,
            confidence=1.0 if vector is not None else 0.0,
            duration=duration,
        )
    
    def _preprocess(self, audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """预处理音频"""
        # 确保是float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=-1)
        
        # 归一化
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        # 检查时长
        duration = len(audio) / sample_rate
        if duration < self.config.min_duration_sec:
            logger.debug(f"Audio too short: {duration:.2f}s")
            return None
        
        # 截断过长音频
        if duration > self.config.max_duration_sec:
            max_samples = int(self.config.max_duration_sec * sample_rate)
            audio = audio[:max_samples]
        
        # 重采样 (如需)
        if sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate, self.config.sample_rate)
        
        return audio
    
    def _resample(self, audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
        """重采样"""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)
        except ImportError:
            # 简单的线性插值
            ratio = tgt_sr / src_sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def _extract_vector(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """提取嵌入向量"""
        # ONNX 模型
        if self._session is not None:
            try:
                input_name = self._session.get_inputs()[0].name
                output_name = self._session.get_outputs()[0].name
                
                # 添加batch维度
                audio_input = audio.reshape(1, -1)
                
                result = self._session.run([output_name], {input_name: audio_input})[0]
                return result.squeeze()
                
            except Exception as e:
                logger.error(f"ONNX inference failed: {e}")
                return None
        
        # ModelScope 模型
        if self._model is not None:
            try:
                # ModelScope pipeline期望输入为音频路径或numpy数组
                result = self._model(audio)
                
                # 提取embedding
                if isinstance(result, dict) and 'spk_embedding' in result:
                    return np.array(result['spk_embedding'])
                elif hasattr(result, 'spk_embedding'):
                    return np.array(result.spk_embedding)
                else:
                    return np.array(result)
                    
            except Exception as e:
                logger.error(f"ModelScope inference failed: {e}")
                return None
        
        # Dummy模型：生成基于音频特征的伪嵌入
        return self._generate_dummy_embedding(audio)
    
    def _generate_dummy_embedding(self, audio: np.ndarray) -> np.ndarray:
        """生成伪嵌入向量（基于简单的声学特征）"""
        # 使用简单的统计特征创建一个"假"嵌入
        # 在真实场景中应该使用深度学习模型
        
        dim = self.config.embedding_dim
        
        # 基于音频特征生成向量
        features = []
        
        # 能量
        energy = np.sqrt(np.mean(audio ** 2))
        features.append(energy)
        
        # 过零率
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)
        
        # 频谱质心 (简化版)
        fft = np.abs(np.fft.rfft(audio[:1024]))
        centroid = np.sum(fft * np.arange(len(fft))) / (np.sum(fft) + 1e-6)
        features.append(centroid / len(fft))
        
        # 基于哈希扩展到目标维度
        base = np.array(features)
        expanded = np.tile(base, dim // len(features) + 1)[:dim]
        
        # 添加噪声使不同音频产生不同嵌入
        noise = np.random.randn(dim) * 0.1
        noise_seed = int(np.sum(audio[:100]) * 1000) % 10000
        np.random.seed(noise_seed)
        
        return expanded + noise
    
    def _empty_embedding(self) -> SpeakerEmbedding:
        """返回空嵌入"""
        return SpeakerEmbedding(
            vector=np.zeros(self.config.embedding_dim, dtype=np.float32),
            confidence=0.0,
        )
    
    def compare(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
    ) -> float:
        """
        比较两段音频是否来自同一说话人
        
        Args:
            audio1: 第一段音频
            audio2: 第二段音频
            
        Returns:
            相似度分数 (0-1)
        """
        emb1 = self.extract(audio1)
        emb2 = self.extract(audio2)
        return emb1.cosine_similarity(emb2)
    
    def is_same_speaker(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        threshold: float = 0.7,
    ) -> bool:
        """判断两段音频是否来自同一说话人"""
        return self.compare(audio1, audio2) > threshold
    
    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.config.embedding_dim


# 便捷函数
def create_embedder(
    model_type: str = "dummy",
    **kwargs
) -> SpeakerEmbedder:
    """创建声纹提取器"""
    return SpeakerEmbedder(model_type=model_type, **kwargs)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Speaker Embedder...")
    
    embedder = SpeakerEmbedder(model_type="dummy")
    print(f"Embedding dim: {embedder.embedding_dim}")
    
    # 生成测试音频
    duration = 2.0
    t = np.linspace(0, duration, int(16000 * duration))
    audio1 = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    audio2 = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    
    emb1 = embedder.extract(audio1)
    emb2 = embedder.extract(audio2)
    
    print(f"Embedding 1: shape={emb1.vector.shape}, conf={emb1.confidence}")
    print(f"Embedding 2: shape={emb2.vector.shape}, conf={emb2.confidence}")
    print(f"Similarity: {emb1.cosine_similarity(emb2):.4f}")
