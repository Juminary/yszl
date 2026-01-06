"""
声纹识别模块 (Speaker Recognition)
使用 3D-Speaker Cam++ 或后备方案实现说话人识别
"""

import torch
import numpy as np
import pickle
import logging
from typing import Dict, Optional, List
from pathlib import Path
import librosa

logger = logging.getLogger(__name__)

# 尝试导入 FunASR 的 Cam++ 模型
try:
    from funasr import AutoModel
    SPEAKER_MODEL_AVAILABLE = True
except ImportError:
    SPEAKER_MODEL_AVAILABLE = False
    logger.warning("FunASR not available for speaker recognition")


class SpeakerModule:
    """声纹识别模块 - 使用3D-Speaker Cam++"""
    
    def __init__(self, db_path: str = "data/speaker_db.pkl", 
                 threshold: float = 0.75, device: str = "cuda"):
        """
        初始化声纹识别模块
        
        Args:
            db_path: 说话人数据库路径
            threshold: 识别阈值（余弦相似度）
            device: 运行设备
        """
        # 设备选择
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.threshold = threshold
        self.db_path = Path(db_path)
        
        # 确保数据库目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self.model = None
        if SPEAKER_MODEL_AVAILABLE:
            logger.info("Loading Cam++ speaker verification model...")
            try:
                # 加载 Cam++ 模型 - 优先使用本地，不存在则下载到 server/models
                from modelscope import snapshot_download
                
                models_dir = Path(__file__).parent.parent.parent / "models" / "speaker"
                model_path = models_dir / "campplus"
                
                if model_path.exists():
                    logger.info(f"Loading local model from: {model_path}")
                    model_to_load = str(model_path)
                else:
                    models_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Downloading Cam++ from ModelScope...")
                    model_to_load = snapshot_download("iic/speech_campplus_sv_zh-cn_16k-common", cache_dir=str(models_dir))
                    logger.info(f"Downloaded to: {model_to_load}")
                
                self.model = AutoModel(
                    model=model_to_load,
                    device=self.device,
                    disable_update=True
                )
                logger.info("Cam++ model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Cam++ model: {e}")
                self.model = None
        else:
            logger.warning("Speaker model not available")
        
        # 加载说话人数据库
        self.speaker_db = self._load_database()
    
    def _load_database(self) -> Dict:
        """加载说话人数据库"""
        if self.db_path.exists():
            try:
                # 尝试使用文件锁（Linux/Unix系统）
                try:
                    import fcntl
                    use_lock = True
                except ImportError:
                    use_lock = False  # Windows系统不支持fcntl
                
                with open(self.db_path, 'rb') as f:
                    if use_lock:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # 获取共享锁
                            db = pickle.load(f)
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
                        except Exception:
                            # 如果锁操作失败，回退到直接加载
                            f.seek(0)  # 重置文件指针
                            db = pickle.load(f)
                    else:
                        db = pickle.load(f)
                logger.debug(f"Loaded speaker database with {len(db)} speakers")
                return db
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                return {}
        else:
            logger.debug("No existing database found. Creating new one.")
            return {}
    
    def _save_database(self):
        """保存说话人数据库"""
        try:
            # 尝试使用文件锁（Linux/Unix系统）
            try:
                import fcntl
                use_lock = True
            except ImportError:
                use_lock = False  # Windows系统不支持fcntl
            
            with open(self.db_path, 'wb') as f:
                if use_lock:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 获取排他锁
                        pickle.dump(self.speaker_db, f)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # 释放锁
                    except Exception:
                        # 如果锁操作失败，回退到直接保存
                        f.seek(0)  # 重置文件指针
                        pickle.dump(self.speaker_db, f)
                else:
                    pickle.dump(self.speaker_db, f)
            logger.info(f"Saved speaker database with {len(self.speaker_db)} speakers")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def extract_embedding(self, audio_path: str = None, 
                         audio_array: np.ndarray = None,
                         sample_rate: int = 16000) -> np.ndarray:
        """提取音频的声纹特征向量"""
        if self.model is None:
            # 使用简单的MFCC特征作为后备
            logger.debug("Model not available, using MFCC embedding")
            return self._extract_mfcc_embedding(audio_path, audio_array, sample_rate)
        
        try:
            # 准备输入
            if audio_array is not None:
                import tempfile
                import soundfile as sf
                if sample_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                temp_path = tempfile.mktemp(suffix=".wav")
                sf.write(temp_path, audio_array, 16000)
                input_path = temp_path
            else:
                input_path = audio_path
            
            # 使用 Cam++ 提取embedding
            result = self.model.generate(input=input_path)
            
            if result and len(result) > 0:
                embedding = result[0].get("spk_embedding", None)
                if embedding is not None:
                    # 转换 tensor 为 numpy
                    if hasattr(embedding, 'cpu'):
                        embedding = embedding.cpu().numpy()
                    embedding = np.array(embedding).flatten()
                    logger.debug(f"Extracted Cam++ embedding with shape: {embedding.shape}")
                    return embedding
                else:
                    logger.warning("No spk_embedding in result, falling back to MFCC")
                    return self._extract_mfcc_embedding(audio_path, audio_array, sample_rate)
            else:
                logger.warning("Empty result from model, falling back to MFCC")
                return self._extract_mfcc_embedding(audio_path, audio_array, sample_rate)
                
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}, falling back to MFCC")
            return self._extract_mfcc_embedding(audio_path, audio_array, sample_rate)
    
    def _extract_mfcc_embedding(self, audio_path: str = None, 
                                audio_array: np.ndarray = None,
                                sample_rate: int = 16000) -> np.ndarray:
        """后备方案：使用MFCC作为简单embedding"""
        try:
            if audio_array is not None:
                y = audio_array
                sr = sample_rate
            elif audio_path is not None:
                y, sr = librosa.load(audio_path, sr=16000)
            else:
                return np.zeros(39)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
            
            embedding = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta])
            return embedding
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            return np.zeros(39)
    
    def register_speaker(self, speaker_id: str, audio_path: str = None,
                        audio_array: np.ndarray = None, 
                        sample_rate: int = 16000,
                        metadata: Dict = None) -> Dict:
        """注册新说话人"""
        try:
            logger.info(f"Registering speaker {speaker_id}")
            embedding = self.extract_embedding(audio_path, audio_array, sample_rate)
            logger.info(f"Extracted embedding with shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
            
            if speaker_id in self.speaker_db:
                existing_embeddings = self.speaker_db[speaker_id]["embeddings"]
                existing_embeddings.append(embedding)
                if len(existing_embeddings) > 10:
                    existing_embeddings = existing_embeddings[-10:]
                self.speaker_db[speaker_id]["embeddings"] = existing_embeddings
                self.speaker_db[speaker_id]["mean_embedding"] = np.mean(existing_embeddings, axis=0)
                # 更新 metadata（如果提供了）
                if metadata:
                    if 'metadata' not in self.speaker_db[speaker_id]:
                        self.speaker_db[speaker_id]['metadata'] = {}
                    self.speaker_db[speaker_id]['metadata'].update(metadata)
                action = "updated"
                logger.info(f"Updated speaker {speaker_id} with {len(existing_embeddings)} samples")
            else:
                self.speaker_db[speaker_id] = {
                    "embeddings": [embedding],
                    "mean_embedding": embedding,
                    "metadata": metadata or {},
                    "audio_path": audio_path
                }
                action = "registered"
                logger.info(f"Registered new speaker {speaker_id}")
            
            self._save_database()
            # 重新加载数据库，确保内存中的数据与文件同步（解决并发问题）
            self.speaker_db = self._load_database()
            logger.info(f"Speaker {speaker_id} {action}, database saved with {len(self.speaker_db)} speakers")
            
            return {
                "status": "success",
                "action": action,
                "speaker_id": speaker_id,
                "num_samples": len(self.speaker_db[speaker_id]["embeddings"]),
                "embedding_dim": len(embedding)
            }
            
        except Exception as e:
            logger.error(f"Speaker registration failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def recognize_speaker(self, audio_path: str = None,
                         audio_array: np.ndarray = None,
                         sample_rate: int = 16000,
                         return_all: bool = False) -> Dict:
        """识别说话人"""
        try:
            # 重新加载数据库，确保使用最新的注册数据（解决并发问题）
            self.speaker_db = self._load_database()
            logger.info(f"Starting speaker recognition. Database has {len(self.speaker_db)} speakers")
            
            if not self.speaker_db:
                logger.warning("No speakers registered in database")
                return {
                    "speaker_id": "unknown",
                    "similarity": 0.0,
                    "confidence": 0.0,
                    "recognized": False,
                    "error": "No speakers registered"
                }
            
            embedding = self.extract_embedding(audio_path, audio_array, sample_rate)
            logger.info(f"Extracted embedding with shape: {embedding.shape}")
            
            similarities = {}
            dimension_mismatches = []
            for speaker_id, speaker_data in self.speaker_db.items():
                mean_embedding = speaker_data["mean_embedding"]
                # 确保维度匹配
                if len(embedding) != len(mean_embedding):
                    dimension_mismatches.append(f"{speaker_id}: {len(embedding)} vs {len(mean_embedding)}")
                    logger.warning(f"Dimension mismatch for {speaker_id}: embedding {len(embedding)} vs mean {len(mean_embedding)}")
                    continue
                similarity = self._cosine_similarity(embedding, mean_embedding)
                similarities[speaker_id] = float(similarity)
                logger.debug(f"Similarity with {speaker_id}: {similarity:.4f}")
            
            if dimension_mismatches:
                logger.warning(f"Dimension mismatches: {', '.join(dimension_mismatches)}")
            
            if not similarities:
                logger.warning("No valid similarities computed (all had dimension mismatches)")
                return {
                    "speaker_id": "unknown",
                    "similarity": 0.0,
                    "recognized": False,
                    "error": "Dimension mismatch with all registered speakers"
                }
            
            best_speaker = max(similarities, key=similarities.get)
            best_similarity = similarities[best_speaker]
            recognized = best_similarity >= self.threshold
            
            logger.info(f"Best match: {best_speaker} with similarity {best_similarity:.4f} (threshold: {self.threshold}, recognized: {recognized})")
            
            result = {
                "speaker_id": best_speaker if recognized else "unknown",
                "similarity": best_similarity,
                "confidence": best_similarity,
                "recognized": recognized,
                "threshold": self.threshold
            }
            
            if return_all:
                result["all_similarities"] = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            return result
            
        except Exception as e:
            logger.error(f"Speaker recognition failed: {e}", exc_info=True)
            return {"speaker_id": "unknown", "similarity": 0.0, "recognized": False, "error": str(e)}
    
    def verify_speaker(self, speaker_id: str, audio_path: str = None,
                      audio_array: np.ndarray = None,
                      sample_rate: int = 16000) -> Dict:
        """验证说话人身份"""
        try:
            if speaker_id not in self.speaker_db:
                return {"verified": False, "similarity": 0.0, "error": f"Speaker {speaker_id} not registered"}
            
            embedding = self.extract_embedding(audio_path, audio_array, sample_rate)
            mean_embedding = self.speaker_db[speaker_id]["mean_embedding"]
            
            if len(embedding) != len(mean_embedding):
                return {"verified": False, "similarity": 0.0, "error": "Embedding dimension mismatch"}
            
            similarity = self._cosine_similarity(embedding, mean_embedding)
            verified = similarity >= self.threshold
            
            return {
                "verified": verified,
                "similarity": float(similarity),
                "threshold": self.threshold,
                "speaker_id": speaker_id
            }
            
        except Exception as e:
            logger.error(f"Speaker verification failed: {e}")
            return {"verified": False, "similarity": 0.0, "error": str(e)}
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def delete_speaker(self, speaker_id: str) -> Dict:
        """删除说话人"""
        if speaker_id in self.speaker_db:
            del self.speaker_db[speaker_id]
            self._save_database()
            return {"status": "success", "speaker_id": speaker_id}
        return {"status": "error", "error": f"Speaker {speaker_id} not found"}
    
    def list_speakers(self) -> List[Dict]:
        """列出所有注册的说话人"""
        return [{"speaker_id": k, "num_samples": len(v["embeddings"]), "metadata": v.get("metadata", {})} 
                for k, v in self.speaker_db.items()]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_speakers": len(self.speaker_db),
            "threshold": self.threshold,
            "device": self.device,
            "database_path": str(self.db_path),
            "model_available": self.model is not None
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    speaker_module = SpeakerModule(device="cpu")
    print("Statistics:", speaker_module.get_statistics())
