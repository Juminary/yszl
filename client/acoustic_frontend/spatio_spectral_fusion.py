"""
空间-声纹融合模块 (Spatio-Spectral Fusion)

创新性地融合两种说话人特征:
1. 空间特征 (DOA): 从麦克风阵列获取的声源方向
2. 频谱特征 (Embedding): 从语音中提取的声纹向量

解决传统纯声纹方法的问题:
- 短语音特征不稳定
- 多人重叠时无法区分
- 相似声纹难以分离

核心算法:
D_total = α * D_spatial + (1-α) * D_spectral
α 根据信噪比和语音长度动态调整
"""

import numpy as np
import logging
import time
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ClusterState(Enum):
    """簇状态"""
    ACTIVE = "active"        # 活跃中
    INACTIVE = "inactive"    # 暂时不活跃  
    EXPIRED = "expired"      # 已过期


@dataclass
class SpeakerCluster:
    """
    说话人簇
    
    融合空间和声纹信息的说话人表示
    """
    speaker_id: str                              # 说话人ID
    
    # 空间信息
    doa_center: float                            # DOA角度中心 (0-360)
    doa_variance: float = 15.0                   # 角度方差
    doa_history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # 声纹信息
    embedding_center: np.ndarray = None          # 声纹中心向量
    embedding_samples: int = 0                   # 累计样本数
    
    # 状态
    state: ClusterState = ClusterState.ACTIVE
    last_active: float = 0.0                     # 最后活跃时间
    total_duration: float = 0.0                  # 累计时长
    utterance_count: int = 0                     # 发言次数
    
    # 元数据
    role: str = "unknown"                        # 角色 (doctor/patient/family)
    name: str = None                             # 姓名 (如已知)
    
    def __post_init__(self):
        self.last_active = time.time()
        if self.embedding_center is None:
            self.embedding_center = np.zeros(192)
    
    def update_doa(self, angle: float):
        """更新DOA信息"""
        self.doa_history.append(angle)
        
        # 计算循环平均 (处理0/360度边界)
        angles_rad = np.radians(list(self.doa_history))
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        self.doa_center = np.degrees(np.arctan2(mean_sin, mean_cos))
        if self.doa_center < 0:
            self.doa_center += 360
        
        # 更新方差
        if len(self.doa_history) > 1:
            diffs = []
            for a in self.doa_history:
                diff = abs(a - self.doa_center)
                if diff > 180:
                    diff = 360 - diff
                diffs.append(diff)
            self.doa_variance = np.std(diffs) + 5  # 最小5度方差
        
        self.last_active = time.time()
    
    def update_embedding(self, embedding: np.ndarray, weight: float = 0.3):
        """
        更新声纹中心 (指数移动平均)
        
        Args:
            embedding: 新的声纹向量
            weight: 新样本权重
        """
        if self.embedding_samples == 0:
            self.embedding_center = embedding.copy()
        else:
            # EMA更新
            self.embedding_center = (
                (1 - weight) * self.embedding_center + 
                weight * embedding
            )
        
        # L2归一化
        norm = np.linalg.norm(self.embedding_center)
        if norm > 1e-6:
            self.embedding_center = self.embedding_center / norm
        
        self.embedding_samples += 1
        self.last_active = time.time()
    
    def doa_distance(self, angle: float) -> float:
        """计算角度距离 (考虑循环)"""
        diff = abs(angle - self.doa_center)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def embedding_distance(self, embedding: np.ndarray) -> float:
        """计算声纹余弦距离 (1 - similarity)"""
        if embedding is None or self.embedding_center is None:
            return 1.0
        
        norm_a = np.linalg.norm(self.embedding_center)
        norm_b = np.linalg.norm(embedding)
        
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 1.0
        
        similarity = np.dot(self.embedding_center, embedding) / (norm_a * norm_b)
        return 1.0 - max(0, min(1, similarity))
    
    def to_dict(self) -> Dict:
        return {
            "speaker_id": self.speaker_id,
            "doa_center": round(self.doa_center, 1),
            "doa_variance": round(self.doa_variance, 1),
            "embedding_samples": self.embedding_samples,
            "state": self.state.value,
            "role": self.role,
            "utterance_count": self.utterance_count,
            "total_duration": round(self.total_duration, 2),
        }


@dataclass
class FusionConfig:
    """融合配置"""
    # 权重参数
    alpha_base: float = 0.5              # 基础空间权重
    alpha_min: float = 0.2               # 最小空间权重 (长语音)
    alpha_max: float = 0.8               # 最大空间权重 (短语音)
    
    # 距离阈值
    doa_threshold: float = 30.0          # DOA匹配阈值 (度)
    embedding_threshold: float = 0.4     # 声纹距离阈值
    new_speaker_threshold: float = 0.6   # 新说话人判定阈值
    
    # 动态调整参数
    short_duration_sec: float = 0.5      # 短语音定义
    long_duration_sec: float = 3.0       # 长语音定义
    
    # 簇管理
    max_speakers: int = 6                # 最大说话人数
    inactive_timeout_sec: float = 30.0   # 不活跃超时
    expire_timeout_sec: float = 300.0    # 过期超时


class SpatioSpectralFusion:
    """
    空间-声纹融合说话人分离器
    
    核心创新:
    1. 融合DOA空间特征和声纹频谱特征
    2. 动态调整融合权重
    3. 在线增量聚类
    
    使用示例:
    ```python
    fusion = SpatioSpectralFusion()
    embedder = SpeakerEmbedder()
    
    # 处理音频帧
    doa = doa_estimator.estimate(audio)
    embedding = embedder.extract(audio)
    
    # 分配说话人
    speaker_id = fusion.assign_speaker(
        doa=doa.angle,
        embedding=embedding.vector,
        duration=len(audio) / 16000
    )
    
    print(f"Speaker: {speaker_id}")
    ```
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        max_speakers: int = 6,
    ):
        """
        初始化融合器
        
        Args:
            alpha: 空间权重 (0=纯声纹, 1=纯空间)
            max_speakers: 最大说话人数
        """
        self.config = FusionConfig(
            alpha_base=alpha,
            max_speakers=max_speakers,
        )
        
        # 说话人簇
        self._clusters: Dict[str, SpeakerCluster] = {}
        self._next_speaker_id = 1
        
        # 回调
        self._callbacks: Dict[str, List[Callable]] = {
            "on_new_speaker": [],
            "on_speaker_expired": [],
        }
        
        # 统计
        self._total_assignments = 0
        
        logger.info(f"SpatioSpectralFusion initialized: alpha={alpha}, max={max_speakers}")
    
    def assign_speaker(
        self,
        doa: float,
        embedding: np.ndarray = None,
        duration: float = 0.0,
        energy: float = 1.0,
    ) -> str:
        """
        分配说话人ID
        
        核心算法:
        1. 计算与各簇的融合距离
        2. 动态调整α权重
        3. 匹配或创建新簇
        
        Args:
            doa: 声源方向角度 (0-360)
            embedding: 声纹向量 (可选)
            duration: 语音时长 (秒)
            energy: 语音能量 (0-1)
            
        Returns:
            说话人ID
        """
        # 清理过期簇
        self._cleanup_expired()
        
        # 计算动态α
        alpha = self._compute_alpha(duration, energy, embedding is not None)
        
        # 如果没有簇，创建第一个
        if not self._clusters:
            return self._create_new_speaker(doa, embedding)
        
        # 计算与所有簇的融合距离
        distances = {}
        for speaker_id, cluster in self._clusters.items():
            if cluster.state == ClusterState.EXPIRED:
                continue
            
            # 空间距离 (归一化到0-1)
            d_spatial = cluster.doa_distance(doa) / 180.0
            
            # 声纹距离
            if embedding is not None and cluster.embedding_samples > 0:
                d_spectral = cluster.embedding_distance(embedding)
            else:
                d_spectral = 0.5  # 无声纹时使用中性距离
            
            # 融合距离
            d_total = alpha * d_spatial + (1 - alpha) * d_spectral
            distances[speaker_id] = {
                "total": d_total,
                "spatial": d_spatial,
                "spectral": d_spectral,
            }
        
        # 找最小距离
        best_speaker = min(distances, key=lambda x: distances[x]["total"])
        best_distance = distances[best_speaker]["total"]
        
        # 判断是否匹配
        if best_distance < self.config.new_speaker_threshold:
            # 匹配现有说话人
            self._update_speaker(best_speaker, doa, embedding, duration)
            return best_speaker
        else:
            # 创建新说话人
            if len(self._clusters) < self.config.max_speakers:
                return self._create_new_speaker(doa, embedding)
            else:
                # 达到上限，强制分配到最近的
                logger.debug("Max speakers reached, forcing assignment")
                self._update_speaker(best_speaker, doa, embedding, duration)
                return best_speaker
    
    def _compute_alpha(
        self,
        duration: float,
        energy: float,
        has_embedding: bool,
    ) -> float:
        """
        计算动态融合权重
        
        规则:
        - 短语音 → 增大α (更依赖空间)
        - 长语音 → 减小α (更依赖声纹)
        - 无声纹 → α=1.0 (纯空间)
        """
        if not has_embedding:
            return 1.0
        
        # 基于时长调整
        if duration < self.config.short_duration_sec:
            # 短语音：增大空间权重
            alpha = self.config.alpha_max
        elif duration > self.config.long_duration_sec:
            # 长语音：减小空间权重
            alpha = self.config.alpha_min
        else:
            # 线性插值
            ratio = (duration - self.config.short_duration_sec) / (
                self.config.long_duration_sec - self.config.short_duration_sec
            )
            alpha = self.config.alpha_max - ratio * (
                self.config.alpha_max - self.config.alpha_min
            )
        
        return alpha
    
    def _create_new_speaker(
        self,
        doa: float,
        embedding: np.ndarray = None,
    ) -> str:
        """创建新说话人簇"""
        speaker_id = f"SPK_{self._next_speaker_id:03d}"
        self._next_speaker_id += 1
        
        cluster = SpeakerCluster(
            speaker_id=speaker_id,
            doa_center=doa,
        )
        cluster.update_doa(doa)
        
        if embedding is not None:
            cluster.update_embedding(embedding, weight=1.0)
        
        cluster.utterance_count = 1
        
        self._clusters[speaker_id] = cluster
        
        logger.info(f"New speaker detected: {speaker_id} at {doa:.1f}°")
        self._emit("on_new_speaker", cluster)
        
        return speaker_id
    
    def _update_speaker(
        self,
        speaker_id: str,
        doa: float,
        embedding: np.ndarray = None,
        duration: float = 0.0,
    ):
        """更新说话人簇"""
        if speaker_id not in self._clusters:
            return
        
        cluster = self._clusters[speaker_id]
        cluster.update_doa(doa)
        
        if embedding is not None:
            cluster.update_embedding(embedding)
        
        cluster.utterance_count += 1
        cluster.total_duration += duration
        cluster.state = ClusterState.ACTIVE
        
        self._total_assignments += 1
    
    def _cleanup_expired(self):
        """清理过期簇"""
        now = time.time()
        
        for speaker_id, cluster in list(self._clusters.items()):
            elapsed = now - cluster.last_active
            
            if elapsed > self.config.expire_timeout_sec:
                # 标记为过期
                cluster.state = ClusterState.EXPIRED
                self._emit("on_speaker_expired", cluster)
                logger.info(f"Speaker expired: {speaker_id}")
                
            elif elapsed > self.config.inactive_timeout_sec:
                # 标记为不活跃
                cluster.state = ClusterState.INACTIVE
    
    def get_speaker(self, speaker_id: str) -> Optional[SpeakerCluster]:
        """获取说话人信息"""
        return self._clusters.get(speaker_id)
    
    def get_all_speakers(self, active_only: bool = True) -> List[SpeakerCluster]:
        """获取所有说话人"""
        speakers = list(self._clusters.values())
        
        if active_only:
            speakers = [s for s in speakers if s.state == ClusterState.ACTIVE]
        
        return sorted(speakers, key=lambda x: x.last_active, reverse=True)
    
    def set_speaker_role(self, speaker_id: str, role: str):
        """设置说话人角色"""
        if speaker_id in self._clusters:
            self._clusters[speaker_id].role = role
    
    def set_speaker_name(self, speaker_id: str, name: str):
        """设置说话人姓名"""
        if speaker_id in self._clusters:
            self._clusters[speaker_id].name = name
    
    def get_speaker_by_doa(self, doa: float, threshold: float = 30.0) -> Optional[str]:
        """根据DOA查找说话人"""
        for speaker_id, cluster in self._clusters.items():
            if cluster.state == ClusterState.ACTIVE:
                if cluster.doa_distance(doa) < threshold:
                    return speaker_id
        return None
    
    def merge_speakers(self, speaker_id1: str, speaker_id2: str) -> str:
        """合并两个说话人"""
        if speaker_id1 not in self._clusters or speaker_id2 not in self._clusters:
            return speaker_id1
        
        c1 = self._clusters[speaker_id1]
        c2 = self._clusters[speaker_id2]
        
        # 合并声纹
        if c2.embedding_samples > 0:
            weight = c2.embedding_samples / (c1.embedding_samples + c2.embedding_samples)
            c1.update_embedding(c2.embedding_center, weight=weight)
        
        # 合并DOA历史
        for doa in c2.doa_history:
            c1.update_doa(doa)
        
        # 合并统计
        c1.utterance_count += c2.utterance_count
        c1.total_duration += c2.total_duration
        
        # 删除被合并的簇
        del self._clusters[speaker_id2]
        
        logger.info(f"Merged {speaker_id2} into {speaker_id1}")
        return speaker_id1
    
    def on(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, data):
        """触发事件"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        active = sum(1 for c in self._clusters.values() if c.state == ClusterState.ACTIVE)
        return {
            "total_speakers": len(self._clusters),
            "active_speakers": active,
            "total_assignments": self._total_assignments,
        }
    
    def reset(self):
        """重置状态"""
        self._clusters.clear()
        self._next_speaker_id = 1
        self._total_assignments = 0


# 便捷函数
def create_fusion(alpha: float = 0.5, **kwargs) -> SpatioSpectralFusion:
    """创建融合器"""
    return SpatioSpectralFusion(alpha=alpha, **kwargs)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Spatio-Spectral Fusion...")
    
    fusion = SpatioSpectralFusion(alpha=0.5)
    
    # 模拟两个说话人
    emb1 = np.random.randn(192).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    
    emb2 = np.random.randn(192).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)
    
    # 说话人1从0度方向
    spk = fusion.assign_speaker(doa=5.0, embedding=emb1, duration=2.0)
    print(f"Assignment 1: {spk}")
    
    spk = fusion.assign_speaker(doa=10.0, embedding=emb1 * 0.9 + 0.1 * np.random.randn(192), duration=1.5)
    print(f"Assignment 2: {spk}")
    
    # 说话人2从120度方向
    spk = fusion.assign_speaker(doa=120.0, embedding=emb2, duration=2.0)
    print(f"Assignment 3: {spk}")
    
    # 说话人1再次说话
    spk = fusion.assign_speaker(doa=8.0, embedding=emb1, duration=1.0)
    print(f"Assignment 4: {spk}")
    
    # 统计
    print(f"\nStats: {fusion.get_stats()}")
    
    for speaker in fusion.get_all_speakers():
        print(f"Speaker: {speaker.to_dict()}")
