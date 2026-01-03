"""
会诊会话管理器
管理医患对话的完整生命周期，支持多说话人追踪和时间戳记录
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """单条发言记录"""
    id: str
    text: str
    speaker_id: str
    speaker_role: str  # "doctor", "patient", "family", "unknown"
    timestamp: float   # 相对于会话开始的秒数
    duration: float    # 发言时长（秒）
    audio_offset: tuple = None  # (start_byte, end_byte) 在原始音频中的位置
    entities: List[Dict] = field(default_factory=list)  # 提取的实体
    emotion: str = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "speaker_role": self.speaker_role,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "audio_offset": self.audio_offset,
            "entities": self.entities,
            "emotion": self.emotion,
            "confidence": self.confidence
        }


@dataclass
class SpeakerInfo:
    """说话人信息"""
    speaker_id: str
    role: str  # "doctor", "patient", "family"
    name: str = None
    registered_at: datetime = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "speaker_id": self.speaker_id,
            "role": self.role,
            "name": self.name,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "metadata": self.metadata
        }


class ConsultationSession:
    """
    会诊会话管理器
    
    功能：
    1. 管理完整会诊过程的状态
    2. 存储带时间戳的对话记录
    3. 关联每个发言与说话人角色
    4. 支持会话的开始、暂停、结束
    5. 提供转录稿和分段导出
    """
    
    STATUS_ACTIVE = "active"
    STATUS_PAUSED = "paused"
    STATUS_COMPLETED = "completed"
    STATUS_CANCELLED = "cancelled"
    
    def __init__(self, session_id: str = None, patient_info: Dict = None):
        """
        初始化会诊会话
        
        Args:
            session_id: 会话 ID，为空则自动生成
            patient_info: 患者基本信息
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.status = self.STATUS_ACTIVE
        
        # 发言记录
        self.utterances: List[Utterance] = []
        
        # 说话人注册表
        self.speakers: Dict[str, SpeakerInfo] = {}
        
        # 患者信息
        self.patient_info = patient_info or {}
        
        # 会话元数据
        self.metadata = {
            "created_at": self.start_time.isoformat(),
            "consultation_type": "outpatient",  # outpatient, inpatient, emergency
            "department": None,
            "chief_complaint": None
        }
        
        # 音频数据（可选，用于源锚定）
        self.audio_segments: List[bytes] = []
        self.total_audio_duration: float = 0.0
        
        logger.info(f"[会诊会话] 创建新会话: {self.session_id}")
    
    def register_speaker(self, speaker_id: str, role: str, 
                         name: str = None, metadata: Dict = None) -> SpeakerInfo:
        """
        注册说话人及其角色
        
        Args:
            speaker_id: 说话人 ID（来自声纹识别）
            role: 角色 ("doctor", "patient", "family")
            name: 姓名
            metadata: 其他元数据
            
        Returns:
            SpeakerInfo 对象
        """
        if role not in ["doctor", "patient", "family", "unknown"]:
            logger.warning(f"未知角色类型: {role}，使用 unknown")
            role = "unknown"
        
        speaker = SpeakerInfo(
            speaker_id=speaker_id,
            role=role,
            name=name,
            registered_at=datetime.now(),
            metadata=metadata or {}
        )
        self.speakers[speaker_id] = speaker
        logger.info(f"[会诊会话] 注册说话人: {speaker_id} -> {role}")
        return speaker
    
    def add_utterance(self, text: str, speaker_id: str = None,
                      speaker_role: str = None, timestamp: float = None,
                      duration: float = 0.0, audio_segment: bytes = None,
                      emotion: str = None, entities: List[Dict] = None,
                      confidence: float = 1.0) -> Utterance:
        """
        添加一条发言记录
        
        Args:
            text: 转录文本
            speaker_id: 说话人 ID
            speaker_role: 说话人角色（如果未注册，可直接指定）
            timestamp: 时间戳（秒），为空则自动计算
            duration: 发言时长
            audio_segment: 音频片段（用于源锚定）
            emotion: 情感标签
            entities: 提取的实体列表
            confidence: ASR 置信度
            
        Returns:
            Utterance 对象
        """
        # 自动计算时间戳
        if timestamp is None:
            timestamp = (datetime.now() - self.start_time).total_seconds()
        
        # 获取说话人角色
        if speaker_role is None:
            if speaker_id and speaker_id in self.speakers:
                speaker_role = self.speakers[speaker_id].role
            else:
                speaker_role = "unknown"
        
        # 处理音频数据
        audio_offset = None
        if audio_segment:
            start_offset = len(b''.join(self.audio_segments))
            self.audio_segments.append(audio_segment)
            audio_offset = (start_offset, start_offset + len(audio_segment))
        
        # 创建发言记录
        utterance = Utterance(
            id=f"utt_{len(self.utterances):04d}",
            text=text,
            speaker_id=speaker_id or "unknown",
            speaker_role=speaker_role,
            timestamp=timestamp,
            duration=duration,
            audio_offset=audio_offset,
            entities=entities or [],
            emotion=emotion,
            confidence=confidence
        )
        
        self.utterances.append(utterance)
        self.total_audio_duration = max(self.total_audio_duration, timestamp + duration)
        
        logger.debug(f"[会诊会话] 添加发言: [{speaker_role}] {text[:50]}...")
        return utterance
    
    def get_transcript(self, include_roles: bool = True, 
                       include_timestamps: bool = False) -> str:
        """
        获取完整转录稿
        
        Args:
            include_roles: 是否包含角色标注
            include_timestamps: 是否包含时间戳
            
        Returns:
            格式化的转录稿
        """
        lines = []
        for utt in self.utterances:
            parts = []
            
            if include_timestamps:
                minutes = int(utt.timestamp // 60)
                seconds = int(utt.timestamp % 60)
                parts.append(f"[{minutes:02d}:{seconds:02d}]")
            
            if include_roles:
                role_names = {
                    "doctor": "医生",
                    "patient": "患者",
                    "family": "家属",
                    "unknown": "未知"
                }
                parts.append(f"【{role_names.get(utt.speaker_role, '未知')}】")
            
            parts.append(utt.text)
            lines.append(" ".join(parts))
        
        return "\n".join(lines)
    
    def get_utterances_by_role(self, role: str) -> List[Utterance]:
        """获取指定角色的所有发言"""
        return [u for u in self.utterances if u.speaker_role == role]
    
    def get_utterances_by_speaker(self, speaker_id: str) -> List[Utterance]:
        """获取指定说话人的所有发言"""
        return [u for u in self.utterances if u.speaker_id == speaker_id]
    
    def get_speaker_segments(self) -> List[Dict]:
        """
        获取按说话人分组的片段（用于说话人日志可视化）
        
        Returns:
            [
                {"start": 0.0, "end": 3.2, "speaker_role": "doctor", "text": "..."},
                ...
            ]
        """
        segments = []
        for utt in self.utterances:
            segments.append({
                "start": utt.timestamp,
                "end": utt.timestamp + utt.duration,
                "speaker_id": utt.speaker_id,
                "speaker_role": utt.speaker_role,
                "text": utt.text
            })
        return segments
    
    def get_entities(self) -> List[Dict]:
        """获取所有提取的实体"""
        all_entities = []
        for utt in self.utterances:
            for entity in utt.entities:
                entity_copy = entity.copy()
                entity_copy["utterance_id"] = utt.id
                entity_copy["speaker_role"] = utt.speaker_role
                all_entities.append(entity_copy)
        return all_entities
    
    def get_audio_segment(self, utterance_id: str) -> Optional[bytes]:
        """获取指定发言的音频片段（源锚定）"""
        for utt in self.utterances:
            if utt.id == utterance_id and utt.audio_offset:
                start, end = utt.audio_offset
                audio_data = b''.join(self.audio_segments)
                return audio_data[start:end]
        return None
    
    def pause(self):
        """暂停会话"""
        self.status = self.STATUS_PAUSED
        logger.info(f"[会诊会话] 会话暂停: {self.session_id}")
    
    def resume(self):
        """恢复会话"""
        if self.status == self.STATUS_PAUSED:
            self.status = self.STATUS_ACTIVE
            logger.info(f"[会诊会话] 会话恢复: {self.session_id}")
    
    def complete(self, summary: str = None):
        """完成会话"""
        self.status = self.STATUS_COMPLETED
        self.end_time = datetime.now()
        if summary:
            self.metadata["summary"] = summary
        logger.info(f"[会诊会话] 会话完成: {self.session_id}")
    
    def cancel(self, reason: str = None):
        """取消会话"""
        self.status = self.STATUS_CANCELLED
        self.end_time = datetime.now()
        if reason:
            self.metadata["cancel_reason"] = reason
        logger.info(f"[会诊会话] 会话取消: {self.session_id}")
    
    def get_duration(self) -> float:
        """获取会话时长（秒）"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_statistics(self) -> Dict:
        """获取会话统计信息"""
        role_counts = {}
        for utt in self.utterances:
            role_counts[utt.speaker_role] = role_counts.get(utt.speaker_role, 0) + 1
        
        return {
            "session_id": self.session_id,
            "status": self.status,
            "duration": self.get_duration(),
            "utterance_count": len(self.utterances),
            "speaker_count": len(self.speakers),
            "utterances_by_role": role_counts,
            "entity_count": len(self.get_entities()),
            "audio_duration": self.total_audio_duration
        }
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "patient_info": self.patient_info,
            "speakers": {k: v.to_dict() for k, v in self.speakers.items()},
            "utterances": [u.to_dict() for u in self.utterances],
            "metadata": self.metadata,
            "statistics": self.get_statistics()
        }
    
    def save(self, path: str):
        """保存会话到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[会诊会话] 保存到: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ConsultationSession':
        """从文件加载会话"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session = cls(session_id=data["session_id"])
        session.start_time = datetime.fromisoformat(data["start_time"])
        session.end_time = datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        session.status = data["status"]
        session.patient_info = data["patient_info"]
        session.metadata = data["metadata"]
        
        # 恢复说话人
        for speaker_id, speaker_data in data["speakers"].items():
            session.speakers[speaker_id] = SpeakerInfo(
                speaker_id=speaker_data["speaker_id"],
                role=speaker_data["role"],
                name=speaker_data.get("name"),
                registered_at=datetime.fromisoformat(speaker_data["registered_at"]) if speaker_data.get("registered_at") else None,
                metadata=speaker_data.get("metadata", {})
            )
        
        # 恢复发言
        for utt_data in data["utterances"]:
            session.utterances.append(Utterance(
                id=utt_data["id"],
                text=utt_data["text"],
                speaker_id=utt_data["speaker_id"],
                speaker_role=utt_data["speaker_role"],
                timestamp=utt_data["timestamp"],
                duration=utt_data["duration"],
                audio_offset=tuple(utt_data["audio_offset"]) if utt_data.get("audio_offset") else None,
                entities=utt_data.get("entities", []),
                emotion=utt_data.get("emotion"),
                confidence=utt_data.get("confidence", 1.0)
            ))
        
        logger.info(f"[会诊会话] 从文件加载: {path}")
        return session


class ConsultationManager:
    """
    会诊会话管理器
    管理多个活跃会话
    """
    
    def __init__(self, storage_dir: str = "data/consultations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, ConsultationSession] = {}
        logger.info(f"[会诊管理器] 初始化，存储目录: {storage_dir}")
    
    def create_session(self, patient_info: Dict = None) -> ConsultationSession:
        """创建新会话"""
        session = ConsultationSession(patient_info=patient_info)
        self.active_sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ConsultationSession]:
        """获取会话"""
        return self.active_sessions.get(session_id)
    
    def end_session(self, session_id: str, save: bool = True) -> Optional[ConsultationSession]:
        """结束并归档会话"""
        session = self.active_sessions.pop(session_id, None)
        if session:
            session.complete()
            if save:
                save_path = self.storage_dir / f"{session_id}.json"
                session.save(str(save_path))
        return session
    
    def list_active_sessions(self) -> List[str]:
        """列出所有活跃会话"""
        return list(self.active_sessions.keys())
    
    def load_session(self, session_id: str) -> Optional[ConsultationSession]:
        """加载已归档的会话"""
        save_path = self.storage_dir / f"{session_id}.json"
        if save_path.exists():
            return ConsultationSession.load(str(save_path))
        return None


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建会话
    session = ConsultationSession(patient_info={"name": "张三", "age": 45})
    
    # 注册说话人
    session.register_speaker("doctor_001", "doctor", "李医生")
    session.register_speaker("patient_001", "patient", "张三")
    
    # 添加对话
    session.add_utterance("您好，请问哪里不舒服？", speaker_id="doctor_001", duration=2.5)
    session.add_utterance("医生您好，我最近胸口有点闷，闷痛，已经两天了。", speaker_id="patient_001", duration=4.0)
    session.add_utterance("疼痛是持续性的还是阵发性的？", speaker_id="doctor_001", duration=2.0)
    session.add_utterance("是阵发性的，有时候会突然疼一下，然后就好了。", speaker_id="patient_001", duration=3.5)
    
    # 获取转录稿
    print("\n=== 转录稿 ===")
    print(session.get_transcript(include_timestamps=True))
    
    # 获取统计信息
    print("\n=== 统计信息 ===")
    print(json.dumps(session.get_statistics(), ensure_ascii=False, indent=2))
