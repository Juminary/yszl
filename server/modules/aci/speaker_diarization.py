"""
说话人日志模块 (Speaker Diarization)
区分医生、患者、家属等不同角色的说话人
"""

import logging
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    说话人日志分析器
    
    功能：
    1. 基于声纹识别区分不同说话人
    2. 自动/手动分配角色（医生、患者、家属）
    3. 根据对话内容推断角色
    4. 提供按说话人分段的对话
    """
    
    ROLES = ["doctor", "patient", "family", "unknown"]
    
    # 医生常用的专业术语/表达
    DOCTOR_PATTERNS = [
        r"(您|你)好.*请问",
        r"哪里不舒服",
        r"什么时候开始",
        r"持续.*多久",
        r"有没有.*过敏",
        r"既往.*病史",
        r"做.*检查",
        r"开.*药",
        r"建议.*治疗",
        r"诊断.*为",
        r"处方",
        r"复查",
        r"注意.*休息",
        r"禁忌",
        r"剂量",
        r"用法用量",
        r"不良反应",
        r"需要.*住院",
        r"病情.*稳定"
    ]
    
    # 患者常用的表达
    PATIENT_PATTERNS = [
        r"医生.*好",
        r"我.*疼",
        r"我.*痛",
        r"我.*不舒服",
        r"我.*难受",
        r"已经.*天了",
        r"一直.*在",
        r"从.*开始",
        r"怎么办",
        r"严重吗",
        r"要紧吗",
        r"吃.*药",
        r"能.*治好",
        r"会不会.*",
        r"担心",
        r"害怕"
    ]
    
    # 家属常用的表达
    FAMILY_PATTERNS = [
        r"(他|她|我爸|我妈|我老公|我老婆|孩子|老人)",
        r"陪.*来",
        r"帮.*问",
        r"在家.*时候",
        r"平时.*怎么"
    ]
    
    def __init__(self, speaker_module=None):
        """
        初始化说话人日志分析器
        
        Args:
            speaker_module: 声纹识别模块（可选，用于高精度识别）
        """
        self.speaker_module = speaker_module
        self.role_registry: Dict[str, str] = {}  # speaker_id -> role
        self.speaker_names: Dict[str, str] = {}  # speaker_id -> name
        
        logger.info("[说话人日志] 初始化完成")
    
    def register_role(self, speaker_id: str, role: str, name: str = None):
        """
        注册说话人角色
        
        Args:
            speaker_id: 说话人 ID
            role: 角色 ("doctor", "patient", "family")
            name: 姓名（可选）
        """
        if role not in self.ROLES:
            logger.warning(f"未知角色: {role}，使用 unknown")
            role = "unknown"
        
        self.role_registry[speaker_id] = role
        if name:
            self.speaker_names[speaker_id] = name
        logger.info(f"[说话人日志] 注册角色: {speaker_id} -> {role}")
    
    def get_role(self, speaker_id: str) -> str:
        """获取说话人角色"""
        return self.role_registry.get(speaker_id, "unknown")
    
    def infer_role_from_content(self, text: str) -> Tuple[str, float]:
        """
        根据内容推断说话人角色
        
        Args:
            text: 对话文本
            
        Returns:
            (角色, 置信度)
        """
        text = text.lower()
        
        # 计算各角色的匹配分数
        scores = {
            "doctor": 0.0,
            "patient": 0.0,
            "family": 0.0
        }
        
        # 匹配医生模式
        for pattern in self.DOCTOR_PATTERNS:
            if re.search(pattern, text):
                scores["doctor"] += 1.0
        
        # 匹配患者模式
        for pattern in self.PATIENT_PATTERNS:
            if re.search(pattern, text):
                scores["patient"] += 1.0
        
        # 匹配家属模式
        for pattern in self.FAMILY_PATTERNS:
            if re.search(pattern, text):
                scores["family"] += 1.0
        
        # 归一化
        total = sum(scores.values())
        if total > 0:
            for role in scores:
                scores[role] /= total
        
        # 找最高分
        best_role = max(scores, key=scores.get)
        confidence = scores[best_role]
        
        # 置信度太低则返回 unknown
        if confidence < 0.3:
            return "unknown", 0.0
        
        return best_role, confidence
    
    def infer_role_from_sequence(self, utterances: List[Dict]) -> Dict[str, str]:
        """
        根据对话序列推断各说话人角色
        
        规则：
        1. 通常对话开始时医生先说话
        2. 医生问，患者答
        3. 统计各说话人的内容特征
        
        Args:
            utterances: 发言列表 [{"speaker_id": "...", "text": "..."}, ...]
            
        Returns:
            {speaker_id: role} 映射
        """
        if not utterances:
            return {}
        
        # 收集每个说话人的所有文本
        speaker_texts: Dict[str, List[str]] = {}
        for utt in utterances:
            speaker_id = utt.get("speaker_id", "unknown")
            if speaker_id not in speaker_texts:
                speaker_texts[speaker_id] = []
            speaker_texts[speaker_id].append(utt.get("text", ""))
        
        # 计算每个说话人的角色得分
        speaker_scores: Dict[str, Dict[str, float]] = {}
        for speaker_id, texts in speaker_texts.items():
            combined_text = " ".join(texts)
            
            scores = {"doctor": 0.0, "patient": 0.0, "family": 0.0}
            
            for pattern in self.DOCTOR_PATTERNS:
                if re.search(pattern, combined_text):
                    scores["doctor"] += 1.0
            
            for pattern in self.PATIENT_PATTERNS:
                if re.search(pattern, combined_text):
                    scores["patient"] += 1.0
            
            for pattern in self.FAMILY_PATTERNS:
                if re.search(pattern, combined_text):
                    scores["family"] += 1.0
            
            speaker_scores[speaker_id] = scores
        
        # 第一个说话人倾向于是医生（如果他的医生特征不太低）
        first_speaker = utterances[0].get("speaker_id", "unknown")
        if first_speaker in speaker_scores:
            speaker_scores[first_speaker]["doctor"] += 0.5
        
        # 分配角色（贪心算法，避免重复）
        result = {}
        assigned_roles = set()
        
        # 按说话人排序（按最高分排序）
        sorted_speakers = sorted(
            speaker_scores.keys(),
            key=lambda s: max(speaker_scores[s].values()),
            reverse=True
        )
        
        for speaker_id in sorted_speakers:
            scores = speaker_scores[speaker_id]
            
            # 找到未分配的最高分角色
            for role in sorted(scores.keys(), key=lambda r: scores[r], reverse=True):
                if role not in assigned_roles or role == "unknown":
                    result[speaker_id] = role
                    if role != "unknown":
                        assigned_roles.add(role)
                    break
            else:
                result[speaker_id] = "unknown"
        
        return result
    
    def diarize_with_voiceprint(self, audio_path: str, 
                                 transcription: List[Dict]) -> List[Dict]:
        """
        结合声纹识别进行说话人日志分析
        
        Args:
            audio_path: 音频文件路径
            transcription: 带时间戳的转录列表
                [{"start": 0.0, "end": 2.5, "text": "..."}, ...]
                
        Returns:
            带说话人标注的分段列表
        """
        if not self.speaker_module:
            logger.warning("[说话人日志] 声纹模块不可用，仅使用内容推断")
            return self._diarize_by_content_only(transcription)
        
        result = []
        
        # TODO: 实现基于声纹的分段切换检测
        # 目前使用简化版本：假设每条转录都由一个说话人完成
        
        for segment in transcription:
            # 识别说话人
            # 理想情况下应该切出对应时间段的音频，这里简化处理
            text = segment.get("text", "")
            
            # 先尝试声纹识别（如果有完整音频）
            speaker_id = "unknown"
            
            # 用内容推断角色
            role, confidence = self.infer_role_from_content(text)
            
            # 如果已注册，使用注册的角色
            if speaker_id in self.role_registry:
                role = self.role_registry[speaker_id]
            
            result.append({
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": text,
                "speaker_id": speaker_id,
                "speaker_role": role,
                "role_confidence": confidence
            })
        
        return result
    
    def _diarize_by_content_only(self, transcription: List[Dict]) -> List[Dict]:
        """仅基于内容进行角色推断"""
        result = []
        
        # 先推断整体角色分配
        role_map = self.infer_role_from_sequence([
            {"speaker_id": f"speaker_{i}", "text": seg.get("text", "")}
            for i, seg in enumerate(transcription)
        ])
        
        for i, segment in enumerate(transcription):
            text = segment.get("text", "")
            speaker_id = f"speaker_{i}"
            
            # 使用单条内容推断
            role, confidence = self.infer_role_from_content(text)
            
            result.append({
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": text,
                "speaker_id": speaker_id,
                "speaker_role": role,
                "role_confidence": confidence
            })
        
        return result
    
    def merge_consecutive_segments(self, segments: List[Dict], 
                                    gap_threshold: float = 0.5) -> List[Dict]:
        """
        合并同一说话人的连续片段
        
        Args:
            segments: 分段列表
            gap_threshold: 合并的最大间隔（秒）
            
        Returns:
            合并后的分段列表
        """
        if not segments:
            return []
        
        merged = [segments[0].copy()]
        
        for seg in segments[1:]:
            last = merged[-1]
            
            # 检查是否可以合并
            if (seg["speaker_role"] == last["speaker_role"] and 
                seg["start"] - last["end"] <= gap_threshold):
                # 合并
                last["end"] = seg["end"]
                last["text"] = last["text"] + " " + seg["text"]
            else:
                merged.append(seg.copy())
        
        return merged
    
    def get_speaker_timeline(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """
        生成按说话人分组的时间线
        
        Args:
            segments: 分段列表
            
        Returns:
            {role: [segments]}
        """
        timeline = {}
        for seg in segments:
            role = seg.get("speaker_role", "unknown")
            if role not in timeline:
                timeline[role] = []
            timeline[role].append(seg)
        return timeline
    
    def calculate_speaking_ratio(self, segments: List[Dict]) -> Dict[str, float]:
        """
        计算各角色的发言时长占比
        
        Returns:
            {role: ratio}
        """
        role_durations = {}
        total_duration = 0.0
        
        for seg in segments:
            role = seg.get("speaker_role", "unknown")
            duration = seg.get("end", 0) - seg.get("start", 0)
            role_durations[role] = role_durations.get(role, 0) + duration
            total_duration += duration
        
        if total_duration == 0:
            return {}
        
        return {role: dur / total_duration for role, dur in role_durations.items()}


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    diarizer = SpeakerDiarizer()
    
    # 测试内容推断
    test_texts = [
        "您好，请问哪里不舒服？",
        "医生您好，我最近胸口有点闷，闷痛，已经两天了。",
        "疼痛是持续性的还是阵发性的？",
        "是阵发性的，有时候会突然疼一下。",
        "我爸他平时血压就高，会不会有关系？",
        "有可能，我建议先做一个心电图检查。"
    ]
    
    print("=== 内容推断测试 ===")
    for text in test_texts:
        role, conf = diarizer.infer_role_from_content(text)
        print(f"[{role:8s}|{conf:.2f}] {text}")
    
    # 测试序列推断
    print("\n=== 序列推断测试 ===")
    utterances = [
        {"speaker_id": "A", "text": text}
        for text in test_texts
    ]
    role_map = diarizer.infer_role_from_sequence(utterances)
    print(f"角色映射: {role_map}")
