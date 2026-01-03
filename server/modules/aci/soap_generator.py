"""
SOAP 病历生成器
从医患对话自动生成结构化 SOAP 病历笔记
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class SourceAnchor:
    """源锚点 - 链接到原始对话"""
    text: str                    # 原始文本
    utterance_id: str            # 发言 ID
    speaker_role: str            # 说话人角色
    timestamp: float             # 时间戳（秒）
    audio_offset: tuple = None   # 音频偏移（字节）
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "utterance_id": self.utterance_id,
            "speaker_role": self.speaker_role,
            "timestamp": self.timestamp,
            "audio_offset": self.audio_offset
        }


@dataclass
class SOAPSection:
    """SOAP 病历的一个部分"""
    content: str                          # 内容文本
    entities: List[Dict] = field(default_factory=list)  # 相关实体
    source_anchors: List[SourceAnchor] = field(default_factory=list)  # 源锚点
    confidence: float = 1.0               # 置信度
    verified: bool = False                # 是否已验证
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "entities": self.entities,
            "source_anchors": [s.to_dict() for s in self.source_anchors],
            "confidence": self.confidence,
            "verified": self.verified
        }


@dataclass
class SOAPNote:
    """完整的 SOAP 病历"""
    session_id: str
    generated_at: datetime
    
    # SOAP 四部分
    subjective: Dict = field(default_factory=dict)   # 主诉
    objective: Dict = field(default_factory=dict)    # 客观检查
    assessment: Dict = field(default_factory=dict)   # 评估
    plan: Dict = field(default_factory=dict)         # 计划
    
    # 元数据
    patient_info: Dict = field(default_factory=dict)
    verification: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "generated_at": self.generated_at.isoformat(),
            "subjective": self.subjective,
            "objective": self.objective,
            "assessment": self.assessment,
            "plan": self.plan,
            "patient_info": self.patient_info,
            "verification": self.verification
        }
    
    def to_markdown(self) -> str:
        """导出为 Markdown 格式"""
        lines = [
            f"# SOAP 病历笔记",
            f"",
            f"**生成时间**: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**会话ID**: {self.session_id}",
            f""
        ]
        
        # 患者信息
        if self.patient_info:
            lines.append("## 患者信息")
            for k, v in self.patient_info.items():
                if v:
                    lines.append(f"- **{k}**: {v}")
            lines.append("")
        
        # Subjective
        lines.append("## S - 主诉 (Subjective)")
        if self.subjective:
            if self.subjective.get("chief_complaint"):
                lines.append(f"**主诉**: {self.subjective['chief_complaint']}")
            if self.subjective.get("history_present_illness"):
                lines.append(f"\n**现病史**: {self.subjective['history_present_illness']}")
            if self.subjective.get("symptoms"):
                lines.append(f"\n**症状**: {', '.join(self.subjective['symptoms'])}")
        lines.append("")
        
        # Objective
        lines.append("## O - 客观检查 (Objective)")
        if self.objective:
            if self.objective.get("vital_signs"):
                lines.append(f"**生命体征**: {self.objective['vital_signs']}")
            if self.objective.get("physical_exam"):
                lines.append(f"\n**体格检查**: {self.objective['physical_exam']}")
            if self.objective.get("test_results"):
                lines.append(f"\n**检查结果**: {self.objective['test_results']}")
        lines.append("")
        
        # Assessment
        lines.append("## A - 评估 (Assessment)")
        if self.assessment:
            if self.assessment.get("diagnoses"):
                lines.append(f"**诊断**: {', '.join(self.assessment['diagnoses'])}")
            if self.assessment.get("differential"):
                lines.append(f"\n**鉴别诊断**: {', '.join(self.assessment['differential'])}")
        lines.append("")
        
        # Plan
        lines.append("## P - 计划 (Plan)")
        if self.plan:
            if self.plan.get("medications"):
                lines.append("**药物治疗**:")
                for med in self.plan["medications"]:
                    lines.append(f"  - {med}")
            if self.plan.get("procedures"):
                lines.append(f"\n**检查/治疗**: {', '.join(self.plan['procedures'])}")
            if self.plan.get("follow_up"):
                lines.append(f"\n**随访**: {self.plan['follow_up']}")
            if self.plan.get("instructions"):
                lines.append(f"\n**医嘱**: {self.plan['instructions']}")
        lines.append("")
        
        # 验证信息
        if self.verification:
            lines.append("---")
            lines.append(f"*置信度: {self.verification.get('confidence_score', 0):.0%}*")
            if self.verification.get("hallucination_warnings"):
                lines.append(f"\n⚠️ **警告**: {len(self.verification['hallucination_warnings'])} 处内容需要核实")
        
        return "\n".join(lines)


class SOAPGenerator:
    """
    SOAP 病历生成器
    
    将医患对话转换为结构化的 SOAP 病历笔记
    """
    
    # LLM 提示模板
    SOAP_PROMPT_TEMPLATE = """你是一位资深的医学文档撰写专家。请根据以下医患对话记录，生成结构化的 SOAP 病历笔记。

【对话记录】
{transcript}

【提取的实体】
症状: {symptoms}
疾病: {diseases}
药物: {medications}
检查: {procedures}

【患者信息】
{patient_info}

请生成 SOAP 病历，严格按照以下 JSON 格式输出：

{{
  "subjective": {{
    "chief_complaint": "主诉（一句话概括）",
    "history_present_illness": "现病史（详细描述）",
    "symptoms": ["症状1", "症状2"],
    "past_history": "既往史",
    "family_history": "家族史",
    "allergy_history": "过敏史"
  }},
  "objective": {{
    "vital_signs": "生命体征",
    "physical_exam": "体格检查发现",
    "test_results": "检查结果"
  }},
  "assessment": {{
    "diagnoses": ["诊断1", "诊断2"],
    "differential": ["鉴别诊断1", "鉴别诊断2"],
    "severity": "病情严重程度"
  }},
  "plan": {{
    "medications": ["药物1及用法", "药物2及用法"],
    "procedures": ["检查/治疗1", "检查/治疗2"],
    "follow_up": "随访计划",
    "instructions": "患者指导"
  }}
}}

注意：
1. 只包含对话中明确提及的内容，不要编造
2. 如果某项信息未提及，使用 null 或空数组
3. 只输出 JSON，不要其他内容"""

    def __init__(self, 
                 entity_extractor=None,
                 dialogue_module=None,
                 hallucination_detector=None):
        """
        初始化 SOAP 生成器
        
        Args:
            entity_extractor: 临床实体抽取器
            dialogue_module: 对话模块（LLM）
            hallucination_detector: 幻觉检测器
        """
        self.extractor = entity_extractor
        self.llm = dialogue_module
        self.verifier = hallucination_detector
        
        logger.info("[SOAP生成器] 初始化完成")
    
    def generate_soap(self, session) -> SOAPNote:
        """
        从会诊记录生成 SOAP 病历
        
        Args:
            session: ConsultationSession 对象
            
        Returns:
            SOAPNote 对象
        """
        # 创建 SOAP 笔记
        soap = SOAPNote(
            session_id=session.session_id,
            generated_at=datetime.now(),
            patient_info=session.patient_info
        )
        
        # 获取转录稿
        transcript = session.get_transcript(include_roles=True, include_timestamps=True)
        
        # 提取实体
        all_entities = self._extract_all_entities(session)
        
        # 使用 LLM 生成 SOAP
        if self.llm:
            soap = self._generate_with_llm(soap, session, transcript, all_entities)
        else:
            # 回退到规则生成
            soap = self._generate_with_rules(soap, session, all_entities)
        
        # 添加源锚点
        self._add_source_anchors(soap, session)
        
        # 验证内容（如果有验证器）
        if self.verifier:
            verification = self.verifier.verify_soap(soap.to_dict(), session)
            soap.verification = verification
        else:
            soap.verification = {"confidence_score": 0.7, "hallucination_warnings": []}
        
        logger.info(f"[SOAP生成器] 生成完成: {session.session_id}")
        return soap
    
    def _extract_all_entities(self, session) -> Dict[str, List]:
        """从会话中提取所有实体"""
        entities = {
            "symptoms": [],
            "diseases": [],
            "medications": [],
            "procedures": [],
            "body_parts": [],
            "time": []
        }
        
        if not self.extractor:
            return entities
        
        # 从每条发言中提取
        for utt in session.utterances:
            extracted = self.extractor.extract_entities(utt.text, utt.speaker_role)
            for entity in extracted:
                if entity.type == "symptom":
                    entities["symptoms"].append(entity.text)
                elif entity.type == "disease":
                    entities["diseases"].append(entity.text)
                elif entity.type == "medication":
                    med_str = entity.text
                    if entity.attributes:
                        attrs = []
                        if entity.attributes.get("dosage"):
                            attrs.append(entity.attributes["dosage"])
                        if entity.attributes.get("frequency"):
                            attrs.append(entity.attributes["frequency"])
                        if attrs:
                            med_str += f" ({', '.join(attrs)})"
                    entities["medications"].append(med_str)
                elif entity.type == "procedure":
                    entities["procedures"].append(entity.text)
                elif entity.type == "body_part":
                    entities["body_parts"].append(entity.text)
                elif entity.type == "time":
                    entities["time"].append(entity.text)
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _generate_with_llm(self, soap: SOAPNote, session, 
                            transcript: str, entities: Dict) -> SOAPNote:
        """使用 LLM 生成 SOAP"""
        # 构建患者信息字符串
        patient_info_str = ""
        if session.patient_info:
            parts = []
            if session.patient_info.get("name"):
                parts.append(f"姓名: {session.patient_info['name']}")
            if session.patient_info.get("age"):
                parts.append(f"年龄: {session.patient_info['age']}")
            if session.patient_info.get("gender"):
                parts.append(f"性别: {session.patient_info['gender']}")
            patient_info_str = ", ".join(parts) if parts else "未提供"
        
        # 构建提示
        prompt = self.SOAP_PROMPT_TEMPLATE.format(
            transcript=transcript[:3000],  # 限制长度
            symptoms=", ".join(entities.get("symptoms", [])) or "未提及",
            diseases=", ".join(entities.get("diseases", [])) or "未提及",
            medications=", ".join(entities.get("medications", [])) or "未提及",
            procedures=", ".join(entities.get("procedures", [])) or "未提及",
            patient_info=patient_info_str
        )
        
        try:
            response = self.llm.chat(
                query=prompt,
                session_id=f"soap_{session.session_id}",
                use_rag=False
            )
            
            response_text = response.get("response", "{}")
            
            # 尝试提取 JSON
            # 处理可能的 markdown 代码块
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            soap_data = json.loads(response_text.strip())
            
            # 填充 SOAP
            soap.subjective = soap_data.get("subjective", {})
            soap.objective = soap_data.get("objective", {})
            soap.assessment = soap_data.get("assessment", {})
            soap.plan = soap_data.get("plan", {})
            
        except json.JSONDecodeError as e:
            logger.warning(f"[SOAP生成器] JSON 解析失败: {e}")
            # 回退到规则生成
            soap = self._generate_with_rules(soap, session, entities)
        except Exception as e:
            logger.error(f"[SOAP生成器] LLM 生成失败: {e}")
            soap = self._generate_with_rules(soap, session, entities)
        
        return soap
    
    def _generate_with_rules(self, soap: SOAPNote, session, entities: Dict) -> SOAPNote:
        """基于规则生成 SOAP（回退方案）"""
        # 收集患者发言作为 Subjective
        patient_utterances = session.get_utterances_by_role("patient")
        patient_text = " ".join([u.text for u in patient_utterances])
        
        # 收集医生发言
        doctor_utterances = session.get_utterances_by_role("doctor")
        doctor_text = " ".join([u.text for u in doctor_utterances])
        
        # Subjective - 主诉
        soap.subjective = {
            "chief_complaint": entities.get("symptoms", ["未记录"])[0] if entities.get("symptoms") else "未记录",
            "history_present_illness": patient_text[:500] if patient_text else "未记录",
            "symptoms": entities.get("symptoms", [])
        }
        
        # Objective - 客观检查
        soap.objective = {
            "vital_signs": None,
            "physical_exam": None,
            "test_results": ", ".join(entities.get("procedures", [])) if entities.get("procedures") else None
        }
        
        # Assessment - 评估
        soap.assessment = {
            "diagnoses": entities.get("diseases", []),
            "differential": [],
            "severity": None
        }
        
        # Plan - 计划
        soap.plan = {
            "medications": entities.get("medications", []),
            "procedures": entities.get("procedures", []),
            "follow_up": None,
            "instructions": None
        }
        
        return soap
    
    def _add_source_anchors(self, soap: SOAPNote, session):
        """为 SOAP 内容添加源锚点"""
        # 为主诉添加锚点
        if soap.subjective.get("chief_complaint"):
            anchor = self._find_anchor(soap.subjective["chief_complaint"], session)
            if anchor:
                if "source_anchors" not in soap.subjective:
                    soap.subjective["source_anchors"] = []
                soap.subjective["source_anchors"].append(anchor.to_dict())
        
        # 为症状添加锚点
        for symptom in soap.subjective.get("symptoms", []):
            anchor = self._find_anchor(symptom, session)
            if anchor:
                if "source_anchors" not in soap.subjective:
                    soap.subjective["source_anchors"] = []
                soap.subjective["source_anchors"].append(anchor.to_dict())
        
        # 为诊断添加锚点
        for diagnosis in soap.assessment.get("diagnoses", []):
            anchor = self._find_anchor(diagnosis, session)
            if anchor:
                if "source_anchors" not in soap.assessment:
                    soap.assessment["source_anchors"] = []
                soap.assessment["source_anchors"].append(anchor.to_dict())
        
        # 为药物添加锚点
        for med in soap.plan.get("medications", []):
            # 提取药物名称（去掉剂量等附加信息）
            med_name = med.split("(")[0].strip() if "(" in med else med
            anchor = self._find_anchor(med_name, session)
            if anchor:
                if "source_anchors" not in soap.plan:
                    soap.plan["source_anchors"] = []
                soap.plan["source_anchors"].append(anchor.to_dict())
    
    def _find_anchor(self, text: str, session) -> Optional[SourceAnchor]:
        """在会话中查找文本对应的源锚点"""
        for utt in session.utterances:
            if text.lower() in utt.text.lower():
                return SourceAnchor(
                    text=utt.text,
                    utterance_id=utt.id,
                    speaker_role=utt.speaker_role,
                    timestamp=utt.timestamp,
                    audio_offset=utt.audio_offset
                )
        return None
    
    def generate_realtime_preview(self, session) -> Dict:
        """
        生成实时预览（轻量级，用于 UI 展示）
        
        Args:
            session: 当前会话
            
        Returns:
            简化的 SOAP 预览
        """
        entities = self._extract_all_entities(session)
        
        preview = {
            "subjective": {
                "symptoms": entities.get("symptoms", [])[:5],
                "utterance_count": len(session.get_utterances_by_role("patient"))
            },
            "objective": {
                "procedures": entities.get("procedures", [])[:3]
            },
            "assessment": {
                "diseases": entities.get("diseases", [])[:3]
            },
            "plan": {
                "medications": entities.get("medications", [])[:3]
            },
            "entity_count": sum(len(v) for v in entities.values()),
            "is_complete": False
        }
        
        return preview


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 需要导入相关模块
    from consultation_session import ConsultationSession
    from clinical_entity_extractor import ClinicalEntityExtractor
    
    # 创建测试会话
    session = ConsultationSession(patient_info={"name": "张三", "age": 45, "gender": "男"})
    session.register_speaker("doctor_001", "doctor", "李医生")
    session.register_speaker("patient_001", "patient", "张三")
    
    # 模拟对话
    session.add_utterance("您好，请问哪里不舒服？", speaker_id="doctor_001", duration=2.0)
    session.add_utterance("医生您好，我最近胸口有点闷痛，已经两天了。", speaker_id="patient_001", duration=3.5)
    session.add_utterance("疼痛是什么性质的？持续性还是阵发性？", speaker_id="doctor_001", duration=2.5)
    session.add_utterance("是阵发性的，活动后会加重，休息后能缓解。", speaker_id="patient_001", duration=3.0)
    session.add_utterance("您有高血压或糖尿病病史吗？", speaker_id="doctor_001", duration=2.0)
    session.add_utterance("有高血压，吃了五年的降压药了，目前在吃硝苯地平。", speaker_id="patient_001", duration=4.0)
    session.add_utterance("好的，我建议您先做个心电图和心脏彩超检查一下。", speaker_id="doctor_001", duration=3.0)
    
    # 创建生成器
    extractor = ClinicalEntityExtractor()
    generator = SOAPGenerator(entity_extractor=extractor)
    
    # 生成 SOAP
    soap = generator.generate_soap(session)
    
    print("=== SOAP 病历 ===")
    print(soap.to_markdown())
    print("\n=== JSON 格式 ===")
    print(json.dumps(soap.to_dict(), ensure_ascii=False, indent=2))
