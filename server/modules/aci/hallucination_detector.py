"""
幻觉检测器
验证 LLM 生成的医学内容是否有原始对话支持
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """验证结果"""
    claim: str                    # 被验证的断言
    verified: bool                # 是否验证通过
    evidence: str = None          # 支持证据（原始文本）
    evidence_offset: Tuple[int, int] = None  # 证据在转录稿中的位置
    confidence: float = 0.0       # 置信度
    risk_level: str = "low"       # 风险级别
    reason: str = None            # 验证失败原因
    
    def to_dict(self) -> Dict:
        return {
            "claim": self.claim,
            "verified": self.verified,
            "evidence": self.evidence,
            "evidence_offset": self.evidence_offset,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "reason": self.reason
        }


class HallucinationDetector:
    """
    幻觉检测器
    
    功能：
    1. 验证生成的医学断言是否有原始对话支持
    2. 评估断言的风险级别
    3. 提供修正建议
    """
    
    # 高风险关键词（药物、诊断相关）
    HIGH_RISK_KEYWORDS = [
        "诊断", "确诊", "患有", "得了",
        "开", "处方", "服用", "用药", "剂量",
        "手术", "住院", "急诊",
        "过敏", "禁忌", "不良反应"
    ]
    
    # 中风险关键词（检查、建议相关）
    MEDIUM_RISK_KEYWORDS = [
        "检查", "化验", "CT", "B超", "心电图",
        "建议", "需要", "应该",
        "病史", "家族史"
    ]
    
    # 同义词映射（用于模糊匹配）
    SYNONYMS = {
        "胸痛": ["胸口痛", "胸部疼", "心口痛", "胸闷痛"],
        "头痛": ["头疼", "偏头痛", "头部疼"],
        "高血压": ["血压高", "高压", "血压偏高"],
        "糖尿病": ["血糖高", "糖尿", "血糖偏高"],
        "发热": ["发烧", "高烧", "低烧", "体温高"],
        "服用": ["吃", "在用", "正在服"],
        "两天": ["2天", "二天", "两日"],
        "一周": ["7天", "七天", "一个星期"],
        "严重": ["很重", "厉害", "剧烈"]
    }
    
    def __init__(self, dialogue_module=None, similarity_threshold: float = 0.6):
        """
        初始化幻觉检测器
        
        Args:
            dialogue_module: 对话模块（可选，用于 LLM 增强验证）
            similarity_threshold: 相似度阈值
        """
        self.llm = dialogue_module
        self.similarity_threshold = similarity_threshold
        
        logger.info("[幻觉检测] 初始化完成")
    
    def verify_claim(self, claim: str, transcript: str) -> VerificationResult:
        """
        验证单个断言是否有事实依据
        
        Args:
            claim: 生成的医学断言（如"患者有高血压史"）
            transcript: 原始转录稿
            
        Returns:
            VerificationResult 对象
        """
        # 评估风险级别
        risk_level = self._assess_risk_level(claim)
        
        # 提取关键信息
        keywords = self._extract_keywords(claim)
        
        # 在转录稿中查找证据
        evidence, offset, similarity = self._find_evidence(keywords, transcript)
        
        # 根据相似度和风险级别判断
        if similarity >= self.similarity_threshold:
            return VerificationResult(
                claim=claim,
                verified=True,
                evidence=evidence,
                evidence_offset=offset,
                confidence=similarity,
                risk_level=risk_level
            )
        elif similarity >= 0.4:
            # 部分匹配
            return VerificationResult(
                claim=claim,
                verified=True,
                evidence=evidence,
                evidence_offset=offset,
                confidence=similarity,
                risk_level=risk_level,
                reason="部分匹配，建议核实"
            )
        else:
            return VerificationResult(
                claim=claim,
                verified=False,
                confidence=similarity,
                risk_level=risk_level,
                reason="未在原始对话中找到支持证据"
            )
    
    def _assess_risk_level(self, claim: str) -> str:
        """评估断言的风险级别"""
        claim_lower = claim.lower()
        
        # 检查高风险关键词
        for keyword in self.HIGH_RISK_KEYWORDS:
            if keyword in claim_lower:
                return "high"
        
        # 检查中风险关键词
        for keyword in self.MEDIUM_RISK_KEYWORDS:
            if keyword in claim_lower:
                return "medium"
        
        return "low"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单分词（可以替换为更复杂的分词器）
        # 去除常见停用词
        stopwords = {"的", "了", "是", "有", "在", "和", "与", "或", "等", "及", 
                     "患者", "医生", "表示", "称", "说", "认为"}
        
        # 提取连续的中文词组
        words = []
        current_word = ""
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                current_word += char
            else:
                if current_word and current_word not in stopwords and len(current_word) >= 2:
                    words.append(current_word)
                current_word = ""
        
        if current_word and current_word not in stopwords and len(current_word) >= 2:
            words.append(current_word)
        
        # 添加同义词
        expanded_words = list(words)
        for word in words:
            for key, synonyms in self.SYNONYMS.items():
                if word == key:
                    expanded_words.extend(synonyms)
                elif word in synonyms:
                    expanded_words.append(key)
                    expanded_words.extend([s for s in synonyms if s != word])
        
        return list(set(expanded_words))
    
    def _find_evidence(self, keywords: List[str], transcript: str) -> Tuple[str, Tuple[int, int], float]:
        """
        在转录稿中查找支持证据
        
        Returns:
            (证据文本, 偏移量, 相似度)
        """
        if not keywords:
            return None, None, 0.0
        
        # 将转录稿分成句子
        sentences = re.split(r'[。！？；\n]', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        best_match = None
        best_offset = None
        best_score = 0.0
        
        current_pos = 0
        for sentence in sentences:
            # 计算关键词命中率
            hit_count = sum(1 for kw in keywords if kw in sentence)
            hit_ratio = hit_count / len(keywords) if keywords else 0
            
            # 计算文本相似度
            text_similarity = self._calculate_similarity(" ".join(keywords), sentence)
            
            # 综合得分
            score = hit_ratio * 0.6 + text_similarity * 0.4
            
            if score > best_score:
                best_score = score
                best_match = sentence
                best_offset = (current_pos, current_pos + len(sentence))
            
            current_pos += len(sentence) + 1
        
        return best_match, best_offset, best_score
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def verify_soap(self, soap: Dict, session) -> Dict:
        """
        验证完整 SOAP 病历
        
        Args:
            soap: SOAP 病历字典
            session: ConsultationSession 对象
            
        Returns:
            验证结果字典
        """
        transcript = session.get_transcript(include_roles=True)
        
        warnings = []
        total_claims = 0
        verified_claims = 0
        
        # 验证 Subjective
        subjective = soap.get("subjective", {})
        
        # 验证主诉
        if subjective.get("chief_complaint"):
            total_claims += 1
            result = self.verify_claim(subjective["chief_complaint"], transcript)
            if result.verified:
                verified_claims += 1
            elif result.risk_level in ["high", "medium"]:
                warnings.append({
                    "section": "subjective",
                    "field": "chief_complaint",
                    "claim": result.claim,
                    "risk_level": result.risk_level,
                    "reason": result.reason
                })
        
        # 验证症状
        for symptom in subjective.get("symptoms", []):
            total_claims += 1
            result = self.verify_claim(symptom, transcript)
            if result.verified:
                verified_claims += 1
            else:
                warnings.append({
                    "section": "subjective",
                    "field": "symptoms",
                    "claim": symptom,
                    "risk_level": result.risk_level,
                    "reason": result.reason
                })
        
        # 验证 Assessment (诊断)
        assessment = soap.get("assessment", {})
        for diagnosis in assessment.get("diagnoses", []):
            total_claims += 1
            result = self.verify_claim(diagnosis, transcript)
            if result.verified:
                verified_claims += 1
            else:
                warnings.append({
                    "section": "assessment",
                    "field": "diagnoses",
                    "claim": diagnosis,
                    "risk_level": "high",
                    "reason": result.reason
                })
        
        # 验证 Plan (用药)
        plan = soap.get("plan", {})
        for med in plan.get("medications", []):
            total_claims += 1
            result = self.verify_claim(med, transcript)
            if result.verified:
                verified_claims += 1
            else:
                warnings.append({
                    "section": "plan",
                    "field": "medications",
                    "claim": med,
                    "risk_level": "high",
                    "reason": result.reason
                })
        
        # 计算置信度
        confidence_score = verified_claims / total_claims if total_claims > 0 else 0.0
        
        return {
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "confidence_score": confidence_score,
            "hallucination_warnings": warnings,
            "high_risk_warnings": [w for w in warnings if w["risk_level"] == "high"],
            "is_safe": len([w for w in warnings if w["risk_level"] == "high"]) == 0
        }
    
    def verify_with_llm(self, claim: str, transcript: str) -> VerificationResult:
        """
        使用 LLM 增强的验证（更准确但更慢）
        
        Args:
            claim: 待验证的断言
            transcript: 原始转录稿
            
        Returns:
            VerificationResult
        """
        if not self.llm:
            return self.verify_claim(claim, transcript)
        
        prompt = f"""你是一位严谨的医学文档审核专家。请验证以下医学断言是否在原始对话中有依据。

【待验证断言】
{claim}

【原始对话】
{transcript[:2000]}

请按以下 JSON 格式回答：
{{
  "verified": true/false,
  "evidence": "支持该断言的原始对话内容（如果有）",
  "confidence": 0.0-1.0,
  "reason": "验证判断的理由"
}}

只返回 JSON，不要其他内容。"""

        try:
            response = self.llm.chat(
                query=prompt,
                session_id="verification",
                use_rag=False
            )
            
            import json
            response_text = response.get("response", "{}")
            
            # 处理可能的 markdown 代码块
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result_data = json.loads(response_text.strip())
            
            return VerificationResult(
                claim=claim,
                verified=result_data.get("verified", False),
                evidence=result_data.get("evidence"),
                confidence=result_data.get("confidence", 0.0),
                risk_level=self._assess_risk_level(claim),
                reason=result_data.get("reason")
            )
            
        except Exception as e:
            logger.warning(f"[幻觉检测] LLM 验证失败: {e}")
            return self.verify_claim(claim, transcript)
    
    def get_correction_suggestions(self, warnings: List[Dict]) -> List[Dict]:
        """
        为幻觉警告生成修正建议
        
        Args:
            warnings: 警告列表
            
        Returns:
            修正建议列表
        """
        suggestions = []
        
        for warning in warnings:
            suggestion = {
                "original": warning["claim"],
                "section": warning["section"],
                "field": warning["field"]
            }
            
            if warning["risk_level"] == "high":
                suggestion["action"] = "delete_or_verify"
                suggestion["message"] = f"高风险内容：'{warning['claim']}' 未在对话中找到依据，建议删除或手动核实"
            else:
                suggestion["action"] = "review"
                suggestion["message"] = f"建议核实：'{warning['claim']}'"
            
            suggestions.append(suggestion)
        
        return suggestions


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    detector = HallucinationDetector()
    
    transcript = """
【医生】您好，请问哪里不舒服？
【患者】医生您好，我最近胸口有点闷痛，已经两天了。
【医生】疼痛是什么性质的？
【患者】是阵发性的，活动后会加重。
【医生】您有高血压病史吗？
【患者】有的，吃了五年降压药了，目前在吃硝苯地平。
【医生】好的，我建议您做个心电图检查。
"""
    
    # 测试验证
    test_claims = [
        "患者胸口闷痛两天",        # 应该验证通过
        "患者有高血压病史",         # 应该验证通过
        "患者服用硝苯地平",         # 应该验证通过
        "建议做心电图检查",         # 应该验证通过
        "患者有糖尿病史",           # 应该验证失败（未提及）
        "开具阿司匹林处方",         # 应该验证失败（未提及）
    ]
    
    print("=== 幻觉检测测试 ===\n")
    
    for claim in test_claims:
        result = detector.verify_claim(claim, transcript)
        status = "✓" if result.verified else "✗"
        print(f"[{status}] {claim}")
        print(f"    风险: {result.risk_level}, 置信度: {result.confidence:.2f}")
        if result.evidence:
            print(f"    证据: {result.evidence[:50]}...")
        if result.reason:
            print(f"    原因: {result.reason}")
        print()
