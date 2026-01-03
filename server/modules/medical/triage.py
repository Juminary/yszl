"""
患者导诊模块
基于 RAG + LLM 的智能导诊系统
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TriageModule:
    """患者导诊模块 - 基于 RAG + LLM 智能导诊"""
    
    def __init__(self, knowledge_path: str = None, rag_module=None, dialogue_module=None):
        """
        初始化导诊模块
        
        Args:
            knowledge_path: 医疗知识库路径（已弃用，保留兼容性）
            rag_module: RAG 检索模块（用于向量检索）
            dialogue_module: 对话模块（用于 LLM 生成）
        """
        # RAG + LLM 模块
        self.rag_module = rag_module
        self.dialogue_module = dialogue_module
        
        # 紧急症状关键词
        self.emergency_keywords = [
            '剧烈', '严重', '大量出血', '呼吸困难', '胸痛', '意识不清',
            '休克', '昏迷', '抽搐', '大面积烧伤', '骨折', '中毒',
            '心梗', '脑梗', '脑出血', '窒息', '心脏骤停'
        ]
        
        self.urgent_keywords = ['高烧', '剧痛', '出血', '呼吸急促', '头晕', '晕厥']
        
        logger.info("TriageModule initialized (RAG + LLM mode)")
    
    def _is_emergency(self, query: str) -> bool:
        """判断是否为紧急情况"""
        return any(keyword in query for keyword in self.emergency_keywords)
    
    def _get_priority(self, query: str) -> str:
        """获取优先级"""
        if self._is_emergency(query):
            return 'emergency'
        if any(keyword in query for keyword in self.urgent_keywords):
            return 'urgent'
        return 'normal'
    
    def triage(self, query: str, age: Optional[int] = None, 
               gender: Optional[str] = None) -> Dict:
        """
        使用 RAG + LLM 进行智能导诊
        
        Args:
            query: 患者症状描述（自然语言）
            age: 年龄
            gender: 性别
        
        Returns:
            导诊结果，包含推荐科室和建议
        """
        # 检查模块是否可用
        if not self.rag_module or not self.dialogue_module:
            logger.error("RAG or Dialogue module not available")
            return {
                'response': '抱歉，导诊服务暂时不可用，请直接前往医院咨询。',
                'query': query,
                'priority': self._get_priority(query),
                'rag_used': False,
                'error': 'Modules not initialized'
            }
        
        try:
            # 1. 使用 RAG 检索相关医疗知识
            logger.info(f"[患者导诊] RAG 检索中: {query[:50]}...")
            rag_context = self.rag_module.build_context(query, top_k=5)
            
            if rag_context:
                logger.info(f"[患者导诊] RAG 检索成功，上下文长度: {len(rag_context)} 字符")
            else:
                logger.info("[患者导诊] RAG 未检索到相关内容")
            
            # 2. 构建患者信息字符串
            patient_info_str = ""
            if age:
                patient_info_str += f"年龄：{age}岁，"
            if gender:
                patient_info_str += f"性别：{gender}，"
            
            # 3. 构建患者端导诊专用 prompt
            system_prompt = """你是一位专业的医院导诊护士。根据患者描述的症状和参考医学信息，推荐最合适的就诊科室，并提供初步的健康生活建议。

请用温和、通俗易懂的语言回复患者，格式要求：
一，只用纯中文回答，禁止英文、数字、字母。
二，只用中文逗号和句号，禁止其他标点如冒号、问号、感叹号、括号、引号。
三，禁止使用星号、井号、横线等任何符号。
四，禁止使用列表、编号、分点格式，必须写成连贯的一段话。
五，回复内容需包含推荐科室及原因，并给出初步的日常护理或生活建议。
六，如果症状紧急，请提醒患者尽快就医。

可选科室包括内科、外科、儿科、妇科、骨科、眼科、耳鼻喉科、皮肤科、急诊科、心血管内科、呼吸内科、消化内科、神经内科等。"""

            # 4. 构建增强查询
            if rag_context:
                enhanced_query = f"参考医学信息：\n{rag_context}\n\n{patient_info_str}患者症状描述：{query}"
            else:
                enhanced_query = f"{patient_info_str}患者症状描述：{query}"
            
            # 5. 调用 LLM 生成导诊建议
            logger.info("[患者导诊] 调用 LLM 生成导诊建议...")
            response = self.dialogue_module.chat(
                query=enhanced_query,
                system_prompt=system_prompt,
                session_id=f"triage_{hash(query) % 10000}",
                use_rag=False  # 已经手动做了 RAG
            )
            
            # 6. 检查是否紧急情况
            priority = self._get_priority(query)
            
            return {
                'response': response.get('response', ''),
                'query': query,
                'patient_info': {'age': age, 'gender': gender},
                'priority': priority,
                'rag_used': bool(rag_context),
                'rag_context': rag_context[:500] if rag_context else None
            }
            
        except Exception as e:
            logger.error(f"Triage failed: {e}")
            return {
                'response': '抱歉，导诊服务出现问题，建议您直接前往医院咨询。',
                'query': query,
                'priority': self._get_priority(query),
                'rag_used': False,
                'error': str(e)
            }
    
    def triage_with_paralinguistics(
        self, 
        query: str, 
        audio_features: Dict = None,
        age: Optional[int] = None, 
        gender: Optional[str] = None
    ) -> Dict:
        """
        结合副语言特征的增强导诊
        
        Args:
            query: 患者症状描述（ASR识别文本）
            audio_features: 副语言特征，包含以下字段：
                - cough_detected: bool - 是否检测到咳嗽
                - cough_type: str - 咳嗽类型 (dry/wet)
                - wheeze_detected: bool - 是否检测到喘息
                - pain_level_acoustic: int - 声学疼痛等级 (1-10)
                - pain_level_semantic: int - 患者自述疼痛等级 (1-10)
                - anxiety_level: str - 焦虑水平 (low/moderate/high)
                - respiratory_distress: bool - 是否有呼吸窘迫迹象
            age: 年龄
            gender: 性别
            
        Returns:
            增强导诊结果
        """
        audio_features = audio_features or {}
        notes = []  # 额外备注
        priority_boost = 0  # 优先级提升
        
        # 1. 咳嗽分析
        if audio_features.get("cough_detected"):
            cough_type = audio_features.get("cough_type", "unknown")
            if cough_type == "wet":
                notes.append("检测到湿咳，提示可能有痰液")
            elif cough_type == "dry":
                notes.append("检测到干咳")
            priority_boost += 1
        
        # 2. 呼吸系统警报
        if audio_features.get("wheeze_detected"):
            notes.append("⚠️ 监测到疑似哮鸣音，建议优先呼吸科")
            priority_boost += 2
        
        if audio_features.get("respiratory_distress"):
            notes.append("⚠️ 检测到呼吸窘迫迹象，建议紧急就诊")
            priority_boost += 3
        
        # 3. 语义-声学不一致检测（疼痛掩饰）
        acoustic_pain = audio_features.get("pain_level_acoustic", 0)
        semantic_pain = audio_features.get("pain_level_semantic", 0)
        
        if acoustic_pain - semantic_pain >= 4:
            notes.append(f"⚠️ 声学特征显示高度痛苦(声学:{acoustic_pain}/自述:{semantic_pain})，与自述不符，建议仔细评估")
            priority_boost += 2
        elif acoustic_pain - semantic_pain >= 3:
            notes.append(f"声学疼痛指标({acoustic_pain})略高于自述({semantic_pain})")
        
        # 4. 焦虑检测
        anxiety = audio_features.get("anxiety_level", "low")
        if anxiety == "high":
            notes.append("患者情绪焦虑，建议安抚")
        
        # 5. 调用基础导诊
        base_result = self.triage(query, age, gender)
        
        # 6. 调整优先级
        original_priority = base_result.get("priority", "normal")
        if priority_boost >= 3 or original_priority == "emergency":
            final_priority = "emergency"
        elif priority_boost >= 2 or original_priority == "urgent":
            final_priority = "urgent"
        else:
            final_priority = original_priority
        
        # 7. 增强响应
        if notes:
            enhanced_response = base_result.get("response", "")
            # 添加副语言分析备注
            para_notes = "，".join(notes)
            enhanced_response = f"{enhanced_response}（声学分析备注：{para_notes}）"
            base_result["response"] = enhanced_response
        
        # 8. 添加副语言分析结果
        base_result["priority"] = final_priority
        base_result["priority_boosted"] = priority_boost > 0
        base_result["paralinguistic_notes"] = notes
        base_result["audio_features"] = audio_features
        
        # 9. 科室推荐调整
        if audio_features.get("wheeze_detected") or audio_features.get("respiratory_distress"):
            base_result["recommended_department_override"] = "呼吸内科/急诊科"
        
        logger.info(f"[增强导诊] 优先级: {original_priority} -> {final_priority}, 备注: {notes}")
        
        return base_result


if __name__ == "__main__":
    # 测试代码（需要初始化 RAG 和对话模块）
    print("TriageModule 需要 RAG 和对话模块才能运行")
    print("请通过 app.py 启动服务器进行测试")
    
    # 副语言增强导诊示例
    print("\n副语言增强导诊示例：")
    print("""
    audio_features = {
        "cough_detected": True,
        "cough_type": "wet",
        "wheeze_detected": True,
        "pain_level_acoustic": 7,
        "pain_level_semantic": 3,
        "anxiety_level": "high",
        "respiratory_distress": True
    }
    result = triage_module.triage_with_paralinguistics(
        query="我咳嗽两天了，有点喘",
        audio_features=audio_features
    )
    """)
