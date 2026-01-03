"""
医生辅助诊断模块
基于 RAG + LLM 的智能辅助诊断系统
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DiagnosisAssistant:
    """医生辅助诊断模块 - 基于 RAG + LLM 智能诊断"""
    
    def __init__(self, knowledge_path: str = None, rag_module=None, dialogue_module=None):
        """
        初始化辅助诊断模块
        
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
            '剧烈疼痛', '大量出血', '呼吸困难', '意识不清', '休克', '胸痛',
            '心梗', '脑梗', '脑出血', '窒息', '心脏骤停', '昏迷', '抽搐'
        ]
        
        logger.info("DiagnosisAssistant initialized (RAG + LLM mode)")
    
    def _check_emergency(self, query: str) -> Optional[str]:
        """检查是否有需要警告的紧急情况"""
        if any(keyword in query for keyword in self.emergency_keywords):
            return "紧急情况，患者存在危急症状，建议立即处理"
        return None
    
    def diagnose(self, query: str = None, symptoms: List[str] = None, 
                 patient_info: Dict = None) -> Dict:
        """
        使用 RAG + LLM 进行智能辅助诊断
        
        Args:
            query: 症状描述（自然语言）
            symptoms: 症状列表（可选，会转换为自然语言）
            patient_info: 患者信息（年龄、性别、病史等）
        
        Returns:
            诊断辅助结果
        """
        # 检查模块是否可用
        if not self.rag_module or not self.dialogue_module:
            logger.error("RAG or Dialogue module not available")
            return {
                'response': '抱歉，诊断服务暂时不可用，请稍后再试。',
                'query': query,
                'rag_used': False,
                'error': 'Modules not initialized'
            }
        
        try:
            # 1. 构建查询文本
            if query:
                search_query = query
            elif symptoms:
                search_query = f"患者症状：{', '.join(symptoms)}"
            else:
                return {'error': '请提供症状描述或症状列表'}
            
            # 2. 构建患者信息字符串
            patient_info_str = ""
            if patient_info:
                if patient_info.get('age'):
                    patient_info_str += f"年龄{patient_info['age']}岁，"
                if patient_info.get('gender'):
                    patient_info_str += f"性别{patient_info['gender']}，"
                if patient_info.get('history'):
                    patient_info_str += f"既往史{patient_info['history']}，"
            
            # 3. 使用 RAG 检索相关医疗知识
            logger.info(f"[医生诊断] RAG 检索中: {search_query[:50]}...")
            rag_context = self.rag_module.build_context(search_query, top_k=5)
            
            if rag_context:
                logger.info(f"[医生诊断] RAG 检索成功，上下文长度: {len(rag_context)} 字符")
            else:
                logger.info("[医生诊断] RAG 未检索到相关内容")
            
            # 4. 构建医生端诊断专用 prompt
            system_prompt = """你是一位资深的临床医生助手。根据患者症状和参考医学资料，提供专业的辅助诊断建议。

请按以下格式用自然语言回复，要求：
一，只用纯中文回答，禁止英文、数字、字母。
二，只用中文逗号和句号，禁止其他标点如冒号、问号、感叹号、括号、引号。
三，禁止使用星号、井号、横线等任何符号。
四，禁止使用列表、编号、分点格式，必须写成连贯的一段话。
五，使用专业医学术语。
六，内容需要涵盖可能的诊断、鉴别要点、建议检查和处理建议。
七，最后提醒这只是辅助建议，最终诊断需要医生综合判断。"""

            # 5. 构建增强查询
            if rag_context:
                enhanced_query = f"参考医学资料：\n{rag_context}\n\n{patient_info_str}{search_query}"
            else:
                enhanced_query = f"{patient_info_str}{search_query}"
            
            # 6. 调用 LLM 生成诊断建议
            logger.info("[医生诊断] 调用 LLM 生成诊断建议...")
            response = self.dialogue_module.chat(
                query=enhanced_query,
                system_prompt=system_prompt,
                session_id=f"diagnosis_{hash(search_query) % 10000}",
                use_rag=False  # 已经手动做了 RAG
            )
            
            # 7. 检查是否有危急情况
            warning = self._check_emergency(search_query)
            
            return {
                'response': response.get('response', ''),
                'query': query or ', '.join(symptoms) if symptoms else '',
                'symptoms': symptoms,
                'patient_info': patient_info,
                'rag_used': bool(rag_context),
                'rag_context': rag_context[:500] if rag_context else None,
                'warning': warning
            }
            
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            return {
                'response': '抱歉，诊断服务出现问题，请稍后再试。',
                'query': query,
                'rag_used': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # 测试代码（需要初始化 RAG 和对话模块）
    print("DiagnosisAssistant 需要 RAG 和对话模块才能运行")
    print("请通过 app.py 启动服务器进行测试")
