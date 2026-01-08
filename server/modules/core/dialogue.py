"""
自然语言对话模块
使用 Qwen2.5-0.5B-Instruct 实现轻量级对话
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, List, Optional
from collections import deque

# 导入情感回复策略模块
try:
    from ..nlp.emotion_response_strategy import EmotionResponseStrategy
    EMOTION_STRATEGY_AVAILABLE = True
except ImportError:
    EMOTION_STRATEGY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DialogueModule:
    """对话系统模块 - 使用Qwen2.5-0.5B-Instruct"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", 
                 device: str = "cuda",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 history_length: int = 10,
                 rag_module = None,
                 system_prompt: str = None):
        """
        初始化对话模块
        
        Args:
            model_name: 模型名称
            device: 运行设备
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            history_length: 保留的对话历史轮数
            rag_module: RAG 检索模块（可选）
            system_prompt: 系统提示词（可选，从配置文件读取）
        """
        # 设备选择
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.history_length = history_length
        
        # RAG 模块
        self.rag_module = rag_module
        
        # 系统提示词（从配置文件读取或使用默认值）
        self.default_system_prompt = system_prompt
        
        # 情感回复策略模块
        self.emotion_strategy = EmotionResponseStrategy() if EMOTION_STRATEGY_AVAILABLE else None
        if self.emotion_strategy:
            logger.info("Emotion Response Strategy initialized")
        
        # 对话历史管理
        self.conversations = {}
        
        logger.info(f"Loading dialogue model: {model_name} on {self.device}")
        try:
            # 处理本地路径：只有明确指定相对/绝对路径时才当作本地文件
            import os
            from pathlib import Path
            
            # 优先使用 ModelScope 下载模型
            if model_name.startswith('Qwen/') or model_name.startswith('qwen/'):
                try:
                    from modelscope import snapshot_download
                    # 从 ModelScope 下载模型
                    models_dir = Path(__file__).parent.parent.parent / "models" / "dialogue"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Downloading {model_name} from ModelScope...")
                    model_path = snapshot_download(model_name, cache_dir=str(models_dir))
                    logger.info(f"Model downloaded to: {model_path}")
                except ImportError:
                    logger.warning("modelscope not available, falling back to HuggingFace")
                    model_path = model_name
            elif model_name.startswith('./') or model_name.startswith('/') or os.path.exists(model_name):
                model_path = os.path.abspath(model_name)
                logger.info(f"Using local model path: {model_path}")
            else:
                # 当作 HuggingFace repo ID，让 transformers 自动处理
                model_path = model_name
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "mps":
                self.model = self.model.to(self.device)
            elif self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Qwen2.5-0.5B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load dialogue model: {e}")
            raise
    
    def _clean_for_tts(self, text: str) -> str:
        """
        清理文本，去除不适合语音合成的符号
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        import re
        
        # 去除 Markdown 格式符号
        text = re.sub(r'\*+', '', text)  # 星号
        text = re.sub(r'#+', '', text)   # 井号
        text = re.sub(r'`+', '', text)   # 反引号
        text = re.sub(r'~+', '', text)   # 波浪号
        text = re.sub(r'_+', '', text)   # 下划线
        
        # 去除列表编号格式
        text = re.sub(r'^\s*\d+[\.\、\)]\s*', '', text, flags=re.MULTILINE)  # 1. 2. 等
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)  # - 或 • 列表
        
        # 替换不适合朗读的标点为逗号或句号
        text = text.replace('：', '，')
        text = text.replace(':', '，')
        text = text.replace('；', '，')
        text = text.replace(';', '，')
        text = text.replace('！', '。')
        text = text.replace('!', '。')
        text = text.replace('？', '。')
        text = text.replace('?', '。')
        text = text.replace('（', '，')
        text = text.replace('）', '，')
        text = text.replace('(', '，')
        text = text.replace(')', '，')
        text = text.replace('"', '')
        text = text.replace('"', '')
        text = text.replace('"', '')
        text = text.replace("'", '')
        text = text.replace("'", '')
        text = text.replace("'", '')
        text = text.replace('【', '')
        text = text.replace('】', '')
        text = text.replace('[', '')
        text = text.replace(']', '')
        
        # 合并多余的标点
        text = re.sub(r'[，,]+', '，', text)
        text = re.sub(r'[。\.]+', '。', text)
        text = re.sub(r'，。', '。', text)
        text = re.sub(r'。，', '。', text)
        
        # 去除多余空白
        text = re.sub(r'\s+', '', text)
        
        # 去除开头的标点
        text = re.sub(r'^[，。,\.]+', '', text)
        
        # 移除英文提示词（只移除明显的提示词，不误删正常内容）
        # 常见的提示词关键词（通常在文本开头或独立出现）
        prompt_keywords = [
            r'\bendofprompt\b',  # <endofprompt> 标签
            r'<endofprompt>',     # 完整标签
            r'\bstyle\b',         # style 关键词
            r'\bcontent\b',       # content 关键词
            r'\bemotion\b',       # emotion 关键词
            r'\[style\]',         # [style] 标签
            r'\[content\]',       # [content] 标签
        ]
        for pattern in prompt_keywords:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除文本开头或结尾的英文单词（可能是残留的提示词）
        # 只移除独立的英文单词，不误删中文中的字母
        text = re.sub(r'^[a-zA-Z]+\s+', '', text)  # 开头的英文单词
        text = re.sub(r'\s+[a-zA-Z]+$', '', text)  # 结尾的英文单词
        
        # 移除可能残留的风格描述（如果开头包含风格关键词且较短）
        style_keywords = ["语气", "语速", "语调", "音量", "音调", "节奏", "温暖", "柔和", "明亮", "轻快", "安慰", "耐心", "稍快", "稍慢", "偏快", "偏慢"]
        lines = text.split('\n')
        if len(lines) > 0:
            first_line = lines[0].strip()
            # 如果第一行包含风格关键词且较短（可能是残留的风格描述），移除它
            if any(kw in first_line for kw in style_keywords) and len(first_line) <= 30:
                text = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        
        return text.strip()
    
    def chat(self, query: str, session_id: str = "default", 
             system_prompt: str = None, reset: bool = False,
             emotion: str = None, speaker_id: str = None,
             use_rag: bool = True, emotional_mode: bool = False) -> Dict:
        """
        进行对话
        
        Args:
            query: 用户输入
            session_id: 会话ID
            system_prompt: 系统提示词
            reset: 是否重置对话历史
            emotion: 用户情感（可选，用于情感感知回复）
            speaker_id: 说话人ID（可选）
            use_rag: 是否使用 RAG 检索（默认 True）
            emotional_mode: 是否启用情感感知模式（输出 style + content）
        
        Returns:
            对话结果（emotional_mode=True 时包含 style 和 content 字段）
        """
        rag_context = None
        try:
            # 初始化或重置会话
            if session_id not in self.conversations or reset:
                # 优先级：1. 调用时传入的 system_prompt  2. 配置文件中的  3. 硬编码默认值
                fallback_prompt = """你是一个智能语音助手。你的回答将被直接用于语音合成朗读，因此必须遵守以下格式要求：
一，只用纯中文回答，禁止英文、数字、字母。
二，只用中文逗号和句号，禁止其他标点如冒号、问号、感叹号、括号、引号。
三，禁止使用星号、井号、横线等任何符号。
四，禁止使用列表、编号、分点格式，必须写成连贯的一段话。
五，态度温和友好。"""
                
                prompt_to_use = system_prompt or self.default_system_prompt or fallback_prompt
                
                self.conversations[session_id] = {
                    "history": deque(maxlen=self.history_length),
                    "system_prompt": prompt_to_use
                }
            
            conversation = self.conversations[session_id]
            
            # 构建消息 - 根据 emotional_mode 决定是否使用情感感知 prompt
            base_system_prompt = conversation["system_prompt"]
            
            if emotional_mode and self.emotion_strategy and emotion:
                # 情感感知模式：使用特殊的 prompt 引导 LLM 输出 [style]<endofprompt>[content] 格式
                effective_system_prompt = self.emotion_strategy.build_emotional_system_prompt(
                    emotion=emotion, 
                    base_prompt=base_system_prompt
                )
                logger.info(f"[Emotional Mode] 启用情感感知模式，情绪: {emotion}")
            else:
                effective_system_prompt = base_system_prompt
                # 传统模式：只在 query 前加情感标签
                if emotion and emotion != "neutral":
                    emotion_context = f"[用户当前情绪: {emotion}] "
                    query = emotion_context + query
            
            messages = [{"role": "system", "content": effective_system_prompt}]
            
            # RAG 检索（如果启用）
            enhanced_query = query
            if use_rag and self.rag_module:
                try:
                    rag_context = self.rag_module.build_context(query)
                    if rag_context:
                        enhanced_query = f"根据以下参考信息回答用户问题。\n\n{rag_context}\n\n用户问题：{query}"
                        # 更明显的日志输出
                        logger.info(f"[RAG] ✓ 检索成功，上下文长度: {len(rag_context)} 字符")
                        logger.info(f"[RAG] 检索内容预览: {rag_context[:100]}...")
                    else:
                        logger.info(f"[RAG] ✗ 未找到相关知识")
                except Exception as e:
                    logger.warning(f"[RAG] 检索失败: {e}")
            elif use_rag and not self.rag_module:
                logger.info("[RAG] ✗ RAG 模块未初始化")
            
            # 添加历史
            for user_msg, assistant_msg in conversation["history"]:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            
            # 添加当前查询（使用增强后的查询）
            messages.append({"role": "user", "content": enhanced_query})
            
            # 生成回复
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 情感感知模式：解析 LLM 输出为 style + content
            style = ""
            content = response
            
            if emotional_mode and self.emotion_strategy:
                parsed = self.emotion_strategy.parse_emotional_response(response)
                style = parsed.get("style", "")
                content = parsed.get("content", response)
                
                # 如果解析失败（没有分隔符），使用默认风格
                if not style and emotion:
                    style = self.emotion_strategy.get_style_for_emotion(emotion)
                    logger.info(f"[Emotional Mode] LLM 未输出风格，使用默认: {style}")
                
                # 额外检查：确保content不包含风格描述关键词（防止解析失败时残留）
                if content:
                    style_keywords = ["语气", "语速", "语调", "音量", "音调", "节奏", "温暖", "柔和", "明亮", "轻快", "安慰", "耐心"]
                    # 如果content开头包含风格关键词，可能是解析失败，尝试移除
                    content_lines = content.split('\n')
                    if len(content_lines) > 0:
                        first_line = content_lines[0].strip()
                        # 如果第一行包含风格关键词且较短，可能是残留的风格描述
                        if any(kw in first_line for kw in style_keywords) and len(first_line) <= 30:
                            logger.warning(f"[Emotional Mode] 检测到content开头可能包含风格描述，已移除: {first_line}")
                            content = '\n'.join(content_lines[1:]).strip() if len(content_lines) > 1 else ""
                
                logger.info(f"[Emotional Mode] 风格: {style}")
                logger.info(f"[Emotional Mode] 内容: {content[:50]}...")
            
            # 后处理：清理不适合语音合成的符号
            content = self._clean_for_tts(content)
            
            # 更新历史（只保存 content 部分）
            conversation["history"].append((query, content))
            
            result = {
                "response": content.strip(),
                "query": query,
                "session_id": session_id,
                "history": list(conversation["history"]),
                "rag_used": rag_context is not None,
                "rag_context": rag_context
            }
            
            # emotional_mode 时添加额外字段
            if emotional_mode:
                result["style"] = style
                result["content"] = content.strip()
                result["emotional_mode"] = True
                # 不使用 instruct 模式（避免 instruct 文本被读出），只传递 emotion 用于语速调整
                result["emotion"] = emotion
            
            return result
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                "response": "抱歉，我遇到了一些问题，请稍后再试。",
                "query": query,
                "session_id": session_id,
                "error": str(e)
            }
    
    def reset_conversation(self, session_id: str = "default"):
        """重置对话历史"""
        if session_id in self.conversations:
            self.conversations[session_id]["history"].clear()
            logger.info(f"Reset conversation for session {session_id}")
    
    def get_history(self, session_id: str = "default") -> List:
        """获取对话历史"""
        if session_id in self.conversations:
            return list(self.conversations[session_id]["history"])
        return []
    
    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"Deleted session {session_id}")
    
    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        return list(self.conversations.keys())
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_type": "Qwen2.5-0.5B-Instruct",
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_length": self.history_length,
            "active_sessions": len(self.conversations),
            "features": [
                "Lightweight (0.5B parameters)",
                "Optimized for Chinese",
                "Low memory footprint (~1GB)",
                "Fast inference on CPU"
            ]
        }


# 简化版对话模块（用于资源受限环境）
class SimplDialogueModule:
    """简化的对话模块（基于规则）"""
    
    def __init__(self):
        """初始化简化对话模块"""
        self.qa_pairs = {
            "你好": ["你好！我是你的医疗语音助手，有什么可以帮助你的吗？", "你好！很高兴为你服务！"],
            "头疼": ["头疼可能有多种原因，建议您多休息，如果持续不适请就医。", "请问头疼持续多长时间了？建议及时就医检查。"],
            "发烧": ["发烧时请注意多喝水、多休息。如果体温超过38.5°C，建议就医。", "请问体温是多少？有没有其他症状？"],
            "感冒": ["感冒建议多喝温水、充足休息。如果症状加重请及时就医。"],
            "天气": ["抱歉，我暂时无法查询天气信息。"],
            "时间": ["请稍等，让我为你查询当前时间。"],
            "再见": ["再见！祝你身体健康！", "拜拜！期待下次为你服务！"],
            "谢谢": ["不客气！很高兴能帮到你！", "不用谢，这是我应该做的！"],
            "帮助": ["我可以帮你进行健康咨询、用药提醒等。请注意，我的建议仅供参考，具体请咨询专业医生。"]
        }
        self.default_responses = [
            "这是个有趣的问题，让我想想...",
            "我理解你的意思，建议您咨询专业医生获取更准确的答案。",
            "这个问题有点复杂，你能详细说明一下吗？"
        ]
    
    def chat(self, query: str, **kwargs) -> Dict:
        """简单对话"""
        import random
        
        # 关键词匹配
        for keyword, responses in self.qa_pairs.items():
            if keyword in query:
                response = random.choice(responses)
                return {
                    "response": response,
                    "query": query,
                    "method": "keyword_match"
                }
        
        # 默认回复
        response = random.choice(self.default_responses)
        return {
            "response": response,
            "query": query,
            "method": "default"
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 使用简化版进行测试
    dialogue = SimplDialogueModule()
    
    # 测试对话
    queries = ["你好", "我头疼怎么办", "谢谢", "再见"]
    for query in queries:
        result = dialogue.chat(query)
        print(f"用户: {query}")
        print(f"助手: {result['response']}\n")
