"""
GPTQ 对话模块
使用 transformers + auto_gptq 加载 GPTQ (W4A16) 量化模型
"""

import os
import logging
from typing import Optional, Dict, List
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


class GPTQDialogueModule:
    """
    基于 GPTQ 量化的对话模块
    支持 W4A16 等 GPTQ 格式的量化模型
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        history_length: int = 10,
        system_prompt: str = None,
        rag_module = None
    ):
        """
        初始化 GPTQ 对话模块
        
        Args:
            model_path: GPTQ 量化模型的本地路径
            device: 运行设备 (cuda/cpu)
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p 采样参数
            history_length: 保留的对话历史轮数
            system_prompt: 系统提示词
            rag_module: RAG 检索模块（可选）
        """
        import torch
        
        # 设备选择
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.history_length = history_length
        self.rag_module = rag_module
        
        self.default_system_prompt = system_prompt or """你是一个专业的医疗语音助手。
你的回答将被直接用于语音合成朗读，因此必须遵守以下格式要求：
一，只用纯中文回答，禁止英文和数字。
二，只用中文逗号和句号，禁止其他标点。
三，禁止使用列表和编号格式，必须写成连贯的一段话。
四，态度温和友好，像一个耐心的专业护士。"""
        
        # 对话历史管理
        self.conversation_history: Dict[str, deque] = {}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载 GPTQ 量化模型"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"\n[GPTQ] 正在加载模型: {self.model_path}")
        logger.info(f"[GPTQ] Loading model from: {self.model_path}")
        
        # 检查模型路径
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[GPTQ] 模型路径不存在: {self.model_path}")
        
        try:
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 加载 GPTQ 量化模型
            # transformers 4.33+ 原生支持 GPTQ 格式
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            
            if self.device == "mps":
                self.model = self.model.to(self.device)
            elif self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            print(f"[GPTQ] 模型加载成功 (设备: {self.device})")
            logger.info(f"[GPTQ] Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"[GPTQ] 模型加载失败: {e}")
            raise
    
    def chat(
        self,
        query: str,
        session_id: str = "default",
        reset: bool = False,
        system_prompt: str = None,
        use_rag: bool = True,
        emotion: str = None,
        speaker_id: str = None,
        emotional_mode: bool = False
    ) -> Dict:
        """
        对话接口
        
        Args:
            query: 用户输入
            session_id: 会话 ID
            reset: 是否重置对话历史
            system_prompt: 系统提示词（覆盖默认）
            use_rag: 是否使用 RAG 检索
            emotion: 用户情感（可选）
            speaker_id: 说话人 ID（可选）
            emotional_mode: 是否启用情感感知模式
        
        Returns:
            对话结果字典
        """
        import torch
        
        # 初始化或重置会话
        if reset or session_id not in self.conversation_history:
            self.conversation_history[session_id] = deque(maxlen=self.history_length)
        
        history = self.conversation_history[session_id]
        
        # RAG 检索
        rag_context = None
        rag_used = False
        if use_rag and self.rag_module:
            try:
                rag_context = self.rag_module.build_context(query, top_k=3)
                if rag_context:
                    rag_used = True
                    logger.info(f"[GPTQ] RAG 检索成功，上下文长度: {len(rag_context)}")
            except Exception as e:
                logger.warning(f"[GPTQ] RAG 检索失败: {e}")
        
        # 构建消息
        sys_prompt = system_prompt or self.default_system_prompt
        if rag_context:
            sys_prompt += f"\n\n参考信息：\n{rag_context}"
        
        # 添加情感上下文
        enhanced_query = query
        if emotion and emotion != "neutral":
            enhanced_query = f"[用户当前情绪: {emotion}] {query}"
        
        messages = [{"role": "system", "content": sys_prompt}]
        
        # 添加历史对话
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": enhanced_query})
        
        # 生成回复
        try:
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
            
            # 移除可能的思考内容 <think>...</think>
            import re
            think_pattern = r'<think>.*?</think>\s*'
            response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
            
            # 后处理：清理文本
            response = self._clean_for_tts(response)
            
            # 更新历史
            history.append((query, response))
            
            result = {
                'response': response,
                'query': query,
                'session_id': session_id,
                'rag_used': rag_used,
                'rag_context': rag_context if rag_used else None,
                'model': 'gptq',
                'history': list(history)
            }
            
            if emotional_mode:
                result['emotional_mode'] = True
                result['emotion'] = emotion
            
            return result
            
        except Exception as e:
            logger.error(f"[GPTQ] 生成失败: {e}")
            return {
                'response': '抱歉，我现在无法回答，请稍后再试。',
                'query': query,
                'session_id': session_id,
                'error': str(e)
            }
    
    def _clean_for_tts(self, text: str) -> str:
        """清理文本，使其适合语音合成"""
        import re
        
        # 去除 Markdown 格式符号
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+', '', text)
        text = re.sub(r'`+', '', text)
        text = re.sub(r'~+', '', text)
        text = re.sub(r'_+', '', text)
        
        # 去除列表编号格式
        text = re.sub(r'^\s*\d+[\.、\)]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
        
        # 替换不适合朗读的标点
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
        
        # 合并多余的标点
        text = re.sub(r'[，,]+', '，', text)
        text = re.sub(r'[。\.]+', '。', text)
        text = re.sub(r'，。', '。', text)
        text = re.sub(r'。，', '。', text)
        
        # 去除多余空白
        text = re.sub(r'\s+', '', text)
        
        # 去除开头的标点
        text = re.sub(r'^[，。,\.]+', '', text)
        
        return text.strip()
    
    def reset_conversation(self, session_id: str = "default"):
        """重置对话历史"""
        if session_id in self.conversation_history:
            self.conversation_history[session_id].clear()
            logger.info(f"[GPTQ] Reset conversation for session {session_id}")
    
    def get_history(self, session_id: str = "default") -> List:
        """获取对话历史"""
        if session_id in self.conversation_history:
            return list(self.conversation_history[session_id])
        return []
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "model_type": "GPTQ (W4A16)",
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_length": self.history_length,
            "active_sessions": len(self.conversation_history),
            "features": [
                "GPTQ W4A16 Quantization",
                "4-bit weights, 16-bit activations",
                "GPU acceleration supported",
                "~4x memory reduction"
            ]
        }
