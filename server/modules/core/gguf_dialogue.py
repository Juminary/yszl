"""
GGUF 对话模块
使用 llama-cpp-python 加载 GGUF 格式的量化模型
"""

import os
import logging
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


def download_gguf_model(repo: str, filename: str, target_dir: str, source: str = "huggingface") -> str:
    """
    下载 GGUF 模型文件
    
    Args:
        repo: 仓库名称 (如 unsloth/Qwen3-4B-GGUF)
        filename: 文件名 (如 Qwen3-4B-Q4_K_M.gguf)
        target_dir: 目标目录
        source: 下载源 (huggingface 或 modelscope)
    
    Returns:
        下载后的文件路径
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    
    if os.path.exists(target_path):
        logger.info(f"[GGUF] 模型已存在: {target_path}")
        return target_path
    
    print(f"\n[GGUF] 正在下载模型: {repo}/{filename}")
    print(f"[GGUF] 目标路径: {target_path}")
    print(f"[GGUF] 这可能需要几分钟，请耐心等待...")
    
    if source == "huggingface":
        try:
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            print(f"[GGUF] 下载完成: {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.error(f"[GGUF] HuggingFace 下载失败: {e}")
            raise
    
    elif source == "modelscope":
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download(repo, cache_dir=target_dir)
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                return model_path
            # 查找 gguf 文件
            for f in os.listdir(model_dir):
                if f.endswith('.gguf'):
                    return os.path.join(model_dir, f)
            raise FileNotFoundError(f"未找到 GGUF 文件: {model_dir}")
        except Exception as e:
            logger.error(f"[GGUF] ModelScope 下载失败: {e}")
            raise
    
    else:
        raise ValueError(f"不支持的下载源: {source}")


class GGUFDialogueModule:
    """
    基于 llama-cpp-python 的对话模块
    支持 GGUF 格式的量化模型
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        system_prompt: str = None,
        rag_module = None
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.rag_module = rag_module
        
        self.default_system_prompt = system_prompt or """你是一个专业的医疗语音助手。
你的回答将被直接用于语音合成朗读，因此必须遵守以下格式要求：
一，只用纯中文回答，禁止英文和数字。
二，只用中文逗号和句号，禁止其他标点。
三，禁止使用列表和编号格式，必须写成连贯的一段话。
四，态度温和友好，像一个耐心的专业护士。"""
        
        self.conversation_history: Dict[str, List[Dict]] = {}
        self._load_model(n_ctx, n_threads, n_gpu_layers)
    
    def _load_model(self, n_ctx: int, n_threads: int, n_gpu_layers: int):
        """加载 GGUF 模型"""
        try:
            from llama_cpp import Llama
            
            print(f"\n[GGUF] 正在加载模型: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            print(f"[GGUF] 模型加载成功 (Metal GPU 加速已启用)")
            logger.info(f"[GGUF] 模型加载成功")
            
        except ImportError:
            raise RuntimeError("llama-cpp-python 未安装，请运行: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"[GGUF] 模型加载失败: {e}")
            raise
    
    def chat(
        self,
        query: str,
        session_id: str = "default",
        reset: bool = False,
        system_prompt: str = None,
        use_rag: bool = True
    ) -> Dict:
        """对话接口"""
        if reset or session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        history = self.conversation_history[session_id]
        
        # RAG 检索
        rag_context = ""
        rag_used = False
        if use_rag and self.rag_module:
            try:
                rag_context = self.rag_module.build_context(query, top_k=3) or ""
                if rag_context:
                    rag_used = True
            except Exception as e:
                logger.warning(f"[GGUF] RAG 检索失败: {e}")
        
        # 构建消息
        messages = []
        sys_prompt = system_prompt or self.default_system_prompt
        if rag_context:
            sys_prompt += f"\n\n参考信息：\n{rag_context}"
        messages.append({"role": "system", "content": sys_prompt})
        
        for msg in history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": query})
        
        # 生成回复
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            
            assistant_message = response['choices'][0]['message']['content']
            
            # 移除 Qwen3 的思考内容 <think>...</think>
            import re
            think_pattern = r'<think>.*?</think>\s*'
            clean_message = re.sub(think_pattern, '', assistant_message, flags=re.DOTALL)
            clean_message = clean_message.strip()
            
            # 如果清理后为空，使用原始消息
            if not clean_message:
                clean_message = assistant_message
            
            # 更新历史（保存原始消息）
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": assistant_message})
            
            return {
                'response': clean_message,  # 返回清理后的消息
                'rag_used': rag_used,
                'rag_context': rag_context if rag_used else '',
                'model': 'gguf'
            }
            
        except Exception as e:
            logger.error(f"[GGUF] 生成失败: {e}")
            return {
                'response': '抱歉，我现在无法回答，请稍后再试。',
                'error': str(e)
            }
    
    def reset_history(self, session_id: str = None):
        """重置对话历史"""
        if session_id:
            self.conversation_history[session_id] = []
        else:
            self.conversation_history.clear()
