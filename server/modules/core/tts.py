"""
语音合成模块 (Text-to-Speech)
使用 CosyVoice-300M-Instruct 实现高质量中文语音合成
"""

import sys
import os
import torch
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class MockSenseVoice:
    """Lightweight placeholder for the SenseVoice ASR + emotion model."""

    def transcribe(self, audio_path: str) -> Dict[str, str]:
        # In production, call the real SenseVoice model here.
        return {"text": "This is a placeholder user utterance.", "emotion": "neutral"}


class MockCosyVoice:
    """Lightweight placeholder for CosyVoice instruct TTS."""

    def synthesize(self, text: str, instruct: Optional[str] = None, **_: Dict) -> Dict:
        # In production, call CosyVoice.inference_instruct or a wrapped TTSModule.synthesize.
        return {
            "audio": b"",  # placeholder bytes
            "sample_rate": 22050,
            "output_path": None,
            "text": text,
            "instruct": instruct,
            "method": "mock_cosyvoice",
        }


class MockLLMClient:
    """Minimal OpenAI-compatible mock client with a chat() method."""

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Deterministic stub output to make the parsing logic easy to see.
        return "[A comforting adult with warm tone and slower pace] <endofprompt> I am here for you and will help."  # noqa: E501


class EmotionalVoiceChatController:
    """Orchestrates SenseVoice -> LLM -> CosyVoice with emotion-aware styling."""

    def __init__(
        self,
        sense_voice=None,
        cosy_voice=None,
        llm_client=None,
        llm_model: str = "gpt-4.1-mini",
    ) -> None:
        self.sense_voice = sense_voice or MockSenseVoice()
        self.cosy_voice = cosy_voice or MockCosyVoice()
        self.llm_client = llm_client or MockLLMClient(model=llm_model)

        # Maps user emotion to how the AI should sound (do not mirror emotion one-to-one).
        self.emotion_to_style = {
            "sad": "A comforting adult with warm tone and slower pace",
            "angry": "A calm and respectful adult with steady tone and medium-slow pace",
            "happy": "An upbeat young adult with bright tone and lively but controlled pace",
            "anxious": "A reassuring adult with soft tone and measured pace",
            "fear": "A steady guide with gentle tone and unhurried pace",
            "neutral": "A professional adult with clear tone and medium pace",
        }

    def _map_user_emotion(self, user_emotion: str) -> str:
        normalized = (user_emotion or "").lower().strip()
        return self.emotion_to_style.get(normalized, self.emotion_to_style["neutral"])

    def _build_system_prompt(self, target_style: str) -> str:
        return (
            "You are an empathetic, concise dialogue agent for healthcare-style conversations. "
            "Always speak naturally as if talking, not writing. \n"
            "You receive the user's transcribed speech and their detected emotion. \n"
            "Pick a speaking persona that supports the user; never mirror negative emotions back. \n"
            "Output format (strict): [Speaking Style Description] <endofprompt> [Spoken Content]. \n"
            "Speaking Style Description should be 8-18 words, concrete, and mention tone and pace. \n"
            "Spoken Content should be a short 1-3 sentence reply, supportive and on-topic, no lists or bullets. \n"
            f"Recommended style to use: {target_style}."
        )

    def _call_llm(self, user_text: str, user_emotion: str, target_style: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt(target_style)},
            {
                "role": "user",
                "content": f"User Emotion: {user_emotion}\nUser Input: {user_text}",
            },
        ]
        return self.llm_client.chat(messages)

    @staticmethod
    def parse_llm_output(llm_output: str) -> Tuple[str, str]:
        """
        Split "[Style] <endofprompt> [Content]" into (style, content).
        The <endofprompt> tag is the anchor between CosyVoice instruct text and spoken text.
        """
        if not llm_output:
            return "", ""

        if "<endofprompt>" not in llm_output:
            # If the delimiter is missing, treat everything as content to avoid failure.
            return "", llm_output.strip()

        style_part, content_part = llm_output.split("<endofprompt>", maxsplit=1)
        style_text = style_part.strip()
        # Remove outer brackets if present: [Style] -> Style.
        if style_text.startswith("[") and style_text.endswith("]"):
            style_text = style_text[1:-1].strip()

        return style_text, content_part.strip()

    def run_conversation(self, audio_path: str) -> Dict:
        # 1) SenseVoice ASR + emotion.
        sv_result = self.sense_voice.transcribe(audio_path)
        user_text = sv_result.get("text", "")
        user_emotion = sv_result.get("emotion", "neutral")

        # 2) Map user emotion to an AI speaking persona (do not mirror negative affect).
        target_style = self._map_user_emotion(user_emotion)

        # 3) LLM generates both style string and spoken reply.
        llm_output = self._call_llm(user_text=user_text, user_emotion=user_emotion, target_style=target_style)
        style_text, content_text = self.parse_llm_output(llm_output)

        # 4) CosyVoice TTS with instruct_text controlling prosody.
        tts_result = self.cosy_voice.synthesize(text=content_text, instruct=style_text)

        return {
            "sensevoice_text": user_text,
            "sensevoice_emotion": user_emotion,
            "mapped_style": target_style,
            "llm_raw": llm_output,
            "instruct_text": style_text,
            "content_text": content_text,
            "tts_result": tts_result,
        }

# CosyVoice 库路径（从 core/ 往上走: core/ -> modules/ -> libs/cosyvoice/）
COSYVOICE_LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'cosyvoice')
if os.path.exists(COSYVOICE_LIB_PATH):
    # 先添加 Matcha-TTS（必须在 CosyVoice 之前）
    third_party_path = os.path.join(COSYVOICE_LIB_PATH, 'third_party', 'Matcha-TTS')
    if os.path.exists(third_party_path) and third_party_path not in sys.path:
        sys.path.insert(0, third_party_path)
    # 再添加 CosyVoice
    if COSYVOICE_LIB_PATH not in sys.path:
        sys.path.insert(0, COSYVOICE_LIB_PATH)

# 尝试导入 CosyVoice
COSYVOICE_AVAILABLE = False
try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    COSYVOICE_AVAILABLE = True
    logger.info("CosyVoice module imported successfully")
except ImportError as e:
    logger.warning(f"CosyVoice not available: {e}")


class TTSModule:
    """语音合成模块 - 使用CosyVoice-300M-Instruct"""
    
    def __init__(self, model_name: str = None, device: str = "cuda", play_prompt: bool = False):
        """
        初始化TTS模块
        """
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.model = None
        self.sample_rate = 22050
        # 存储已注册的音色克隆 {speaker_id: audio_path}
        self.voice_clones = {}
        # 是否保留提示文本在最终音频中的朗读（默认关闭，避免输出“你好，我是医生”）
        self.play_prompt = play_prompt
        
        # CosyVoice 模型路径（从 core/ 往上走: core/ -> modules/ -> server/）
        models_parent_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'tts')
        model_dir = os.path.join(models_parent_dir, 'CosyVoice-300M-Instruct')
        
        # 如果模型不存在，尝试从 ModelScope 下载
        if COSYVOICE_AVAILABLE and not os.path.exists(model_dir):
            logger.info("CosyVoice model not found locally. Attempting to download from ModelScope...")
            try:
                from modelscope import snapshot_download
                # 下载到 models/tts 目录下，ModelScope 会自动创建子目录
                download_path = snapshot_download('iic/CosyVoice-300M-Instruct', local_dir=model_dir)
                logger.info(f"CosyVoice model downloaded to {download_path}")
            except Exception as e:
                logger.error(f"Failed to download CosyVoice model: {e}")
        
        if COSYVOICE_AVAILABLE and os.path.exists(model_dir):
            logger.info(f"Loading CosyVoice from {model_dir}")
            try:
                # 注意：fp16=True 在处理长文本时会产生 nan（数值溢出），必须使用 fp16=False
                self.model = CosyVoice(model_dir=model_dir, load_jit=False, fp16=False)
                self.sample_rate = self.model.sample_rate
                logger.info(f"CosyVoice loaded successfully. Sample rate: {self.sample_rate}")
                
                # 获取可用的说话人
                self.available_spks = self.model.list_available_spks()
                logger.info(f"Available speakers: {self.available_spks}")
                
                # 加载已注册的音色克隆
                self._load_voice_clones()
                
            except Exception as e:
                logger.error(f"Failed to load CosyVoice: {e}")
                self.model = None
        else:
            if not COSYVOICE_AVAILABLE:
                logger.warning("CosyVoice module not available")
            if not os.path.exists(model_dir):
                logger.warning(f"CosyVoice model not found at {model_dir} and download failed")
    
    def _preprocess_text(self, text: str) -> str:
        """
        增强型文本预处理，确保格式正确，适合TTS合成
        重点处理数字转中文、英文字母空格化、特殊符号过滤
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        if not text or not text.strip():
            return ""
        
        import re
        
        # 内部函数：数字转中文
        def num_to_chinese(num_str: str) -> str:
            """将数字字符串转换为中文汉字"""
            num_map = {
                '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
            }
            result = ''
            for char in num_str:
                if char in num_map:
                    result += num_map[char]
                else:
                    result += char
            return result
        
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 第一步：处理数字和字母组合（如 "B402", "H7N7", "2人"）
        # 匹配模式：字母+数字组合 或 数字+中文单位
        def replace_num_pattern(match):
            matched = match.group(0)
            # 如果包含字母，保留字母但确保前后有空格
            if re.search(r'[a-zA-Z]', matched):
                # 分离字母和数字
                letters = ''.join(re.findall(r'[a-zA-Z]', matched))
                numbers = ''.join(re.findall(r'\d', matched))
                chinese_nums = num_to_chinese(numbers)
                # 确保字母和中文之间有空格
                return f"{letters} {chinese_nums}"
            else:
                # 纯数字，转换为中文
                return num_to_chinese(matched)
        
        # 匹配数字和字母数字组合（如 B402, H7N7, 2人, 3人）
        text = re.sub(r'[A-Za-z]+\d+|\d+[A-Za-z]+|\d+[人个位]', replace_num_pattern, text)
        # 匹配单独的数字（如 "2", "3", "7"）
        text = re.sub(r'\d+', lambda m: num_to_chinese(m.group(0)), text)
        
        # 第二步：处理英文字母，确保前后有空格（防止和中文粘连）
        # 在英文字母和中文之间添加空格
        text = re.sub(r'([a-zA-Z]+)([\u4e00-\u9fa5])', r'\1 \2', text)
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z]+)', r'\1 \2', text)
        
        # 第三步：规范化标点符号
        text = text.replace(',', '，').replace('.', '。')
        text = text.replace('!', '！').replace('?', '？')
        
        # 处理病症列表的停顿：将连续病名之间的逗号替换为句号，增加合成稳定性
        # 匹配模式：中文词 + 逗号 + 中文词（病名列表）
        # 例如："咽喉炎，急性咽炎" -> "咽喉炎。急性咽炎"
        text = re.sub(r'([\u4e00-\u9fa5]+)，([\u4e00-\u9fa5]+)', r'\1。\2', text)
        # 如果还有连续的逗号分隔的病名，继续处理（最多处理3次，避免过度替换）
        for _ in range(3):
            new_text = re.sub(r'([\u4e00-\u9fa5]+)，([\u4e00-\u9fa5]+)', r'\1。\2', text)
            if new_text == text:
                break
            text = new_text
        
        # 第四步：过滤特殊符号，只保留中文、英文、数字、基本标点 [。，！？]
        # 保留：中文、英文字母、数字、句号、逗号、感叹号、问号、空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？\s]', '', text)
        
        # 第五步：清理多余空格
        text = re.sub(r'\s+', ' ', text)
        # 确保标点符号前后没有多余空格
        text = re.sub(r'\s+([，。！？])', r'\1', text)
        text = re.sub(r'([，。！？])\s+', r'\1', text)
        
        # 第六步：确保文本以句号结尾（CosyVoice需要）
        text = text.strip()
        if text and text[-1] not in ['。', '！', '？']:
            text += '。'
        
        # 修复 Input size 0 错误：检查预处理后的文本是否全是非文字符号
        # 如果文本中没有任何中文字符、英文字母或数字，返回空字符串或固定文本
        import re
        has_text_content = bool(re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', text))
        if not has_text_content:
            logger.warning(f"Preprocessed text contains no text content (only punctuation/spaces), returning empty string to prevent Input size 0 error")
            return ""
        
        return text.strip()
    
    def _split_long_text(self, text: str, max_length: int = 200) -> list:
        """
        将长文本分段，确保每段不超过最大长度
        优先使用句号切分，如果没有句号才找逗号
        
        Args:
            text: 要分段的文本
            max_length: 每段最大长度（字符数）
        
        Returns:
            文本段列表
        """
        if len(text) <= max_length:
            return [text]
        
        import re
        
        # 第一步：优先按句号、问号、感叹号分段
        sentences = re.split(r'([。！？])', text)
        
        segments = []
        current_segment = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
            
            if len(current_segment) + len(sentence) <= max_length:
                current_segment += sentence
            else:
                if current_segment:
                    segments.append(current_segment)
                # 如果单个句子就超过max_length，才按逗号进一步分段
                if len(sentence) > max_length:
                    # 按逗号进一步分段（只有在没有句号的情况下才使用）
                    sub_sentences = re.split(r'([，,])', sentence)
                    temp = ""
                    for j in range(0, len(sub_sentences), 2):
                        sub = sub_sentences[j] + (sub_sentences[j+1] if j+1 < len(sub_sentences) else '')
                        if len(temp) + len(sub) <= max_length:
                            temp += sub
                        else:
                            if temp:
                                segments.append(temp)
                            temp = sub
                    current_segment = temp
                else:
                    current_segment = sentence
        
        if current_segment:
            segments.append(current_segment)
        
        return segments if segments else [text]
    
    def synthesize(self, text: str, output_path: str = None, 
                   speaker: str = None, language: str = "zh-cn",
                   speed: float = 1.0, style: str = None,
                   instruct: str = None, voice_clone_id: str = None,
                   max_retries: int = 2) -> Dict:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径
            speaker: 说话人ID (如 "中文女", "中文男")，如果voice_clone_id不为None则忽略此参数
            language: 语言
            speed: 语速
            style: 风格 (soothing, professional, friendly)
            instruct: 指令文本，用于控制语音风格
            voice_clone_id: 音色克隆ID（如果提供，将使用该音色克隆）
            max_retries: 最大重试次数（当合成失败时）
        
        Returns:
            合成结果
        """
        if self.model is None:
            return {
                "audio": None,
                "sample_rate": 0,
                "output_path": None,
                "text": text,
                "error": "CosyVoice model not available"
            }
        
        # 预处理文本
        text = self._preprocess_text(text)
        if not text:
            return {
                "audio": None,
                "sample_rate": 0,
                "output_path": None,
                "text": text,
                "error": "Empty text after preprocessing"
            }
        
        try:
            import torchaudio
            
            # 生成输出路径
            if output_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                temp_dir = Path(base_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)
                output_path = str(temp_dir / f"tts_{abs(hash(text))}.wav")
            
            # 检查是否使用音色克隆
            # 重要：只有当 voice_clone_id 明确存在、非空且在 voice_clones 中时，才使用音色克隆
            # 如果 voice_clone_id 为 None、空字符串或不在 voice_clones 中，使用默认音色
            if voice_clone_id and voice_clone_id.strip() and voice_clone_id in self.voice_clones:
                logger.info(f"Using voice clone: {voice_clone_id} (explicitly requested)")
                # 使用音色克隆
                prompt_wav = self.voice_clones[voice_clone_id]
                
                # 从speaker_db获取prompt_text（如果存在）
                prompt_text = "你好，我是医生"  # 默认提示文本（15字符以内）
                try:
                    speaker_db_path = Path(__file__).parent.parent.parent / "data" / "speaker_db.pkl"
                    if speaker_db_path.exists():
                        import pickle
                        with open(speaker_db_path, 'rb') as f:
                            speaker_db = pickle.load(f)
                        if voice_clone_id in speaker_db:
                            metadata = speaker_db[voice_clone_id].get('metadata', {})
                            if metadata.get('prompt_text'):
                                prompt_text = metadata['prompt_text']
                                # 限制prompt_text长度（建议不超过50字，约10秒）
                                if len(prompt_text) > 50:
                                    logger.warning(f"Prompt text too long ({len(prompt_text)} chars), truncating to 50 chars")
                                    prompt_text = prompt_text[:50]
                except Exception as e:
                    logger.warning(f"Failed to load prompt_text from speaker_db: {e}")
                
                # 强制裁剪 prompt_text 为前15个字符
                # 经验表明：过长的提示文字是导致 sampling reaches max_trials 的元凶
                def clean_prompt_text(prompt: str) -> str:
                    """轻量级清理 prompt_text，强制截断为前15个字符"""
                    if not prompt:
                        return "你好，我是医生"
                    import re
                    # 只去除多余的空白字符，保留所有原始内容
                    prompt = re.sub(r'\s+', ' ', prompt.strip())
                    # 强制截断为前15个字符
                    if len(prompt) > 15:
                        prompt = prompt[:15]
                        logger.info(f"Prompt text truncated to 15 chars: {prompt}")
                    return prompt
                
                prompt_text_processed = clean_prompt_text(prompt_text)
                if not prompt_text_processed:
                    prompt_text_processed = "你好，我是医生"
                
                logger.info(f"Using voice clone: {voice_clone_id}, prompt_text: {prompt_text_processed[:50]}... (length: {len(prompt_text_processed)}), text length: {len(text)} chars")
                
                audio_outputs = []
                # 动态分段策略优化：将分段阈值统一调整为固定的50字符
                # 这个长度对 CosyVoice 最稳定，允许拆分成10-20字的小段，降低LLM采样压力
                text_segments = self._split_long_text(text, max_length=50)
                logger.info(f"Text split into {len(text_segments)} segments for zero-shot voice clone (max_len=50)")
                
                # 记录每个段的处理状态
                segment_status = []
                
                # 辅助函数：尝试用克隆音色合成一段文本
                def try_zero_shot_synthesis(text_to_synth, seg_name="segment", max_attempts=3, add_punctuation_on_error=True, is_first_segment=False):
                    """尝试用零样本克隆合成文本，返回(成功标志, 音频列表, 错误信息)"""
                    # 移除最小字符限制，允许拆分成10-20字的小段，降低LLM采样压力
                    if not text_to_synth or not text_to_synth.strip():
                        return False, [], f"Text is empty"
                    
                    text_processed = self._preprocess_text(text_to_synth)
                    # 预处理后检查：如果为空或全是非文字符号，返回错误
                    if not text_processed or not text_processed.strip():
                        logger.warning(f"{seg_name}: Text became empty after preprocessing (original: {text_to_synth[:30]}...)")
                        return False, [], f"Text became empty after preprocessing"
                    
                    # 废除占位符逻辑：移除所有在文本开头添加"。，"或空格的逻辑
                    # 保持文本干净，只需确保以句号结尾（已在_preprocess_text中处理）
                    
                    zero_shot_speed = min(speed, 0.95) if speed > 0.95 else speed
                    
                    for attempt in range(max_attempts):
                        try:
                            attempt_audio = []
                            current_text = text_processed
                            current_speed = zero_shot_speed
                            
                            # 采样参数微调：如果发生 RuntimeError，第二次重试时降低 speed 0.05，并在文本末尾增加句号
                            if attempt > 0 and add_punctuation_on_error:
                                # 降低 speed 0.05
                                current_speed = max(0.5, current_speed - 0.05)
                                # 确保文本末尾有句号
                                if current_text[-1] not in ['。', '！', '？']:
                                    current_text = current_text + '。'
                                logger.info(f"{seg_name} attempt {attempt+1}: Reduced speed to {current_speed:.2f} and ensured period ending")
                            
                            for output in self.model.inference_zero_shot(
                                tts_text=current_text,
                                prompt_text=prompt_text_processed,
                                prompt_wav=prompt_wav,
                                zero_shot_spk_id=voice_clone_id,
                                stream=False,
                                speed=current_speed,
                                text_frontend=False,
                                play_prompt=self.play_prompt
                            ):
                                if output.get("tts_speech") is not None:
                                    audio_tensor = output["tts_speech"]
                                    if isinstance(audio_tensor, torch.Tensor):
                                        # 检测 NaN 或全零音频
                                        if audio_tensor.numel() > 0:
                                            has_nan = torch.isnan(audio_tensor).any()
                                            is_all_zero = (audio_tensor.abs() < 1e-6).all()
                                            
                                            if has_nan:
                                                logger.warning(f"{seg_name} attempt {attempt+1}: Audio contains NaN, will retry")
                                                attempt_audio = []  # 清空，触发重试
                                                break
                                            elif is_all_zero:
                                                logger.warning(f"{seg_name} attempt {attempt+1}: Audio is all zeros, will retry")
                                                attempt_audio = []  # 清空，触发重试
                                                break
                                            else:
                                                audio_length = audio_tensor.shape[-1] / self.sample_rate
                                                if 0.1 <= audio_length <= 30.0:
                                                    attempt_audio.append(audio_tensor)
                                                else:
                                                    logger.warning(f"{seg_name} attempt {attempt+1}: Audio length abnormal ({audio_length:.2f}s)")
                                        else:
                                            logger.warning(f"{seg_name} attempt {attempt+1}: Audio tensor is empty")
                            
                            if attempt_audio:
                                total_length = sum(a.shape[-1] for a in attempt_audio) / self.sample_rate
                                
                                # 基础长度检查：确保音频长度合理
                                expected_min_length = len(text_processed) * 0.05
                                if total_length >= expected_min_length:
                                    return True, attempt_audio, None
                                else:
                                    logger.warning(f"{seg_name} attempt {attempt+1}: audio too short ({total_length:.2f}s < {expected_min_length:.2f}s), retrying...")
                                    attempt_audio = []  # 清空，触发重试
                            else:
                                raise ValueError("No valid audio produced")
                        except RuntimeError as e:
                            error_str = str(e)
                            # 检查是否是 max_trials 错误
                            if "max_trials" in error_str or "sampling reaches" in error_str:
                                if attempt < max_attempts - 1:
                                    logger.warning(f"{seg_name} attempt {attempt+1}: RuntimeError (max_trials), will retry with punctuation and space modification")
                                    # 下次尝试时会添加标点符号和空格
                                else:
                                    logger.error(f"{seg_name}: All attempts failed with RuntimeError (max_trials): {error_str}")
                                    logger.error(f"  Current prompt_text: {prompt_text_processed}")
                                    logger.error(f"  Failed text: {current_text}")
                                    return False, [], f"RuntimeError (max_trials): {error_str}"
                            else:
                                if attempt < max_attempts - 1:
                                    logger.warning(f"{seg_name} attempt {attempt+1} failed: {e}, retrying...")
                                else:
                                    logger.error(f"  Current prompt_text: {prompt_text_processed}")
                                    logger.error(f"  Failed text: {current_text}")
                                    return False, [], str(e)
                        except Exception as e:
                            if attempt < max_attempts - 1:
                                logger.debug(f"{seg_name} attempt {attempt+1} failed: {e}, retrying...")
                            else:
                                logger.error(f"  Current prompt_text: {prompt_text_processed}")
                                logger.error(f"  Failed text: {current_text}")
                                return False, [], str(e)
                    return False, [], "All attempts failed"

                # 逐段零样本合成，失败时拆分更小的段继续用克隆音色
                for seg_idx, segment in enumerate(text_segments, 1):
                    segment_audio = []
                    clone_success = False
                    segment_error = None
                    
                    # 第一级：尝试直接合成整个segment
                    # 标记首段，用于增强首段权重保护
                    is_first_segment = (seg_idx == 1)
                    clone_success, segment_audio, segment_error = try_zero_shot_synthesis(
                        segment, f"Segment {seg_idx}", max_attempts=max_retries + 1, is_first_segment=is_first_segment
                    )
                    
                    # 第二级：如果失败，拆分成更小的子段（30字符）继续用克隆音色
                    # 移除最小字符限制，允许拆分成10-20字的小段
                    if not clone_success and len(segment) > 30:
                        logger.info(f"Segment {seg_idx} failed, splitting into sub-segments (30 chars) and retrying with clone voice...")
                        sub_segments = self._split_long_text(segment, max_length=30)
                        sub_audio_list = []
                        sub_success_count = 0
                        
                        for sub_idx, sub_seg in enumerate(sub_segments, 1):
                            # 移除最小字符限制，只要不是空字符串就尝试合成
                            if not sub_seg or not sub_seg.strip():
                                logger.warning(f"Segment {seg_idx}-Sub {sub_idx} is empty, skipping...")
                                continue
                            
                            sub_success, sub_audio, sub_err = try_zero_shot_synthesis(
                                sub_seg, f"Segment {seg_idx}-Sub {sub_idx}", max_attempts=2
                            )
                            if sub_success:
                                sub_audio_list.extend(sub_audio)
                                sub_success_count += 1
                            else:
                                logger.warning(f"Segment {seg_idx}-Sub {sub_idx} failed: {sub_err}")
                                # 不再进行第三级拆分，因为严禁 30 字符以下的微段拆分
                                logger.error(f"Segment {seg_idx}-Sub {sub_idx}: Cannot split further (minimum 30 chars required)")
                        
                        if sub_audio_list:
                            segment_audio = sub_audio_list
                            clone_success = True
                            logger.info(f"Segment {seg_idx}: succeeded with sub-segments ({sub_success_count}/{len(sub_segments)} sub-segments succeeded)")
                        else:
                            logger.error(f"Segment {seg_idx}: ALL sub-segments failed with clone voice")
                    elif not clone_success:
                        # 对于短segment（<30字符），不再拆分，直接标记为失败
                        if not segment or not segment.strip():
                            logger.warning(f"Segment {seg_idx} is empty, skipping...")
                            segment_error = f"Segment is empty"
                        else:
                            logger.info(f"Segment {seg_idx} failed, trying more attempts (not splitting segments < 60 chars)...")
                            clone_success, segment_audio, segment_error = try_zero_shot_synthesis(
                                segment, f"Segment {seg_idx} (retry)", max_attempts=5
                            )

                    # 记录段状态
                    if clone_success:
                        segment_status.append(f"段{seg_idx}: 零样本克隆成功")
                    else:
                        segment_status.append(f"段{seg_idx}: 失败 - {segment_error or '未知错误'}")
                        # 记录清晰的错误日志，包含该段的原文和 prompt_text
                        logger.error(f"Segment {seg_idx}/{len(text_segments)}: ALL clone attempts failed!")
                        logger.error(f"  Original segment text: {segment}")
                        logger.error(f"  Current prompt_text: {prompt_text_processed}")
                        logger.error(f"  Error: {segment_error or '未知错误'}")
                        # 注意：这里不再回退到默认音色，保持克隆音色的完整性

                    # 如果段合成成功，添加到输出
                    if segment_audio:
                        # 优化音频拼接：使用0.1秒的淡入淡出（Crossfade）替代硬性静音
                        # 减少段落间的机械感，使语音更自然
                        if audio_outputs and seg_idx > 1:
                            crossfade_duration = 0.1  # 0.1秒淡入淡出
                            crossfade_samples = int(crossfade_duration * self.sample_rate)
                            
                            # 获取前一段的最后部分和当前段的第一部分
                            last_audio = audio_outputs[-1]
                            first_audio = segment_audio[0]
                            
                            # 确保是2D tensor (1, samples)
                            if last_audio.dim() == 1:
                                last_audio = last_audio.unsqueeze(0)
                            if first_audio.dim() == 1:
                                first_audio = first_audio.unsqueeze(0)
                            
                            # 计算交叉淡入淡出的样本数（取两者中较小的）
                            last_samples = last_audio.shape[-1]
                            first_samples = first_audio.shape[-1]
                            actual_crossfade = min(crossfade_samples, last_samples, first_samples)
                            
                            if actual_crossfade > 0:
                                # 创建淡出和淡入的权重
                                fade_out = torch.linspace(1.0, 0.0, actual_crossfade, device=last_audio.device, dtype=last_audio.dtype)
                                fade_in = torch.linspace(0.0, 1.0, actual_crossfade, device=first_audio.device, dtype=first_audio.dtype)
                                
                                # 对前一段的末尾进行淡出
                                if last_samples >= actual_crossfade:
                                    last_audio[:, -actual_crossfade:] = last_audio[:, -actual_crossfade:] * fade_out.unsqueeze(0)
                                
                                # 对当前段的开头进行淡入
                                if first_samples >= actual_crossfade:
                                    first_audio[:, :actual_crossfade] = first_audio[:, :actual_crossfade] * fade_in.unsqueeze(0)
                                
                                # 更新音频
                                audio_outputs[-1] = last_audio
                                segment_audio[0] = first_audio
                            
                            logger.debug(f"Applied {actual_crossfade/self.sample_rate:.3f}s crossfade between segments {seg_idx-1} and {seg_idx}")
                        
                        audio_outputs.extend(segment_audio)
                
                # 记录所有段的处理状态
                logger.info(f"Segment processing summary: {', '.join(segment_status)}")
                
                # 检查是否有段失败
                failed_segments = [i+1 for i, status in enumerate(segment_status) if '失败' in status]
                if failed_segments:
                    logger.warning(f"Warning: {len(failed_segments)} segments failed with clone voice: {failed_segments}")
                    # 如果失败的段数少于总段数，至少部分音频已生成（全部使用克隆音色）
                    if len(failed_segments) < len(text_segments):
                        logger.warning(f"Partial audio generated with clone voice: {len(text_segments) - len(failed_segments)}/{len(text_segments)} segments succeeded")
                        logger.warning(f"Note: Failed segments will be skipped to maintain voice consistency (no default voice fallback)")
                    else:
                        logger.error("All segments failed with clone voice! This is an extreme case.")
                
                # 如果最终仍无音频（极端情况：所有segment和所有子段都失败），才用默认音色合成全文兜底
                # 但这种情况应该很少发生，因为我们已经做了多级分段重试
                if not audio_outputs:
                    logger.error("CRITICAL: All clone voice attempts failed, including all sub-segments and micro-segments!")
                    logger.error("This indicates a serious issue with the voice clone or the text. Using default voice as last resort.")
                    fallback_speaker = speaker or (self.available_spks[0] if self.available_spks else None)
                    if fallback_speaker:
                        logger.warning("Using default voice for full text as emergency fallback (this should rarely happen).")
                        for output in self.model.inference_sft(
                            tts_text=text,
                            spk_id=fallback_speaker,
                            stream=False,
                            speed=speed
                        ):
                            if output.get("tts_speech") is not None:
                                audio_outputs.append(output["tts_speech"])
            else:
                # 使用标准语音合成
                fallback_speaker = speaker or (self.available_spks[0] if self.available_spks else None)
                if fallback_speaker is None:
                    logger.error("No speaker available for synthesis")
                    return {
                        "audio": None,
                        "sample_rate": 0,
                        "output_path": None,
                        "text": text,
                        "error": "No speaker available"
                    }

                speaker = fallback_speaker
                logger.info(f"Synthesizing with speaker: {speaker}, text length: {len(text)} chars")

                # 使用标准语音合成（默认音色），固定分段合成，降低长句截断风险
                audio_outputs = []
                text_segments = self._split_long_text(text, max_length=80)
                if len(text_segments) > 1:
                    logger.info(f"Text split into {len(text_segments)} segments for standard synthesis (max_len=80)")
                
                segment_status = []
                for seg_idx, segment in enumerate(text_segments, 1):
                    segment_audio = []
                    segment_success = False
                    segment_error = None
                    
                    try:
                        # instruct 提示强约束语气；若失败则回退到普通合成
                        if instruct:
                            try:
                                iterator = self.model.inference_instruct(
                                    tts_text=segment,
                                    spk_id=speaker,
                                    instruct_text=instruct,
                                    stream=False,
                                    speed=speed
                                )
                                for output in iterator:
                                    if output.get("tts_speech") is not None:
                                        segment_audio.append(output["tts_speech"])
                                if segment_audio:
                                    segment_success = True
                                    logger.info(f"Segment {seg_idx}/{len(text_segments)}: instruct synthesis success")
                            except Exception as e:
                                logger.warning(f"Segment {seg_idx}/{len(text_segments)}: instruct synthesis failed, falling back to sft: {e}")
                                segment_error = str(e)
                        else:
                            iterator = self.model.inference_sft(
                                tts_text=segment,
                                spk_id=speaker,
                                stream=False,
                                speed=speed
                            )
                            for output in iterator:
                                if output.get("tts_speech") is not None:
                                    segment_audio.append(output["tts_speech"])
                            if segment_audio:
                                segment_success = True
                                logger.info(f"Segment {seg_idx}/{len(text_segments)}: sft synthesis success")
                    except Exception as e:
                        segment_error = str(e)
                        logger.warning(f"Segment {seg_idx}/{len(text_segments)}: synthesis failed: {e}")
                        # 尝试fallback到sft
                        try:
                            segment_audio = []
                            for output in self.model.inference_sft(
                                tts_text=segment,
                                spk_id=speaker,
                                stream=False,
                                speed=speed
                            ):
                                if output.get("tts_speech") is not None:
                                    segment_audio.append(output["tts_speech"])
                            if segment_audio:
                                segment_success = True
                                logger.info(f"Segment {seg_idx}/{len(text_segments)}: fallback sft success")
                        except Exception as fallback_err:
                            logger.error(f"Segment {seg_idx}/{len(text_segments)}: fallback also failed: {fallback_err}")
                    
                    # 记录段状态
                    if segment_success:
                        segment_status.append(f"段{seg_idx}: 成功")
                    else:
                        segment_status.append(f"段{seg_idx}: 失败 - {segment_error or '未知错误'}")
                        logger.error(f"Segment {seg_idx}/{len(text_segments)}: ALL methods failed! Segment text: {segment}")
                    
                    # 如果段合成成功，添加到输出
                    if segment_audio:
                        audio_outputs.extend(segment_audio)
                    else:
                        # 如果段合成失败，尝试进一步拆分
                        if len(segment) > 20:
                            logger.info(f"Attempting to split failed segment {seg_idx} into smaller chunks")
                            sub_segments = self._split_long_text(segment, max_length=40)
                            for sub_seg in sub_segments:
                                try:
                                    for output in self.model.inference_sft(
                                        tts_text=sub_seg,
                                        spk_id=speaker,
                                        stream=False,
                                        speed=speed
                                    ):
                                        if output.get("tts_speech") is not None:
                                            audio_outputs.append(output["tts_speech"])
                                            logger.info(f"Sub-segment of segment {seg_idx} succeeded: {sub_seg[:30]}...")
                                            break
                                except Exception as sub_err:
                                    logger.error(f"Sub-segment also failed: {sub_err}")
                
                # 记录所有段的处理状态
                if len(text_segments) > 1:
                    logger.info(f"Standard synthesis segment summary: {', '.join(segment_status)}")
                    failed_segments = [i+1 for i, status in enumerate(segment_status) if '失败' in status]
                    if failed_segments:
                        logger.warning(f"Warning: {len(failed_segments)} segments failed: {failed_segments}")
            if audio_outputs:
                # 合并音频前，先验证所有音频片段的质量
                valid_audio_outputs = []
                for idx, audio_tensor in enumerate(audio_outputs):
                    if isinstance(audio_tensor, torch.Tensor):
                        if audio_tensor.numel() > 0 and not torch.isnan(audio_tensor).any() and not torch.isinf(audio_tensor).any():
                            # 检查音频是否在合理范围内（避免异常大的值）
                            max_val = audio_tensor.abs().max().item()
                            if max_val < 10.0:  # 合理的音频范围
                                valid_audio_outputs.append(audio_tensor)
                            else:
                                logger.warning(f"Audio segment {idx} has abnormal values (max={max_val:.2f}), normalizing...")
                                # 归一化异常音频
                                audio_tensor = audio_tensor / max_val * 0.95
                                valid_audio_outputs.append(audio_tensor)
                        else:
                            logger.warning(f"Audio segment {idx} is invalid (NaN/Inf/empty), skipping")
                    else:
                        valid_audio_outputs.append(audio_tensor)
                
                if not valid_audio_outputs:
                    logger.error("All audio segments are invalid")
                    return {
                        "audio": None,
                        "sample_rate": 0,
                        "output_path": None,
                        "text": text,
                        "error": "All audio segments are invalid"
                    }
                
                # 合并音频
                audio = torch.cat(valid_audio_outputs, dim=1)
                
                # 检查音频是否有效
                if audio.shape[1] == 0:
                    logger.error("Generated audio is empty")
                    return {
                        "audio": None,
                        "sample_rate": 0,
                        "output_path": None,
                        "text": text,
                        "error": "Generated audio is empty"
                    }
                
                # 保存音频 - 使用 soundfile 避免 torchcodec 依赖
                import soundfile as sf
                audio_np = audio.cpu().numpy().T  # (channels, samples) -> (samples, channels)
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(-1, 1)
                
                # 确保音频数据有效
                if len(audio_np) == 0:
                    logger.error("Audio numpy array is empty")
                    return {
                        "audio": None,
                        "sample_rate": 0,
                        "output_path": None,
                        "text": text,
                        "error": "Audio numpy array is empty"
                    }
                
                sf.write(output_path, audio_np, self.sample_rate)
                
                # 验证文件是否成功创建
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    logger.error(f"Failed to save audio file: {output_path}")
                    return {
                        "audio": None,
                        "sample_rate": 0,
                        "output_path": None,
                        "text": text,
                        "error": "Failed to save audio file"
                    }
                
                logger.info(f"TTS success: {output_path}, duration: {len(audio_np) / self.sample_rate:.2f}s")
                
                result = {
                    "audio": audio.cpu().numpy(),
                    "sample_rate": self.sample_rate,
                    "output_path": output_path,
                    "text": text,
                    "speaker": speaker if not voice_clone_id else None,
                    "voice_clone_id": voice_clone_id if voice_clone_id else None,
                    "instruct": instruct,
                    "method": "cosyvoice_zero_shot" if voice_clone_id else "cosyvoice",
                    "duration": len(audio_np) / self.sample_rate
                }
                return result
            else:
                logger.error("No audio outputs generated")
                return {
                    "audio": None,
                    "sample_rate": 0,
                    "output_path": None,
                    "text": text,
                    "error": "No audio output"
                }
                
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "audio": None,
                "sample_rate": 0,
                "output_path": None,
                "text": text,
                "error": str(e)
            }
    
    def synthesize_with_emotion(self, text: str, emotion: str,
                                output_path: str = None,
                                voice_clone_id: str = None,
                                speaker: str = None,
                                speed: float = 1.0) -> Dict:
        """
        根据情感合成语音
        
        Args:
            text: 文本
            emotion: 情感 (happy, sad, calm, etc.)
            output_path: 输出路径
        
        Returns:
            合成结果
        """
        # 情感到指令/语速的映射，突出听感差异并加入口语化亲近感
        # neutral 不下发 instruct，避免不必要的提示
        emotion_profiles = {
            "happy": {
                "instruct": "语气明亮愉快，语调稍高，语速稍快，偶尔加好的呢、太好了、真不错等亲切口头语",
                "speed": 1.10
            },
            "sad": {
                "instruct": "语气柔软安慰，音量放轻，语速放慢，时不时说嗯嗯、别着急、我在听，传递陪伴",
                "speed": 0.90
            },
            "angry": {
                "instruct": "语气克制冷静，语调中低，语速略快但不强硬，用理解的口吻如我明白、先别急",
                "speed": 1.05
            },
            "fear": {
                "instruct": "语气镇定稳重，音量柔和，语速偏慢，用安抚词如别担心、我陪着你",
                "speed": 0.94
            },
            "anxious": {
                "instruct": "语气安抚耐心，语调柔和，语速中等偏慢，加入慢慢来、先深呼吸这类口头安慰",
                "speed": 0.96
            },
            "surprise": {
                "instruct": "语气轻快明亮，音调上扬，语速稍快，带一点惊喜感如哇、真不错",
                "speed": 1.08
            },
            "neutral": {"instruct": None, "speed": 1.0},
        }

        profile = emotion_profiles.get(emotion, {"instruct": None, "speed": speed})
        instruct = profile.get("instruct")
        # 优先使用情感配置的语速，否则沿用外部速度
        speed = profile.get("speed", speed)
        return self.synthesize(
            text,
            output_path=output_path,
            instruct=instruct,
            voice_clone_id=voice_clone_id,
            speaker=speaker,
            speed=speed
        )
    
    def list_speakers(self):
        """列出可用的说话人"""
        return self.available_spks if hasattr(self, 'available_spks') else []
    
    def _load_voice_clones(self):
        """从声纹数据库和voice_clones目录加载已注册的音色克隆"""
        try:
            # 优先从voice_clones目录加载
            voice_clone_dir = Path(__file__).parent.parent.parent / "data" / "voice_clones"
            if voice_clone_dir.exists():
                for audio_file in voice_clone_dir.glob("*.wav"):
                    speaker_id = audio_file.stem  # 文件名（不含扩展名）作为speaker_id
                    audio_path = str(audio_file)
                    try:
                        # 注册到CosyVoice
                        prompt_text = "你好，我是医生"  # 默认提示文本（15字符以内）
                        # 强制截断为前15个字符
                        if len(prompt_text) > 15:
                            prompt_text = prompt_text[:15]
                        success = self.model.add_zero_shot_spk(
                            prompt_text=prompt_text,
                            prompt_wav=audio_path,
                            zero_shot_spk_id=speaker_id
                        )
                        if success:
                            self.voice_clones[speaker_id] = audio_path
                            logger.info(f"Loaded voice clone from directory: {speaker_id}")
                        else:
                            logger.warning(f"Failed to register voice clone for {speaker_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load voice clone for {speaker_id}: {e}")
            
            # 也从speaker_db加载（如果voice_clones目录中没有）
            speaker_db_path = Path(__file__).parent.parent.parent / "data" / "speaker_db.pkl"
            if speaker_db_path.exists():
                import pickle
                with open(speaker_db_path, 'rb') as f:
                    speaker_db = pickle.load(f)
                
                for speaker_id, speaker_info in speaker_db.items():
                    # 如果已经在voice_clones中加载过，跳过
                    if speaker_id in self.voice_clones:
                        continue
                    
                    # 优先使用 metadata 中的 voice_clone_path，如果没有则使用 audio_path
                    metadata = speaker_info.get('metadata', {})
                    audio_path = metadata.get('voice_clone_path') or speaker_info.get('audio_path')
                    
                    if audio_path and os.path.exists(audio_path):
                        try:
                            # 从 metadata 获取 prompt_text，如果没有则使用默认值
                            prompt_text = metadata.get('prompt_text', "你好，我是医生")
                            
                            # 强制截断 prompt_text 为前15个字符
                            # 经验表明：过长的提示文字是导致 sampling reaches max_trials 的元凶
                            if len(prompt_text) > 15:
                                logger.warning(f"Prompt text too long ({len(prompt_text)} chars) for {speaker_id}, truncating to 15 chars")
                                prompt_text = prompt_text[:15]
                            
                            # 注册到CosyVoice
                            success = self.model.add_zero_shot_spk(
                                prompt_text=prompt_text,
                                prompt_wav=audio_path,
                                zero_shot_spk_id=speaker_id
                            )
                            if success:
                                self.voice_clones[speaker_id] = audio_path
                                logger.info(f"Loaded voice clone from speaker_db: {speaker_id} (path: {audio_path})")
                            else:
                                logger.warning(f"Failed to register voice clone for {speaker_id}")
                        except Exception as e:
                            logger.warning(f"Failed to load voice clone for {speaker_id}: {e}")
            
            logger.info(f"Loaded {len(self.voice_clones)} voice clones total")
        except Exception as e:
            logger.error(f"Failed to load voice clones: {e}")
    
    def register_voice_clone(self, speaker_id: str, audio_path: str, prompt_text: str = None):
        """
        注册音色克隆
        
        Args:
            speaker_id: 说话人ID
            audio_path: 参考音频路径（至少3秒）
            prompt_text: 提示文本（可选，用于控制语音风格）
        
        Returns:
            bool: 是否注册成功
        """
        if self.model is None:
            logger.error("CosyVoice model not available")
            return False
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        try:
            if prompt_text is None:
                prompt_text = "你好，我是医生"
            
            # 重要：不能对 prompt_text 进行破坏性预处理（如数字转中文、字母处理等）
            # 只做轻量级清理，确保参考音频里的文字和模型看到的文字一致
            import re
            prompt_text = re.sub(r'\s+', ' ', prompt_text.strip())
            
            # 强制截断 prompt_text 为前15个字符
            # 经验表明：过长的提示文字是导致 sampling reaches max_trials 的元凶
            original_length = len(prompt_text)
            if len(prompt_text) > 15:
                logger.warning(f"Prompt text too long ({original_length} chars), truncating to 15 chars to prevent max_trials error")
                prompt_text = prompt_text[:15]
            
            # 检查音频长度（建议3-10秒）
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                duration = len(audio_data) / sample_rate
                if duration > 15:
                    logger.warning(f"Audio duration too long ({duration:.2f}s), recommended: 3-10s. This may cause performance issues.")
                elif duration < 2:
                    logger.warning(f"Audio duration too short ({duration:.2f}s), recommended: 3-10s. This may affect voice clone quality.")
            except Exception as e:
                logger.warning(f"Failed to check audio duration: {e}")
            
            # 注册到CosyVoice
            success = self.model.add_zero_shot_spk(
                prompt_text=prompt_text,
                prompt_wav=audio_path,
                zero_shot_spk_id=speaker_id
            )
            
            if success:
                self.voice_clones[speaker_id] = audio_path
                logger.info(f"Registered voice clone: {speaker_id}")
                return True
            else:
                logger.error(f"Failed to register voice clone for {speaker_id}")
                return False
        except Exception as e:
            logger.error(f"Error registering voice clone: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_voice_clones(self):
        """列出所有已注册的音色克隆"""
        return list(self.voice_clones.keys())
    
    def unregister_voice_clone(self, speaker_id: str) -> bool:
        """
        注销音色克隆
        
        Args:
            speaker_id: 要注销的说话人ID
        
        Returns:
            bool: 是否成功注销
        """
        if speaker_id not in self.voice_clones:
            logger.warning(f"Voice clone {speaker_id} not found in registry")
            return False
        
        try:
            # 从字典中移除（CosyVoice模型内部可能无法直接删除，但至少我们不再使用它）
            del self.voice_clones[speaker_id]
            logger.info(f"Unregistered voice clone: {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister voice clone {speaker_id}: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_type": "CosyVoice-300M-Instruct",
            "device": self.device,
            "available": self.model is not None,
            "sample_rate": self.sample_rate,
            "speakers": self.list_speakers(),
            "features": [
                "Instruction-aware synthesis",
                "Emotion-aware speaking style",
                "Zero-shot voice cloning",
                "Natural prosody",
                "Streaming output"  # 新增
            ]
        }
    
    def synthesize_stream(self, text: str, speaker: str = None, speed: float = 1.0):
        """
        流式语音合成 - 边生成边返回音频块
        
        Args:
            text: 要合成的文本
            speaker: 说话人ID
            speed: 语速
            
        Yields:
            bytes: WAV 格式音频数据块
        """
        if self.model is None:
            logger.error("CosyVoice model not available for streaming")
            return
        
        try:
            import struct
            
            # 选择说话人
            if speaker is None and self.available_spks:
                speaker = self.available_spks[0]
            
            logger.info(f"[Streaming TTS] Starting with speaker: {speaker}, text: {text[:50]}...")
            
            # 先发送 WAV 头部（占位，稍后更新）
            # 使用流式模式时，我们无法预知总长度，所以用 chunked transfer
            first_chunk = True
            
            for output in self.model.inference_sft(
                tts_text=text,
                spk_id=speaker,
                stream=True,  # 启用流式
                speed=speed
            ):
                audio_tensor = output['tts_speech']
                
                # 转换为 16-bit PCM
                audio_np = audio_tensor.cpu().numpy().flatten()
                audio_int16 = (audio_np * 32767).astype('int16')
                audio_bytes = audio_int16.tobytes()
                
                if first_chunk:
                    # 发送 WAV 头部（使用一个足够大的占位长度，约10分钟音频）
                    # 这样播放器会持续播放直到数据结束
                    estimated_total_size = self.sample_rate * 2 * 60 * 10  # 10分钟单声道16bit音频
                    header = self._create_wav_header(estimated_total_size, self.sample_rate)
                    yield header + audio_bytes
                    first_chunk = False
                    logger.info(f"[Streaming TTS] First chunk sent ({len(audio_bytes)} bytes, header declares ~10min)")
                else:
                    yield audio_bytes
            
            logger.info("[Streaming TTS] Completed")
            
        except Exception as e:
            logger.error(f"[Streaming TTS] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_wav_header(self, data_size: int, sample_rate: int) -> bytes:
        """创建 WAV 文件头"""
        import struct
        
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        
        # WAV 头部结构
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,  # 文件大小 - 8
            b'WAVE',
            b'fmt ',
            16,  # fmt 块大小
            1,   # 音频格式 (PCM)
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        return header


class SimpleTTSModule:
    """简化的TTS模块（跨平台支持）
    
    优先级：
    1. edge-tts (Microsoft Edge TTS API, 跨平台, 高质量, 需联网)
    2. macOS say (仅macOS, 本地离线)
    """
    
    def __init__(self):
        """初始化简化TTS模块"""
        self.tts_method = None
        
        # 1. 优先尝试 edge-tts (跨平台，高质量)
        try:
            import edge_tts
            self.tts_method = "edge_tts"
            self.voice = "zh-CN-XiaoxiaoNeural"  # 中文女声
            logger.info(f"Using edge-tts for TTS (voice: {self.voice})")
            return
        except ImportError:
            logger.info("edge-tts not available, trying alternatives...")
        
        # 2. macOS say 命令作为备选
        import platform
        if platform.system() == "Darwin":
            self.tts_method = "macos_say"
            logger.info("Using macOS say command for TTS")
            return
        
        logger.warning("No TTS engine available! Install edge-tts: pip install edge-tts")
    
    def synthesize(self, text: str, output_path: str = None, **kwargs) -> Dict:
        """合成语音"""
        try:
            # 生成输出路径
            if output_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                temp_dir = Path(base_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)
                output_path = str(temp_dir / f"tts_{abs(hash(text))}.mp3")
            
            # 1. edge-tts (推荐)
            if self.tts_method == "edge_tts":
                import asyncio
                import edge_tts
                
                async def _synthesize():
                    communicate = edge_tts.Communicate(text, self.voice)
                    mp3_path = output_path.replace('.wav', '.mp3').replace('.aiff', '.mp3')
                    await communicate.save(mp3_path)
                    return mp3_path
                
                # 运行异步任务
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                mp3_path = loop.run_until_complete(_synthesize())
                loop.close()
                
                if os.path.exists(mp3_path):
                    return {
                        "audio": None,
                        "sample_rate": 24000,
                        "output_path": mp3_path,
                        "text": text,
                        "method": "edge_tts"
                    }
            
            # 2. macOS say
            elif self.tts_method == "macos_say":
                import subprocess
                aiff_path = output_path.replace('.mp3', '.aiff').replace('.wav', '.aiff')
                result = subprocess.run(
                    ['say', '-v', 'Tingting', '-o', aiff_path, text],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and os.path.exists(aiff_path):
                    return {
                        "audio": None,
                        "sample_rate": 22050,
                        "output_path": aiff_path,
                        "text": text,
                        "method": "macos_say"
                    }
            
            return {"audio": None, "sample_rate": 0, "output_path": None, "text": text, "error": "No TTS engine"}
            
        except Exception as e:
            logger.error(f"Simple TTS failed: {e}")
            import traceback
            traceback.print_exc()
            return {"audio": None, "sample_rate": 0, "output_path": None, "text": text, "error": str(e)}
    
    def synthesize_with_emotion(self, text: str, emotion: str, output_path: str = None, **kwargs) -> Dict:
        # 简化版只透传
        return self.synthesize(text, output_path)
    
    def get_model_info(self) -> Dict:
        return {
            "model_type": "SimpleTTS",
            "method": self.tts_method,
            "cross_platform": True,
            "available_methods": ["edge_tts", "pyttsx3", "macos_say"]
        }



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试 CosyVoice
    tts = TTSModule(device="cpu")
    print("Model Info:", tts.get_model_info())
    
    if tts.model:
        result = tts.synthesize("你好，欢迎使用医疗语音助手。", style="soothing")
        print("Result:", result)
