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
    
    def synthesize(self, text: str, output_path: str = None,
                   speaker: str = None, language: str = "zh-cn",
                   speed: float = 1.0, style: str = None,
                   instruct: str = None, voice_clone_id: str = None,
                   target_sample_rate: int | None = None) -> Dict:
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
            target_sample_rate: 若提供且不同于模型采样率，则对输出做重采样
        
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

        try:
            # 生成输出路径
            if output_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                temp_dir = Path(base_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)
                output_path = str(temp_dir / f"tts_{abs(hash(text))}.wav")

            # 将长文本拆句，避免一次性推理被截断
            text_segments = self._split_text(text, max_len=120)

            audio_outputs = []

            # 检查是否使用音色克隆
            if voice_clone_id and voice_clone_id in self.voice_clones:
                # 使用音色克隆
                prompt_wav = self.voice_clones[voice_clone_id]

                # 从 speaker_db 获取 prompt_text（如果存在）
                prompt_text = "你好，我是医生，很高兴为您服务。"  # 默认提示文本
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
                                # 限制 prompt_text 长度（建议不超过 50 字，约 10 秒）
                                if len(prompt_text) > 50:
                                    logger.warning(f"Prompt text too long ({len(prompt_text)} chars), truncating to 50 chars")
                                    prompt_text = prompt_text[:50]
                except Exception as e:
                    logger.warning(f"Failed to load prompt_text from speaker_db: {e}")

                logger.info(
                    f"Using voice clone: {voice_clone_id}, "
                    f"prompt_text: {prompt_text[:30]}... ({len(prompt_text)} chars), "
                    f"text: {text[:50]}..."
                )

                try:
                    for segment in text_segments:
                        # 强制不朗读提示词，避免输出中带 prompt
                        for output in self.model.inference_zero_shot(
                            tts_text=segment,
                            prompt_text=prompt_text,
                            prompt_wav=prompt_wav,
                            zero_shot_spk_id=voice_clone_id,
                            stream=False,
                            speed=speed,
                            play_prompt=False
                        ):
                            if 'tts_speech' in output and output['tts_speech'] is not None:
                                audio_outputs.append(output['tts_speech'])

                    # 如果音色克隆失败（没有输出），回退到默认音色
                    if not audio_outputs:
                        logger.warning("Voice clone synthesis produced no output, falling back to default voice")
                        raise ValueError("Voice clone synthesis produced no output")
                except Exception as e:
                    logger.error(f"Voice clone synthesis failed: {e}, falling back to default voice")
                    # 回退到默认音色
                    voice_clone_id = None
                    audio_outputs = []

            # 如果没有使用 / 或已经从音色克隆回退，则使用标准语音合成
            if not audio_outputs:
                # 选择说话人
                if speaker is None and self.available_spks:
                    # 默认使用第一个可用的说话人
                    speaker = self.available_spks[0]

                logger.info(f"Synthesizing with speaker: {speaker}, text: {text[:50]}...")

                for segment in text_segments:
                    try:
                        # instruct 提示强约束语气；若失败则回退到普通合成
                        if instruct:
                            iterator = self.model.inference_instruct(
                                tts_text=segment,
                                spk_id=speaker,
                                instruct_text=instruct,
                                stream=False,
                                speed=speed
                            )
                        else:
                            iterator = self.model.inference_sft(
                                tts_text=segment,
                                spk_id=speaker,
                                stream=False,
                                speed=speed
                            )

                        for output in iterator:
                            if 'tts_speech' in output and output['tts_speech'] is not None:
                                audio_outputs.append(output['tts_speech'])
                    except Exception as e:
                        logger.warning(f"Instruct synthesis failed, falling back to sft: {e}")
                        for output in self.model.inference_sft(
                            tts_text=segment,
                            spk_id=speaker,
                            stream=False,
                            speed=speed
                        ):
                            if 'tts_speech' in output and output['tts_speech'] is not None:
                                audio_outputs.append(output['tts_speech'])

            if audio_outputs:
                # 合并音频（audio_outputs 是 [1, T] 的列表，按时间维拼接）
                audio = torch.cat(audio_outputs, dim=1)
                
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

                # 若用户指定了输出采样率，则在保存前重采样（模型内部仍用 self.sample_rate）
                audio = audio.cpu()
                final_sample_rate = self.sample_rate
                if target_sample_rate and target_sample_rate != self.sample_rate:
                    try:
                        import torchaudio
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=self.sample_rate,
                            new_freq=target_sample_rate
                        )
                        audio = resampler(audio)
                        final_sample_rate = target_sample_rate
                    except Exception as e:
                        logger.warning(f"Resample to {target_sample_rate} Hz failed, fallback to {self.sample_rate}: {e}")

                audio_np = audio.numpy().T  # (channels, samples) -> (samples, channels)
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
                
                sf.write(output_path, audio_np, final_sample_rate)
                
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
                
                logger.info(f"TTS success: {output_path}, duration: {len(audio_np) / final_sample_rate:.2f}s")
                
                result = {
                    "audio": audio.numpy(),
                    "sample_rate": final_sample_rate,
                    "output_path": output_path,
                    "text": text,
                    "speaker": speaker if not voice_clone_id else None,
                    "voice_clone_id": voice_clone_id if voice_clone_id else None,
                    "instruct": instruct,
                    "method": "cosyvoice_zero_shot" if voice_clone_id else "cosyvoice",
                    "duration": len(audio_np) / final_sample_rate
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

    def _split_text(self, text: str, max_len: int = 120):
        """
        将长文本拆分为短句，减少TTS截断风险
        """
        import re
        if len(text) <= max_len:
            return [text]

        # 按句号/问号/感叹号/分号切分
        sentences = re.split(r'(?<=[。！？!?；;])', text)
        chunks = []
        buffer = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidate = buffer + sent
            if len(candidate) <= max_len:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sent
        if buffer:
            chunks.append(buffer)

        # 如果拆分失败，至少返回原文本
        return chunks or [text]
    
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
                        prompt_text = "你好，我是医生，很高兴为您服务。"  # 默认提示文本
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
                            prompt_text = metadata.get('prompt_text', "你好，我是医生，很高兴为您服务。")
                            
                            # 限制prompt_text长度（建议不超过50字）
                            if len(prompt_text) > 50:
                                logger.warning(f"Prompt text too long ({len(prompt_text)} chars) for {speaker_id}, truncating to 50 chars")
                                prompt_text = prompt_text[:50]
                            
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
                prompt_text = "你好，我是医生，很高兴为您服务。"
            
            # 检查并限制prompt_text长度（建议不超过50字）
            original_length = len(prompt_text)
            if len(prompt_text) > 50:
                logger.warning(f"Prompt text too long ({original_length} chars), truncating to 50 chars for better performance")
                prompt_text = prompt_text[:50]
            
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
