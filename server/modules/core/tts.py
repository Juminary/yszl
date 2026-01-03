"""
语音合成模块 (Text-to-Speech)
使用 CosyVoice-300M-Instruct 实现高质量中文语音合成
"""

import sys
import os
import torch
import logging
from typing import Dict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

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
    
    def __init__(self, model_name: str = None, device: str = "cuda"):
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
                   instruct: str = None) -> Dict:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出文件路径
            speaker: 说话人ID (如 "中文女", "中文男")
            language: 语言
            speed: 语速
            style: 风格 (soothing, professional, friendly)
            instruct: 指令文本，用于控制语音风格
        
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
            import torchaudio
            
            # 生成输出路径
            if output_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                temp_dir = Path(base_dir) / "temp"
                temp_dir.mkdir(exist_ok=True)
                output_path = str(temp_dir / f"tts_{abs(hash(text))}.wav")
            
            # 选择说话人
            if speaker is None and self.available_spks:
                # 默认使用第一个可用的说话人
                speaker = self.available_spks[0]
            
            logger.info(f"Synthesizing with speaker: {speaker}, text: {text[:50]}...")
            
            # 使用 inference_sft 进行标准语音合成（不带指令朗读）
            audio_outputs = []
            for output in self.model.inference_sft(
                tts_text=text,
                spk_id=speaker,
                stream=False,
                speed=speed
            ):
                audio_outputs.append(output['tts_speech'])
            
            if audio_outputs:
                # 合并音频
                audio = torch.cat(audio_outputs, dim=1)
                
                # 保存音频 - 使用 soundfile 避免 torchcodec 依赖
                import soundfile as sf
                audio_np = audio.cpu().numpy().T  # (channels, samples) -> (samples, channels)
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(-1, 1)
                sf.write(output_path, audio_np, self.sample_rate)
                
                logger.info(f"TTS success: {output_path}")
                
                return {
                    "audio": audio.cpu().numpy(),
                    "sample_rate": self.sample_rate,
                    "output_path": output_path,
                    "text": text,
                    "speaker": speaker,
                    "instruct": instruct,
                    "method": "cosyvoice"
                }
            else:
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
                                output_path: str = None) -> Dict:
        """
        根据情感合成语音
        
        Args:
            text: 文本
            emotion: 情感 (happy, sad, calm, etc.)
            output_path: 输出路径
        
        Returns:
            合成结果
        """
        # 情感到指令的映射
        emotion_instructions = {
            "happy": "用开心、愉快的语气说话",
            "sad": "用温柔、安慰的语气说话",
            "angry": "用平静、安抚的语气说话",
            "fear": "用舒缓、镇定的语气说话",
            "neutral": "用自然、清晰的语气说话",
            "surprise": "用友好、亲切的语气说话"
        }
        instruct = emotion_instructions.get(emotion, "用自然、清晰的语气说话")
        return self.synthesize(text, output_path, instruct=instruct)
    
    def list_speakers(self):
        """列出可用的说话人"""
        return self.available_spks if hasattr(self, 'available_spks') else []
    
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
                    # 发送 WAV 头部（假设总长度，实际用 chunked transfer）
                    header = self._create_wav_header(len(audio_bytes), self.sample_rate)
                    yield header + audio_bytes
                    first_chunk = False
                    logger.info(f"[Streaming TTS] First chunk sent ({len(audio_bytes)} bytes)")
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
    
    def synthesize_with_emotion(self, text: str, emotion: str, output_path: str = None) -> Dict:
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
