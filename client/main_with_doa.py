"""
树莓派客户端主程序 - DOA增强版
整合 ODAS 声源定位功能

功能:
- 语音交互 (ASR + 对话 + TTS)
- 实时声源定位 (DOA)
- 多声源跟踪
- DOA 信息与后端集成
"""

import requests
import logging
import yaml
import argparse
import subprocess
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict

# 添加父目录以导入 acoustic_frontend
sys.path.insert(0, str(Path(__file__).parent))

from audio_capture import AudioCapture
from audio_player import AudioPlayer
from wakeword_detector import WakeWordDetector

# 导入声学前端模块
from acoustic_frontend.odas_client import ODASClient, TrackedSource
from acoustic_frontend.beamformer import Beamformer
import numpy as np
import wave
import io

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client_doa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiChannelCapture:
    """
    多通道音频采集
    
    使用 arecord 采集 8 通道原始音频，用于波束成形
    """
    
    def __init__(self, card_id: int = 3, sample_rate: int = 16000, num_channels: int = 8):
        self.card_id = card_id
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.bytes_per_sample = 4  # S32_LE
        
        # WebRTC VAD (更准确的语音端点检测)
        self._vad = None
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(2)  # 模式 2: 中等灵敏度
            logger.info("MultiChannelCapture: WebRTC VAD enabled")
        except ImportError:
            logger.warning("webrtcvad not installed, using energy-based VAD")
        
    def record(self, duration: float = 5.0) -> Optional[np.ndarray]:
        """
        录制多通道音频
        
        Args:
            duration: 录音时长（秒）
            
        Returns:
            音频数据 shape=(samples, 8)，或 None（失败时）
        """
        try:
            cmd = [
                'arecord',
                '-D', f'hw:{self.card_id},0',
                '-f', 'S32_LE',
                '-r', str(self.sample_rate),
                '-c', str(self.num_channels),
                '-d', str(int(duration)),
                '-t', 'raw',
                '-q',
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=duration + 5)
            
            if result.returncode != 0:
                logger.error(f"arecord failed: {result.stderr.decode()}")
                return None
            
            # 解析原始数据
            raw_data = result.stdout
            audio = np.frombuffer(raw_data, dtype=np.int32)
            
            # 重塑为 (samples, channels)
            num_samples = len(audio) // self.num_channels
            audio = audio[:num_samples * self.num_channels].reshape(num_samples, self.num_channels)
            
            # 转换为 float32 并归一化
            audio = audio.astype(np.float32) / (2**31)
            
            return audio
            
        except subprocess.TimeoutExpired:
            logger.error("arecord timeout")
            return None
        except Exception as e:
            logger.error(f"MultiChannelCapture error: {e}")
            return None
    
    def record_with_vad(
        self, 
        max_duration: float = 30.0,
        silence_duration: float = 1.0,
        energy_threshold: float = 0.005,
        frame_duration_ms: int = 30
    ) -> Optional[np.ndarray]:
        """
        带 VAD 的多通道录音
        
        使用 WebRTC VAD 进行语音端点检测，持续录音直到检测到足够长的静音
        
        Args:
            max_duration: 最大录音时长
            silence_duration: 静音判定时长（秒）
            energy_threshold: 能量阈值（备用）
            frame_duration_ms: VAD 帧长度（ms，必须是 10/20/30）
            
        Returns:
            音频数据 shape=(samples, 8)
        """
        try:
            cmd = [
                'arecord',
                '-D', f'hw:{self.card_id},0',
                '-f', 'S32_LE',
                '-r', str(self.sample_rate),
                '-c', str(self.num_channels),
                '-t', 'raw',
                '-q',
                '-'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            chunks = []
            # VAD 帧: 30ms = 480 samples @ 16kHz
            frame_samples = int(self.sample_rate * frame_duration_ms / 1000)
            frame_bytes = frame_samples * self.num_channels * self.bytes_per_sample
            
            silence_frames = 0
            speech_started = False
            total_samples = 0
            max_samples = int(max_duration * self.sample_rate)
            silence_frames_threshold = int(silence_duration * 1000 / frame_duration_ms)
            
            logger.info("MultiChannelCapture: VAD recording started")
            
            while total_samples < max_samples:
                raw_data = process.stdout.read(frame_bytes)
                if not raw_data or len(raw_data) < frame_bytes:
                    break
                
                # 解析 S32_LE → int32
                audio_frame = np.frombuffer(raw_data, dtype=np.int32)
                num_samples = len(audio_frame) // self.num_channels
                audio_frame = audio_frame[:num_samples * self.num_channels].reshape(num_samples, self.num_channels)
                
                total_samples += num_samples
                
                # 提取第一个麦克风通道做 VAD（需要转 16-bit）
                mono_float = audio_frame[:, 0].astype(np.float32) / (2**31)
                mono_int16 = (mono_float * 32767).astype(np.int16)
                
                # 判断是否有语音
                is_speech = False
                if self._vad is not None:
                    try:
                        is_speech = self._vad.is_speech(mono_int16.tobytes(), self.sample_rate)
                    except Exception:
                        # VAD 失败，用能量判断
                        energy = np.sqrt(np.mean(mono_float ** 2))
                        is_speech = energy > energy_threshold
                else:
                    # 无 VAD，用能量判断
                    energy = np.sqrt(np.mean(mono_float ** 2))
                    is_speech = energy > energy_threshold
                
                if is_speech:
                    speech_started = True
                    silence_frames = 0
                    chunks.append(audio_frame)
                elif speech_started:
                    chunks.append(audio_frame)
                    silence_frames += 1
                    if silence_frames >= silence_frames_threshold:
                        logger.info(f"MultiChannelCapture: Speech ended after {total_samples/self.sample_rate:.1f}s")
                        break
            
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            
            if not chunks:
                logger.warning("MultiChannelCapture: No speech detected")
                return None
            
            # 合并
            audio = np.vstack(chunks)
            audio_float = audio.astype(np.float32) / (2**31)
            
            logger.info(f"MultiChannelCapture: Recorded {len(audio_float)/self.sample_rate:.2f}s ({len(chunks)} frames)")
            return audio_float
            
        except Exception as e:
            logger.error(f"MultiChannelCapture VAD error: {e}")
            import traceback
            traceback.print_exc()
            return None


class MicGainManager:
    """麦克风增益管理器"""
    
    def __init__(self, card_id: int = 3, adc_gain: int = 8, digital_volume: int = 160):
        self.card_id = card_id
        self.adc_gain = adc_gain
        self.digital_volume = digital_volume
    
    def set_gains(self) -> bool:
        """设置麦克风增益"""
        try:
            for i in range(1, 9):
                subprocess.run(
                    ['amixer', '-c', str(self.card_id), 'cset', f'name=ADC{i} PGA gain', str(self.adc_gain)],
                    capture_output=True, check=False
                )
                subprocess.run(
                    ['amixer', '-c', str(self.card_id), 'cset', f'name=CH{i} digital volume', str(self.digital_volume)],
                    capture_output=True, check=False
                )
            logger.info(f"Mic gains set: ADC={self.adc_gain}, Digital={self.digital_volume}")
            return True
        except Exception as e:
            logger.error(f"Failed to set mic gains: {e}")
            return False
    
    def check_gains(self) -> Dict[str, int]:
        """检查当前增益设置"""
        try:
            result = subprocess.run(
                ['amixer', '-c', str(self.card_id), 'cget', 'name=ADC1 PGA gain'],
                capture_output=True, text=True
            )
            # 解析 values=X
            for line in result.stdout.split('\n'):
                if 'values=' in line:
                    value = int(line.split('values=')[1].split()[0])
                    return {'adc_gain': value}
            return {}
        except Exception as e:
            logger.error(f"Failed to check gains: {e}")
            return {}


class ODASProcessManager:
    """ODAS 进程管理器"""
    
    def __init__(self, odas_dir: str = None):
        if odas_dir is None:
            odas_dir = Path(__file__).parent / "odas"
        self.odas_dir = Path(odas_dir)
        self.odas_binary = self.odas_dir / "odas_build" / "build" / "bin" / "odaslive"
        self.config_file = self.odas_dir / "respeaker_6mic.cfg"
        self.process: Optional[subprocess.Popen] = None
        self._log_file: Optional[Path] = Path('/tmp/odas.log')
    
    def start(self) -> bool:
        """启动 ODAS 进程"""
        if not self.odas_binary.exists():
            logger.error(f"ODAS binary not found: {self.odas_binary}")
            logger.info("请先运行: cd client/odas && ./start_odas.sh install")
            return False
        
        if not self.config_file.exists():
            logger.error(f"ODAS config not found: {self.config_file}")
            return False
        
        # 检查是否已运行
        if self.process and self.process.poll() is None:
            logger.warning("ODAS already running")
            return True
        
        # 杀掉可能存在的旧进程
        self._kill_existing_odas()
        
        try:
            # 动态更新配置文件中的声卡 ID
            card_id = self._detect_card_id()
            if card_id is not None:
                runtime_config = self._create_runtime_config(card_id)
                logger.info(f"Detected sound card ID: {card_id}")
            else:
                runtime_config = str(self.config_file)
                logger.warning("Could not detect sound card, using default config")
            
            # 打开日志文件
            log_file = open(self._log_file, 'w')
            
            self.process = subprocess.Popen(
                [str(self.odas_binary), '-c', runtime_config],
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            
            time.sleep(1.5)  # 等待 ODAS 启动
            
            if self.process.poll() is None:
                logger.info(f"ODAS started with PID {self.process.pid}")
                return True
            else:
                logger.error(f"ODAS failed to start, check {self._log_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start ODAS: {e}")
            return False
    
    def _kill_existing_odas(self):
        """杀掉可能存在的旧 ODAS 进程"""
        try:
            subprocess.run(['pkill', '-f', 'odaslive'], capture_output=True)
            time.sleep(0.5)
        except Exception:
            pass
    
    def stop(self):
        """停止 ODAS 进程"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("ODAS stopped")
    
    def is_running(self) -> bool:
        """检查 ODAS 是否运行中"""
        return self.process is not None and self.process.poll() is None
    
    def _detect_card_id(self) -> Optional[int]:
        """检测 ReSpeaker 声卡 ID"""
        try:
            result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'seeed' in line.lower():
                    # 提取 card X
                    import re
                    match = re.search(r'card (\d+)', line)
                    if match:
                        return int(match.group(1))
        except Exception:
            pass
        return None
    
    def _create_runtime_config(self, card_id: int) -> str:
        """创建运行时配置文件"""
        runtime_config = '/tmp/odas_runtime.cfg'
        try:
            with open(self.config_file, 'r') as f:
                content = f.read()
            
            # 替换 card ID
            import re
            content = re.sub(r'card = \d+;', f'card = {card_id};', content)
            
            with open(runtime_config, 'w') as f:
                f.write(content)
            
            return runtime_config
        except Exception as e:
            logger.warning(f"Failed to create runtime config: {e}")
            return str(self.config_file)


class VoiceAssistantWithDOA:
    """
    语音助手客户端 - DOA增强版
    
    整合 ODAS 声源定位功能
    """
    
    def __init__(self, config_path: str = None):
        # 自动查找配置文件
        if config_path is None:
            possible_paths = [
                Path(__file__).parent.parent / "config" / "config.yaml",
                Path("../config/config.yaml"),
                Path("config/config.yaml"),
            ]
            for p in possible_paths:
                if p.exists():
                    config_path = str(p)
                    break
            else:
                config_path = "../config/config.yaml"
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 服务器地址
        self.server_url = self.config.get('client', {}).get('server_url', 'http://localhost:5001')
        
        # 初始化音频模块
        self.capture = AudioCapture(
            sample_rate=self.config.get('audio', {}).get('sample_rate', 16000),
            channels=self.config.get('audio', {}).get('channels', 1)
        )
        self.player = AudioPlayer()
        
        # 会话ID
        self.session_id = f"raspberrypi_doa_{int(time.time())}"
        
        # 流式 TTS 设置
        self.use_streaming_tts = self.config.get('tts', {}).get('streaming', True)
        
        # ===== DOA 相关组件 =====
        # 麦克风增益管理
        self.mic_gain = MicGainManager(
            card_id=self.config.get('odas', {}).get('card_id', 3),
            adc_gain=self.config.get('odas', {}).get('adc_gain', 8),
            digital_volume=self.config.get('odas', {}).get('digital_volume', 160)
        )
        
        # ODAS 进程管理
        self.odas_manager = ODASProcessManager()
        
        # ODAS 客户端 (Python Socket 接收器)
        self.odas_client = ODASClient(
            sst_port=self.config.get('odas', {}).get('sst_port', 9000),
            sss_port=self.config.get('odas', {}).get('ssl_port', 9001)
        )
        
        # DOA 状态
        self._doa_enabled = False
        self._current_sources: List[TrackedSource] = []
        self._doa_lock = threading.Lock()
        
        # ===== 波束成形相关组件 =====
        self._beamforming_enabled = self.config.get('beamforming', {}).get('enabled', True)
        
        # 多通道音频采集
        self.multichannel_capture = MultiChannelCapture(
            card_id=self.config.get('odas', {}).get('card_id', 3),
            sample_rate=self.config.get('audio', {}).get('sample_rate', 16000)
        )
        
        # 波束成形器
        # ReSpeaker 6-Mic 阵列配置
        # 麦克风角度: 官方配置对应 0°, 60°, 120°, 180°, 240°, 300°
        self.beamformer = Beamformer(
            sample_rate=self.config.get('audio', {}).get('sample_rate', 16000),
            mic_angles=[0, 60, 120, 180, 240, 300],
            array_radius=0.0463  # ReSpeaker 6-Mic 阵列半径
        )
        
        logger.info(f"VoiceAssistantWithDOA initialized")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Beamforming: {'enabled' if self._beamforming_enabled else 'disabled'}")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {}
    
    # ==================== DOA 功能 ====================
    
    def start_doa(self) -> bool:
        """
        启动 DOA 系统
        
        启动顺序:
        1. 设置麦克风增益
        2. 启动 ODAS 客户端 (Python 监听)
        3. 启动 ODAS 进程 (C++)
        """
        logger.info("Starting DOA system...")
        
        # 1. 设置麦克风增益
        self.mic_gain.set_gains()
        
        # 2. 启动 ODAS 客户端 (先启动，作为服务器等待 ODAS 连接)
        self.odas_client.start()
        logger.info("ODAS client started (waiting for ODAS to connect)")
        
        # 等待监听器就绪
        time.sleep(0.5)
        
        # 3. 启动 ODAS 进程
        if self.odas_manager.start():
            # 等待 ODAS 连接
            time.sleep(2)
            
            if self.odas_client.is_connected():
                self._doa_enabled = True
                logger.info("✅ DOA system started successfully")
                return True
            else:
                logger.warning("ODAS started but not connected to client")
                # 继续运行，可能稍后会连接
                self._doa_enabled = True
                return True
        else:
            logger.error("Failed to start ODAS process")
            return False
    
    def stop_doa(self):
        """停止 DOA 系统"""
        self._doa_enabled = False
        self.odas_manager.stop()
        self.odas_client.stop()
        logger.info("DOA system stopped")
    
    def get_current_doa(self) -> Optional[float]:
        """
        获取当前主声源的 DOA 角度
        
        Returns:
            方位角 (0-360°)，无声源时返回 None
        """
        if not self._doa_enabled:
            return None
        
        sources = self.odas_client.get_tracked_sources(active_only=True)
        if sources:
            return sources[0].azimuth
        return None
    
    def get_tracked_sources(self) -> List[TrackedSource]:
        """获取所有跟踪的声源"""
        if not self._doa_enabled:
            return []
        return self.odas_client.get_tracked_sources(active_only=True)
    
    # ==================== 波束成形功能 ====================
    
    def record_with_beamforming(
        self, 
        max_duration: float = 30.0,
        silence_duration: float = 0.8,
        output_path: str = None
    ) -> Optional[str]:
        """
        使用波束成形录音
        
        1. 多通道录音
        2. 获取 DOA 角度
        3. 波束成形增强
        4. 输出单通道音频
        
        Args:
            max_duration: 最大录音时长
            silence_duration: 静音判定时长
            output_path: 输出路径
            
        Returns:
            输出文件路径，失败返回 None
        """
        if not self._beamforming_enabled:
            logger.info("Beamforming disabled, using normal capture")
            return self._record_normal(max_duration, silence_duration, output_path)
        
        logger.info("Recording with beamforming...")
        
        # 1. 获取当前 DOA 角度作为初始波束指向
        initial_doa = self.get_current_doa()
        if initial_doa is not None:
            self.beamformer.steer(initial_doa)
            logger.info(f"Initial beam direction: {initial_doa:.1f}°")
        
        # 2. 多通道录音
        multichannel_audio = self.multichannel_capture.record_with_vad(
            max_duration=max_duration,
            silence_duration=silence_duration
        )
        
        if multichannel_audio is None or len(multichannel_audio) == 0:
            logger.warning("No audio captured")
            return None
        
        # 3. 获取录音期间的平均 DOA
        final_doa = self.get_current_doa()
        if final_doa is not None:
            beam_angle = final_doa
            logger.info(f"Final DOA: {final_doa:.1f}°")
        elif initial_doa is not None:
            beam_angle = initial_doa
        else:
            beam_angle = 0.0
            logger.warning("No DOA available, using 0°")
        
        # 4. 提取麦克风通道 (前6通道)
        mic_channels = multichannel_audio[:, :6]
        
        # 5. 波束成形
        logger.info(f"Applying beamforming at {beam_angle:.1f}°")
        enhanced_audio = self.beamformer.process(mic_channels, target_angle=beam_angle)
        
        # 6. 保存为 WAV 文件
        if output_path is None:
            output_path = "temp_beamformed.wav"
        
        # 转换为 int16
        enhanced_int16 = (enhanced_audio * 32767).astype(np.int16)
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.config.get('audio', {}).get('sample_rate', 16000))
            wf.writeframes(enhanced_int16.tobytes())
        
        logger.info(f"Beamformed audio saved: {output_path} ({len(enhanced_audio)/16000:.2f}s)")
        return output_path
    
    def _record_normal(
        self, 
        max_duration: float = 30.0,
        silence_duration: float = 0.8,
        output_path: str = None
    ) -> Optional[str]:
        """普通单通道录音（波束成形禁用时的回退）"""
        if output_path is None:
            output_path = "temp_input.wav"
        
        audio = self.capture.record_with_vad(
            max_duration=max_duration,
            silence_duration=silence_duration,
            output_path=output_path
        )
        
        if len(audio) == 0:
            return None
        
        return output_path
    
    def enable_beamforming(self, enabled: bool = True):
        """启用/禁用波束成形"""
        self._beamforming_enabled = enabled
        logger.info(f"Beamforming {'enabled' if enabled else 'disabled'}")
    
    # ==================== 服务器交互 ====================
    
    def check_server(self) -> bool:
        """检查服务器连接"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Server is healthy")
                return True
            else:
                logger.error(f"Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def synthesize_and_play(self, text: str, use_streaming: bool = None):
        """合成并播放语音"""
        if use_streaming is None:
            use_streaming = self.use_streaming_tts
        
        if use_streaming:
            return self._play_streaming_tts(text)
        else:
            return self._play_normal_tts(text)
    
    def _play_streaming_tts(self, text: str) -> bool:
        """流式 TTS"""
        try:
            logger.info(f"[Streaming TTS] Requesting: {text[:30]}...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/tts/stream",
                json={"text": text},
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                logger.warning(f"Streaming TTS failed ({response.status_code}), falling back to normal TTS")
                return self._play_normal_tts(text)
            
            sample_rate = self.config.get('tts', {}).get('sample_rate', 22050)
            streaming_player = self.player.create_streaming_player(
                sample_rate=sample_rate,
                channels=1
            )
            
            total_bytes = 0
            first_chunk_time = None
            header_skipped = False
            
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency = first_chunk_time - start_time
                        logger.info(f"[Streaming TTS] First audio latency: {latency:.2f}s")
                        print(f"🔊 首音频延迟: {latency:.2f}s")
                    
                    if not header_skipped and len(chunk) >= 44:
                        if chunk[:4] == b'RIFF':
                            chunk = chunk[44:]
                            header_skipped = True
                    
                    if chunk:
                        streaming_player.feed(chunk)
                        total_bytes += len(chunk)
            
            streaming_player.wait_until_done()
            
            total_time = time.time() - start_time
            logger.info(f"[Streaming TTS] Complete: {total_bytes} bytes in {total_time:.2f}s")
            return True
                
        except Exception as e:
            logger.error(f"[Streaming TTS] Error: {e}")
            return self._play_normal_tts(text)
    
    def _play_normal_tts(self, text: str) -> bool:
        """普通 TTS"""
        try:
            response = requests.post(
                f"{self.server_url}/tts",
                json={"text": text},
                timeout=120
            )
            
            if response.status_code == 200:
                temp_audio = "temp_tts_response.wav"
                with open(temp_audio, 'wb') as f:
                    f.write(response.content)
                
                self.player.play_file(temp_audio)
                Path(temp_audio).unlink(missing_ok=True)
                return True
            else:
                logger.error(f"TTS failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    # ==================== 对话功能 ====================
    
    def voice_chat_with_doa(self):
        """
        带 DOA 和波束成形的连续语音对话模式
        
        显示实时声源方向，使用波束成形增强语音
        """
        print("\n" + "=" * 60)
        print("语音对话模式 (DOA + 波束成形增强版)")
        print("=" * 60)
        print("说话后会自动识别并回复")
        print("实时显示声源方向")
        print(f"波束成形: {'✅ 已启用' if self._beamforming_enabled else '❌ 已禁用'}")
        print("按 Ctrl+C 退出")
        print("=" * 60)
        
        # 选择音色
        voice_clone_id = self._select_voice_clone()
        
        try:
            while True:
                # 显示当前 DOA 状态
                doa = self.get_current_doa()
                sources = self.get_tracked_sources()
                
                if sources:
                    doa_info = f"🎯 DOA: {sources[0].azimuth:.1f}° (activity: {sources[0].activity:.2f})"
                else:
                    doa_info = "🎯 DOA: 无活跃声源"
                
                print(f"\n{doa_info}")
                if self._beamforming_enabled:
                    print("🎤 请开始说话 (波束成形录音)...")
                else:
                    print("🎤 请开始说话...")
                
                # 录制音频（根据设置使用波束成形或普通录音）
                temp_audio = "temp_input.wav"
                
                if self._beamforming_enabled:
                    # 使用波束成形录音
                    audio_path = self.record_with_beamforming(
                        max_duration=30.0,
                        silence_duration=0.8,
                        output_path=temp_audio
                    )
                    if audio_path is None:
                        print("未检测到语音，继续监听...")
                        continue
                else:
                    # 普通单通道录音
                    audio = self.capture.record_with_vad(
                        max_duration=30.0,
                        silence_duration=0.8,
                        output_path=temp_audio
                    )
                    if len(audio) == 0:
                        print("未检测到语音，继续监听...")
                        continue
                
                # 获取录音时的 DOA
                recording_doa = self.get_current_doa()
                if recording_doa is not None:
                    print(f"📍 录音时声源方向: {recording_doa:.1f}°")
                    if self._beamforming_enabled:
                        print(f"📡 波束指向: {self.beamformer._current_angle:.1f}°")
                
                print("处理中...")
                
                try:
                    with open(temp_audio, 'rb') as f:
                        files = {'audio': f}
                        data = {
                            'session_id': self.session_id,
                            'voice_clone_id': voice_clone_id or '0'
                        }
                        
                        # 添加 DOA 信息到请求
                        if recording_doa is not None:
                            data['doa_angle'] = str(recording_doa)
                        
                        response = requests.post(
                            f"{self.server_url}/chat",
                            files=files,
                            data=data,
                            stream=True,
                            timeout=180
                        )
                    
                    if response.status_code == 200:
                        from urllib.parse import unquote
                        asr_text = unquote(response.headers.get('X-ASR-Text', ''))
                        response_text = unquote(response.headers.get('X-Response-Text', ''))
                        emotion = response.headers.get('X-Emotion', '')
                        speaker = response.headers.get('X-Speaker', '')
                        
                        print(f"\n👤 你: {asr_text}")
                        print(f"😊 情感: {emotion} | 🎯 说话人: {speaker}")
                        print(f"🤖 助手: {response_text}")
                        
                        # 播放回复
                        self._play_response(response)
                        
                    else:
                        print(f"请求失败: {response.status_code}")
                        print(response.text)
                    
                    Path(temp_audio).unlink(missing_ok=True)
                    
                except requests.exceptions.Timeout:
                    print("\n⚠️ 请求超时，请重试")
                except Exception as e:
                    logger.error(f"Chat request failed: {e}")
                    print(f"请求失败: {e}")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n退出对话模式")
    
    def _select_voice_clone(self) -> Optional[str]:
        """选择音色克隆"""
        try:
            response = requests.get(f"{self.server_url}/voice-clone/list", timeout=30)
            if response.status_code == 200:
                result = response.json()
                voice_clones = result.get('voice_clones', [])
                if voice_clones:
                    print("\n可用的音色克隆：")
                    print("0 - 使用默认音色")
                    for idx, clone_id in enumerate(voice_clones, start=1):
                        print(f"{idx} - {clone_id}")
                    
                    choice = input("\n请选择音色 (回车使用默认): ").strip()
                    if choice.isdigit() and 0 < int(choice) <= len(voice_clones):
                        return voice_clones[int(choice) - 1]
            return None
        except Exception as e:
            logger.warning(f"Failed to list voice clones: {e}")
            return None
    
    def _play_response(self, response):
        """播放响应音频"""
        try:
            is_streaming = response.headers.get('X-Streaming-Audio', 'False') == 'True'
            
            if self.use_streaming_tts and is_streaming:
                sample_rate = self.config.get('tts', {}).get('sample_rate', 22050)
                streaming_player = self.player.create_streaming_player(
                    sample_rate=sample_rate, channels=1
                )
                
                header_skipped = False
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    
                    if not header_skipped and len(chunk) >= 44:
                        if chunk[:4] == b'RIFF':
                            chunk = chunk[44:]
                            header_skipped = True
                    
                    if chunk:
                        streaming_player.feed(chunk)
                
                streaming_player.wait_until_done()
            else:
                response_audio = "temp_response.wav"
                with open(response_audio, 'wb') as f:
                    f.write(response.content)
                self.player.play_file(response_audio)
                Path(response_audio).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to play response: {e}")
    
    def doa_monitor(self):
        """
        DOA 实时监控模式
        
        持续显示声源方向
        """
        print("\n" + "=" * 60)
        print("DOA 实时监控模式")
        print("=" * 60)
        print("显示实时声源定位信息")
        print("按 Ctrl+C 退出")
        print("=" * 60)
        
        try:
            frame_count = 0
            while True:
                sources = self.get_tracked_sources()
                all_sources = self.odas_client.get_tracked_sources(active_only=False)
                
                frame_count += 1
                
                if sources:
                    print(f"\n🎯 检测到 {len(sources)} 个活跃声源:")
                    for s in sources:
                        print(f"   声源 {s.id}: 方位角={s.azimuth:.1f}°, activity={s.activity:.3f}")
                else:
                    activities = [f"{s.activity:.3f}" for s in all_sources]
                    print(f"\r[帧 {frame_count}] 无活跃声源 | activity: {activities}    ", end="", flush=True)
                
                time.sleep(0.3)
                
        except KeyboardInterrupt:
            print("\n\n退出监控模式")
    
    # ==================== 主界面 ====================
    
    def run_interactive(self):
        """运行交互式界面"""
        print("\n" + "=" * 60)
        print("语音助手客户端 (DOA增强版)")
        print("=" * 60)
        
        # 检查服务器
        if not self.check_server():
            print("⚠️ 无法连接到服务器，部分功能可能不可用")
        
        # 启动 DOA 系统
        print("\n正在启动 DOA 系统...")
        if self.start_doa():
            print("✅ DOA 系统已启动")
        else:
            print("⚠️ DOA 系统启动失败，将在无DOA模式下运行")
        
        print("\n可用命令:")
        print("  talk   - 带DOA+波束成形的语音对话（推荐）")
        print("  doa    - DOA实时监控")
        print("  bf     - 开关波束成形")
        print("  status - 查看系统状态")
        print("  gain   - 设置麦克风增益")
        print("  quit   - 退出")
        print()
        
        try:
            while True:
                command = input("\n请输入命令: ").strip().lower()
                
                if command in ['quit', 'q', 'exit']:
                    print("正在关闭...")
                    break
                
                elif command in ['talk', 't']:
                    self.voice_chat_with_doa()
                
                elif command == 'doa':
                    self.doa_monitor()
                
                elif command == 'status':
                    self._show_status()
                
                elif command == 'gain':
                    self._configure_gain()
                
                elif command == 'bf':
                    self._toggle_beamforming()
                
                else:
                    print("未知命令，请重试")
                    
        except KeyboardInterrupt:
            print("\n\n程序被中断")
        finally:
            self.stop_doa()
            print("再见！")
    
    def _show_status(self):
        """显示系统状态"""
        print("\n" + "=" * 40)
        print("系统状态")
        print("=" * 40)
        
        # 服务器状态
        server_ok = self.check_server()
        print(f"服务器: {'✅ 已连接' if server_ok else '❌ 未连接'}")
        
        # ODAS 状态
        odas_running = self.odas_manager.is_running()
        odas_connected = self.odas_client.is_connected()
        print(f"ODAS 进程: {'✅ 运行中' if odas_running else '❌ 未运行'}")
        print(f"ODAS 连接: {'✅ 已连接' if odas_connected else '❌ 未连接'}")
        
        # DOA 状态
        if self._doa_enabled:
            stats = self.odas_client.get_stats()
            print(f"DOA 帧数: {stats.get('frame_count', 0)}")
            print(f"活跃声源: {stats.get('active_sources', 0)}")
            doa = self.get_current_doa()
            if doa is not None:
                print(f"当前 DOA: {doa:.1f}°")
        
        # 波束成形状态
        print(f"波束成形: {'✅ 已启用' if self._beamforming_enabled else '❌ 已禁用'}")
        if self._beamforming_enabled:
            print(f"  波束指向: {self.beamformer._current_angle:.1f}°")
        
        # 增益状态
        gains = self.mic_gain.check_gains()
        if gains:
            print(f"麦克风增益: ADC={gains.get('adc_gain', '?')}")
        
        print("=" * 40)
    
    def _configure_gain(self):
        """配置麦克风增益"""
        print("\n当前增益设置:")
        gains = self.mic_gain.check_gains()
        print(f"  ADC PGA gain: {gains.get('adc_gain', '?')}")
        print(f"  Digital volume: {self.mic_gain.digital_volume}")
        
        try:
            adc = input("输入新的 ADC 增益 (0-31, 回车保持): ").strip()
            if adc:
                self.mic_gain.adc_gain = int(adc)
            
            digital = input("输入新的数字音量 (0-255, 回车保持): ").strip()
            if digital:
                self.mic_gain.digital_volume = int(digital)
            
            self.mic_gain.set_gains()
            print("✅ 增益已更新")
            
        except ValueError:
            print("❌ 无效的输入")
    
    def _toggle_beamforming(self):
        """开关波束成形"""
        self._beamforming_enabled = not self._beamforming_enabled
        status = "✅ 已启用" if self._beamforming_enabled else "❌ 已禁用"
        print(f"\n波束成形: {status}")
        
        if self._beamforming_enabled:
            print("  - 使用 6 通道麦克风阵列")
            print("  - 根据 DOA 角度增强目标方向声音")
            print("  - 抑制其他方向干扰")
        else:
            print("  - 使用单通道录音")
            print("  - 适用于简单场景")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Voice Assistant Client with DOA')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--server', type=str, help='服务器地址')
    parser.add_argument('--no-doa', action='store_true', help='禁用DOA功能')
    
    args = parser.parse_args()
    
    # 初始化客户端
    client = VoiceAssistantWithDOA(config_path=args.config)
    
    # 覆盖服务器地址
    if args.server:
        client.server_url = args.server
    
    # 运行
    client.run_interactive()


if __name__ == "__main__":
    main()

