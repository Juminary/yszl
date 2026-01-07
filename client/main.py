"""
树莓派客户端主程序
实现完整的语音交互流程
"""

import requests
import logging
import yaml
import argparse
from pathlib import Path
import time
import sys

from audio_capture import AudioCapture
from audio_player import AudioPlayer
from wakeword_detector import WakeWordDetector

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG以便查看详细日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VoiceAssistantClient:
    """语音助手客户端"""
    
    def __init__(self, config_path: str = None):
        """
        初始化客户端
        
        Args:
            config_path: 配置文件路径
        """
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
        
        # 服务器地址（默认端口改为5001）
        self.server_url = self.config.get('client', {}).get('server_url', 'http://localhost:5001')
        
        # 初始化音频模块
        self.capture = AudioCapture(
            sample_rate=self.config.get('audio', {}).get('sample_rate', 16000),
            channels=self.config.get('audio', {}).get('channels', 1)
        )
        self.player = AudioPlayer()
        
        # 会话ID
        self.session_id = f"raspberrypi_{int(time.time())}"
        
        # 流式 TTS 设置
        self.use_streaming_tts = self.config.get('tts', {}).get('streaming', True)  # 默认启用流式
        
        # 唤醒词设置
        self.wakeword_enabled = self.config.get('wakeword', {}).get('enabled', False)
        self.wakeword = self.config.get('wakeword', {}).get('keyword', '康康')
        
        # 初始化唤醒词检测器（如果启用）
        if self.wakeword_enabled:
            self.wakeword_detector = WakeWordDetector(
                server_url=self.server_url,
                wakeword=self.wakeword
            )
            logger.info(f"Wake word detector enabled: '{self.wakeword}'")
        else:
            self.wakeword_detector = None
        
        logger.info(f"Voice Assistant Client initialized")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Streaming TTS: {'enabled' if self.use_streaming_tts else 'disabled'}")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {}
    
    def synthesize_and_play(self, text: str, use_streaming: bool = None):
        """
        合成并播放语音（自动选择流式或普通模式）
        
        Args:
            text: 要合成的文本
            use_streaming: 是否使用流式，None 则使用配置
        """
        if use_streaming is None:
            use_streaming = self.use_streaming_tts
        
        if use_streaming:
            return self._play_streaming_tts(text)
        else:
            return self._play_normal_tts(text)
    
    def _play_streaming_tts(self, text: str) -> bool:
        """
        流式 TTS：边下载边播放
        使用 StreamingAudioPlayer 实现真正的实时播放
        
        Args:
            text: 要合成的文本
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"[Streaming TTS] Requesting: {text[:30]}...")
            start_time = time.time()
            
            # 流式请求
            response = requests.post(
                f"{self.server_url}/tts/stream",
                json={"text": text},
                stream=True,  # 流式接收
                timeout=120
            )
            
            if response.status_code != 200:
                logger.warning(f"Streaming TTS failed ({response.status_code}), falling back to normal TTS")
                return self._play_normal_tts(text)
            
            # 创建流式播放器（使用 CosyVoice 的采样率）
            sample_rate = self.config.get('tts', {}).get('sample_rate', 22050)
            streaming_player = self.player.create_streaming_player(
                sample_rate=sample_rate,
                channels=1
            )
            
            total_bytes = 0
            first_chunk_time = None
            header_skipped = False
            
            # 边下载边播放
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency = first_chunk_time - start_time
                        logger.info(f"[Streaming TTS] First audio latency: {latency:.2f}s")
                        print(f"🔊 首音频延迟: {latency:.2f}s")
                    
                    # 跳过 WAV 头部（44 字节）
                    if not header_skipped and len(chunk) >= 44:
                        # 检查是否是 WAV 头部
                        if chunk[:4] == b'RIFF':
                            chunk = chunk[44:]  # 跳过头部
                            header_skipped = True
                    
                    if chunk:  # 确保还有数据
                        streaming_player.feed(chunk)
                        total_bytes += len(chunk)
            
            # 等待播放完成
            streaming_player.wait_until_done()
            
            total_time = time.time() - start_time
            logger.info(f"[Streaming TTS] Complete: {total_bytes} bytes in {total_time:.2f}s")
            return True
                
        except Exception as e:
            logger.error(f"[Streaming TTS] Error: {e}")
            import traceback
            traceback.print_exc()
            # 回退到普通模式
            return self._play_normal_tts(text)
    
    def _play_normal_tts(self, text: str) -> bool:
        """
        普通 TTS：等待完整音频后播放
        
        Args:
            text: 要合成的文本
            
        Returns:
            是否成功
        """
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
    
    def list_speakers(self):
        """
        列出所有已注册的说话人（需要服务器端支持 /speaker/list 接口）
        
        Returns:
            说话人字典，格式：{speaker_id: sample_count}
        """
        try:
            response = requests.get(f"{self.server_url}/speaker/list", timeout=5)
            if response.status_code == 200:
                speakers = response.json()
                print(f"\n已注册的说话人:")
                for speaker_id, count in speakers.items():
                    print(f"  - {speaker_id}: {count} 个样本")
                return speakers
            else:
                print(f"查询失败: {response.status_code}")
                if response.status_code == 404:
                    print("提示: 服务器端可能没有实现 /speaker/list 接口")
                return None
        except Exception as e:
            logger.error(f"Failed to list speakers: {e}")
            print(f"查询失败: {e}")
            return None
    
    def register_speaker(self, speaker_id: str):
        """
        注册说话人声纹（同时注册音色克隆）
        
        Args:
            speaker_id: 说话人ID
        """
        try:
            print(f"\n开始注册声纹和音色克隆，说话人ID: {speaker_id}")
            print("=" * 50)
            
            # 1. 让用户输入要朗读的文本
            print("\n【步骤1】请输入要朗读的文本（用于音色克隆）")
            print("提示：")
            print("  - 建议长度：15-50字（约3-10秒朗读）")
            print("  - 文本过长可能导致音色克隆失败")
            print("  - 示例：")
            print("    • '你好，我是医生，很高兴为您服务。'")
            print("    • '您好，我是张医生，有什么可以帮助您的吗？'")
            print("    • '欢迎使用医疗语音助手，我是您的专属医生。'")
            
            while True:
                prompt_text = input("\n请输入要朗读的文本（直接回车使用默认文本）: ").strip()
                if not prompt_text:
                    prompt_text = "你好，我是医生，很高兴为您服务。"
                    print(f"使用默认文本: {prompt_text}")
                    break
                elif len(prompt_text) > 50:
                    print(f"⚠️  文本过长（{len(prompt_text)}字），建议不超过50字")
                    choice = input("是否继续使用此文本？(y/n，默认n): ").strip().lower()
                    if choice == 'y':
                        print("⚠️  警告：文本过长可能导致音色克隆失败")
                        break
                    else:
                        print("请重新输入较短的文本")
                elif len(prompt_text) < 10:
                    print(f"⚠️  文本过短（{len(prompt_text)}字），建议至少15字")
                    choice = input("是否继续使用此文本？(y/n，默认y): ").strip().lower()
                    if choice != 'n':
                        break
                    else:
                        print("请重新输入较长的文本")
                else:
                    break
            
            # 2. 显示文本，让用户准备
            print(f"\n【步骤2】请准备朗读以下文本：")
            print(f"  「{prompt_text}」")
            print("\n提示：")
            print("  - 请用自然、清晰的语气朗读")
            print("  - 建议录音时长3-10秒")
            print("  - 录音过程中请保持安静")
            
            input("\n准备好后，按Enter开始录音...")
            
            # 3. 录制音频
            print("\n【步骤3】正在录音...（请开始朗读）")
            audio_path = f"temp_register_{speaker_id}.wav"
            audio = self.capture.record_with_vad(
                max_duration=30.0,
                silence_duration=2.0,
                output_path=audio_path
            )
            
            if len(audio) == 0:
                print("❌ 未检测到语音，请重试")
                return
            
            audio_duration = len(audio) / self.capture.sample_rate
            print(f"✅ 录音完成，时长: {audio_duration:.2f}秒")
            
            # 检查录音时长
            if audio_duration > 15:
                print(f"⚠️  警告：录音时长过长（{audio_duration:.2f}秒），建议3-10秒")
                print("   这可能导致音色克隆失败或性能问题")
            elif audio_duration < 2:
                print(f"⚠️  警告：录音时长过短（{audio_duration:.2f}秒），建议3-10秒")
                print("   这可能影响音色克隆质量")
            
            # 4. 发送到服务器
            print("\n【步骤4】正在上传并注册...")
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                data = {
                    'speaker_id': speaker_id,
                    'prompt_text': prompt_text  # 传递提示文本
                }
                response = requests.post(
                    f"{self.server_url}/speaker/register",
                    files=files,
                    data=data,
                    timeout=60  # 增加超时时间，因为需要处理音色克隆
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ 声纹注册成功: {speaker_id}")
                print(f"   样本数: {result.get('num_samples', 0)}")
                
                if result.get('voice_clone_registered'):
                    print(f"✅ 音色克隆注册成功: {speaker_id}")
                    print("   现在可以在对话中选择使用此音色了")
                else:
                    print(f"⚠️  音色克隆注册失败")
                    if result.get('voice_clone_error'):
                        print(f"   错误: {result.get('voice_clone_error')}")
            else:
                print(f"❌ 注册失败: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"   错误信息: {error_info.get('error', response.text)}")
                except:
                    print(f"   错误信息: {response.text}")
            
            # 删除临时文件
            Path(audio_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Speaker registration failed: {e}")
            print(f"❌ 注册失败: {e}")
            import traceback
            traceback.print_exc()
    
    def chat_once(self, use_vad: bool = True):
        """
        执行一次完整的对话流程
        
        Args:
            use_vad: 是否使用VAD自动检测
        """
        try:
            print("\n请开始说话...")
            
            # 录制音频
            temp_audio = "temp_input.wav"
            
            if use_vad:
                audio = self.capture.record_with_vad(
                    max_duration=30.0,
                    silence_duration=0.8,  # 减少静默等待时间，加快响应
                    output_path=temp_audio
                )
            else:
                duration = float(input("录音时长（秒）: "))
                audio = self.capture.record(
                    duration=duration,
                    output_path=temp_audio
                )
            
            if len(audio) == 0:
                print("未检测到语音")
                return
            
            print("处理中...")
            
            # 发送到服务器
            with open(temp_audio, 'rb') as f:
                files = {'audio': f}
                data = {'session_id': self.session_id}
                response = requests.post(
                    f"{self.server_url}/chat",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                # 从响应头获取信息（URL解码中文）
                from urllib.parse import unquote
                asr_text = unquote(response.headers.get('X-ASR-Text', ''))
                response_text = unquote(response.headers.get('X-Response-Text', ''))
                emotion = response.headers.get('X-Emotion', '')
                speaker = response.headers.get('X-Speaker', '')
                
                print(f"\n识别文本: {asr_text}")
                print(f"情感: {emotion}")
                print(f"说话人: {speaker}")
                print(f"回复: {response_text}")
                
                # 保存并播放回复音频
                response_audio = "temp_response.wav"
                with open(response_audio, 'wb') as f:
                    f.write(response.content)
                
                print("播放回复...")
                self.player.play_file(response_audio)
                
                # 删除临时文件
                Path(response_audio).unlink(missing_ok=True)
                
            else:
                print(f"请求失败: {response.status_code}")
                print(response.text)
            
            # 删除临时文件
            Path(temp_audio).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            print(f"对话失败: {e}")
    
    def voice_chat_loop(self):
        """
        连续语音对话模式
        自动循环录音、识别、对话、播放回复
        按 Ctrl+C 退出
        """
        print("\n" + "="*50)
        print("连续语音对话模式")
        if self.wakeword_enabled:
            print(f"唤醒词: '{self.wakeword}' (需要先说唤醒词才能开始对话)")
        else:
            print("说话后会自动识别并回复，按 Ctrl+C 退出")
        print("="*50)
        
        # 选择音色克隆
        voice_clone_id = None
        try:
            response = requests.get(f"{self.server_url}/voice-clone/list", timeout=30)  # 增加超时时间
            if response.status_code == 200:
                result = response.json()
                voice_clones = result.get('voice_clones', [])
                if voice_clones:
                    print("\n可用的音色克隆：")
                    print("0 - 使用默认音色")
                    for idx, clone_id in enumerate(voice_clones, start=1):
                        print(f"{idx} - {clone_id}")
                    
                    while True:
                        try:
                            choice = input("\n请选择要使用的音色（输入数字，0为默认音色）: ").strip()
                            if choice == "0":
                                voice_clone_id = None
                                print("已选择默认音色")
                                break
                            elif choice.isdigit():
                                idx = int(choice) - 1
                                if 0 <= idx < len(voice_clones):
                                    voice_clone_id = voice_clones[idx]
                                    print(f"已选择音色: {voice_clone_id}")
                                    break
                                else:
                                    print("无效的选择，请重新输入")
                            else:
                                print("无效的输入，请输入数字")
                        except KeyboardInterrupt:
                            print("\n已取消，使用默认音色")
                            voice_clone_id = None
                            break
                else:
                    print("\n没有可用的音色克隆，将使用默认音色")
            else:
                print("\n无法获取音色克隆列表，将使用默认音色")
        except Exception as e:
            logger.warning(f"Failed to list voice clones: {e}")
            print("\n无法获取音色克隆列表，将使用默认音色")
        
        # 如果启用了唤醒词，先检测唤醒词
        if self.wakeword_enabled and self.wakeword_detector:
            detected, wakeword_text, detected_audio_path = self.wakeword_detector.listen_for_wakeword(
                capture=self.capture,
                check_interval=2.0,  # 每2秒检测一次
                max_listen_time=300.0  # 最多监听5分钟
            )
            
            if not detected:
                print("\n❌ 未检测到唤醒词，退出对话模式")
                return
            
            print(f"\n✅ 唤醒成功！识别文本: '{wakeword_text}'")
            
            # 检测到唤醒词后，先回复"我在呢，有什么可以帮您？"
            print("🔊 播放唤醒回复...")
            try:
                wakeword_response = requests.post(
                    f"{self.server_url}/tts",
                    json={"text": "我在呢，有什么可以帮您？"},
                    timeout=10
                )
                
                if wakeword_response.status_code == 200:
                    # 保存并播放回复音频
                    wakeword_audio = "temp_wakeword_response.wav"
                    with open(wakeword_audio, 'wb') as f:
                        f.write(wakeword_response.content)
                    
                    self.player.play_file(wakeword_audio)
                    
                    # 删除临时文件
                    Path(wakeword_audio).unlink(missing_ok=True)
                    print("✅ 唤醒回复已播放")
                else:
                    print(f"⚠️ 唤醒回复播放失败: {wakeword_response.status_code}")
            except Exception as e:
                logger.error(f"Failed to play wakeword response: {e}")
                print(f"⚠️ 唤醒回复播放失败: {e}")
            
            print("开始对话...")
            time.sleep(0.3)  # 短暂延迟，让用户准备说话
        
        try:
            while True:
                print("\n🎤 请开始说话...")
                
                # 录制音频
                temp_audio = "temp_input.wav"
                
                audio = self.capture.record_with_vad(
                    max_duration=30.0,
                    silence_duration=0.8,  # 减少静默等待时间，加快响应
                    output_path=temp_audio
                )
                
                if len(audio) == 0:
                    print("未检测到语音，继续监听...")
                    continue
                
                print("处理中...")
                
                # 发送到服务器（增加超时时间，TTS合成可能需要较长时间）
                try:
                    # 显示进度提示
                    import threading
                    progress_stop = threading.Event()
                    
                    def show_progress():
                        dots = 0
                        while not progress_stop.is_set():
                            print(f"\r处理中{'...'[:dots%3+1]}", end='', flush=True)
                            dots += 1
                            time.sleep(0.5)
                    
                    progress_thread = threading.Thread(target=show_progress, daemon=True)
                    progress_thread.start()
                    
                    try:
                        with open(temp_audio, 'rb') as f:
                            files = {'audio': f}
                            data = {'session_id': self.session_id}
                            # 如果选择了音色克隆，添加到请求中
                            if voice_clone_id:
                                data['voice_clone_id'] = voice_clone_id
                            else:
                                data['voice_clone_id'] = '0'  # 明确指定使用默认音色
                            response = requests.post(
                                f"{self.server_url}/chat",
                                files=files,
                                data=data,
                                stream=True,  # 流式接收响应
                                timeout=180  # 增加到180秒，TTS合成特别是音色克隆可能需要更长时间
                            )
                    finally:
                        progress_stop.set()
                        print()  # 换行
                    
                    if response.status_code == 200:
                        # 从响应头获取信息（URL解码中文）
                        from urllib.parse import unquote
                        asr_text = unquote(response.headers.get('X-ASR-Text', ''))
                        response_text = unquote(response.headers.get('X-Response-Text', ''))
                        emotion = response.headers.get('X-Emotion', '')
                        speaker = response.headers.get('X-Speaker', '')
                        
                        print(f"\n👤 你: {asr_text}")
                        print(f"😊 情感: {emotion} | 🎯 说话人: {speaker}")
                        print(f"🤖 助手: {response_text}")
                        
                        # 流式播放回复音频（边下载边播放）
                        try:
                            sample_rate = self.config.get('tts', {}).get('sample_rate', 22050)
                            streaming_player = self.player.create_streaming_player(
                                sample_rate=sample_rate,
                                channels=1
                            )
                            
                            total_bytes = 0
                            first_chunk_time = None
                            header_skipped = False
                            start_time = time.time()
                            
                            # 边下载边播放
                            for chunk in response.iter_content(chunk_size=4096):
                                if chunk:
                                    if first_chunk_time is None:
                                        first_chunk_time = time.time()
                                        latency = first_chunk_time - start_time
                                        print(f"🔊 首音频延迟: {latency:.2f}s")
                                    
                                    # 跳过 WAV 头部（44 字节）
                                    if not header_skipped and len(chunk) >= 44:
                                        if chunk[:4] == b'RIFF':
                                            chunk = chunk[44:]
                                            header_skipped = True
                                    
                                    if chunk:
                                        streaming_player.feed(chunk)
                                        total_bytes += len(chunk)
                            
                            # 等待播放完成
                            streaming_player.wait_until_done()
                            
                        except Exception as e:
                            logger.warning(f"Streaming playback failed, falling back to file playback: {e}")
                            # 回退到文件播放
                            response_audio = "temp_response.wav"
                            with open(response_audio, 'wb') as f:
                                f.write(response.content)
                            self.player.play_file(response_audio)
                            Path(response_audio).unlink(missing_ok=True)
                        
                    else:
                        print(f"请求失败: {response.status_code}")
                        print(response.text)
                    
                    # 删除临时文件
                    Path(temp_audio).unlink(missing_ok=True)
                    
                except requests.exceptions.Timeout:
                    logger.error("Chat request timeout (TTS synthesis may take too long)")
                    print("\n⚠️ 请求超时：语音合成可能需要更长时间，请重试")
                    print("   提示：如果使用音色克隆，合成时间会更长（可能需要30-60秒）")
                    print("   建议：可以尝试使用默认音色，或等待服务器处理完成")
                except Exception as e:
                    logger.error(f"Chat request failed: {e}")
                    print(f"请求失败: {e}")
                
                # 短暂延迟，避免立即开始下一轮
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n退出连续语音对话模式")
    
    def text_chat(self, text: str = None):
        """
        文字对话模式（不需要麦克风）
        
        Args:
            text: 输入的文字，如果为空则提示输入
        """
        try:
            if not text:
                text = input("请输入文字: ").strip()
            
            if not text:
                print("输入不能为空")
                return
            
            print(f"\n你: {text}")
            print("处理中...")
            
            # 1. 发送到对话API
            response = requests.post(
                f"{self.server_url}/dialogue",
                json={"query": text, "session_id": self.session_id},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"对话请求失败: {response.text}")
                return
            
            dialogue_result = response.json()
            response_text = dialogue_result.get('response', '')
            print(f"助手: {response_text}")
            
            # 2. 语音合成并播放
            tts_response = requests.post(
                f"{self.server_url}/tts",
                json={"text": response_text},
                timeout=60
            )
            
            if tts_response.status_code == 200:
                # 保存并播放音频
                response_audio = "temp_tts_response.wav"
                with open(response_audio, 'wb') as f:
                    f.write(tts_response.content)
                
                print("播放回复...")
                self.player.play_file(response_audio)
                
                # 删除临时文件
                Path(response_audio).unlink(missing_ok=True)
            else:
                print(f"语音合成失败: {tts_response.status_code}")
            
        except Exception as e:
            logger.error(f"Text chat failed: {e}")
            print(f"文字对话失败: {e}")
    
    def text_chat_loop(self):
        """
        连续文字对话模式
        """
        print("\n进入连续文字对话模式（输入 'exit' 或 'quit' 退出）")
        print("-" * 40)
        
        while True:
            try:
                text = input("\n你: ").strip()
                
                if not text:
                    continue
                
                if text.lower() in ['exit', 'quit', 'q', '退出']:
                    print("退出文字对话模式")
                    break
                
                print("处理中...")
                
                # 发送到对话API
                response = requests.post(
                    f"{self.server_url}/dialogue",
                    json={"query": text, "session_id": self.session_id},
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"对话请求失败: {response.text}")
                    continue
                
                dialogue_result = response.json()
                response_text = dialogue_result.get('response', '')
                print(f"助手: {response_text}")
                
                # 语音合成并播放（自动使用流式或普通模式）
                self.synthesize_and_play(response_text)
                
            except KeyboardInterrupt:
                print("\n退出文字对话模式")
                break
            except Exception as e:
                logger.error(f"Text chat loop error: {e}")
                print(f"对话失败: {e}")
    
    def tts_then_asr_chat(self):
        """
        文字转语音后发送给服务器（测试ASR）
        流程：文字 -> TTS转语音 -> 发送音频到/chat -> ASR识别 -> 对话 -> TTS回复
        """
        import subprocess
        import os
        
        print("\n进入 TTS+ASR 测试模式（输入 'exit' 退出）")
        print("流程: 你的文字 -> TTS转语音 -> 发送服务器 -> ASR识别 -> 对话 -> TTS回复")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n输入文字: ").strip()
                
                if not text:
                    continue
                
                if text.lower() in ['exit', 'quit', 'q', '退出']:
                    print("退出 TTS+ASR 测试模式")
                    break
                
                # 1. 本地TTS: 文字转语音
                print("① 本地TTS: 文字转语音...")
                temp_audio = "temp_tts_input.wav"
                temp_aiff = "temp_tts_input.aiff"
                
                # 使用macOS say命令生成音频
                result = subprocess.run(
                    ['say', '-v', 'Tingting', '-o', temp_aiff, text],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode != 0 or not os.path.exists(temp_aiff):
                    print(f"TTS失败: {result.stderr}")
                    continue
                
                # 使用 soundfile 转换为 16kHz WAV（ASR需要）
                try:
                    import soundfile as sf
                    import numpy as np
                    from scipy import signal
                    
                    # 读取 AIFF
                    audio_data, sample_rate = sf.read(temp_aiff)
                    
                    # 转为单声道
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # 重采样到 16kHz
                    if sample_rate != 16000:
                        num_samples = int(len(audio_data) * 16000 / sample_rate)
                        audio_data = signal.resample(audio_data, num_samples)
                    
                    # 归一化音频（放大到合适的振幅）
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val * 0.9  # 归一化到90%
                    
                    # 转换为 int16 格式（ASR需要）
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    
                    # 保存为 16kHz 16-bit WAV
                    sf.write(temp_audio, audio_int16, 16000, subtype='PCM_16')
                    print(f"   音频已转换: {sample_rate}Hz -> 16000Hz, 时长: {len(audio_data)/16000:.2f}s")
                except Exception as e:
                    print(f"   音频转换失败: {e}, 尝试ffmpeg...")
                    # 回退到 ffmpeg
                    subprocess.run(
                        ['ffmpeg', '-y', '-i', temp_aiff, '-ar', '16000', '-ac', '1', temp_audio],
                        capture_output=True, timeout=30
                    )
                
                if not os.path.exists(temp_audio):
                    print("   无法转换音频，跳过")
                    continue
                
                print(f"② 发送音频到服务器 /chat ...")
                
                # 2. 发送音频到服务器的 /chat 接口
                with open(temp_audio, 'rb') as f:
                    files = {'audio': f}
                    data = {'session_id': self.session_id}
                    response = requests.post(
                        f"{self.server_url}/chat",
                        files=files,
                        data=data,
                        stream=True,  # 流式接收响应
                        timeout=120
                    )
                
                if response.status_code == 200:
                    # 从响应头获取信息（URL解码中文）
                    from urllib.parse import unquote
                    asr_text = unquote(response.headers.get('X-ASR-Text', ''))
                    response_text = unquote(response.headers.get('X-Response-Text', ''))
                    emotion = response.headers.get('X-Emotion', '')
                    speaker = response.headers.get('X-Speaker', '')
                    rag_used = response.headers.get('X-RAG-Used', 'False') == 'True'
                    
                    print(f"\n③ ASR识别结果: {asr_text}")
                    print(f"④ 情感: {emotion}")
                    print(f"⑤ 说话人: {speaker}")
                    print(f"⑥ RAG知识检索: {'✓ 已使用' if rag_used else '✗ 未使用'}")
                    print(f"⑦ 助手回复: {response_text}")
                    
                    # 流式播放回复音频（边下载边播放）
                    print("⑧ 播放回复...")
                    try:
                        sample_rate = self.config.get('tts', {}).get('sample_rate', 22050)
                        streaming_player = self.player.create_streaming_player(
                            sample_rate=sample_rate,
                            channels=1
                        )
                        
                        total_bytes = 0
                        first_chunk_time = None
                        header_skipped = False
                        start_time = time.time()
                        
                        # 边下载边播放
                        for chunk in response.iter_content(chunk_size=4096):
                            if chunk:
                                if first_chunk_time is None:
                                    first_chunk_time = time.time()
                                    latency = first_chunk_time - start_time
                                    print(f"🔊 首音频延迟: {latency:.2f}s")
                                
                                # 跳过 WAV 头部（44 字节）
                                if not header_skipped and len(chunk) >= 44:
                                    if chunk[:4] == b'RIFF':
                                        chunk = chunk[44:]
                                        header_skipped = True
                                
                                if chunk:
                                    streaming_player.feed(chunk)
                                    total_bytes += len(chunk)
                        
                        # 等待播放完成
                        streaming_player.wait_until_done()
                        
                    except Exception as e:
                        logger.warning(f"Streaming playback failed, falling back to file playback: {e}")
                        # 回退到文件播放
                        response_audio = "temp_response.wav"
                        with open(response_audio, 'wb') as f:
                            f.write(response.content)
                        self.player.play_file(response_audio)
                        Path(response_audio).unlink(missing_ok=True)
                else:
                    print(f"请求失败: {response.status_code}")
                    try:
                        print(response.json())
                    except:
                        print(response.text)
                
                # 清理临时文件
                Path(temp_aiff).unlink(missing_ok=True)
                if temp_audio != temp_aiff:
                    Path(temp_audio).unlink(missing_ok=True)
                
            except KeyboardInterrupt:
                print("\n退出 TTS+ASR 测试模式")
                break
            except Exception as e:
                logger.error(f"TTS+ASR test error: {e}")
                print(f"测试失败: {e}")
    
    def run_interactive(self):
        """运行交互式对话模式"""
        print("\n" + "="*50)
        print("语音助手客户端")
        print("="*50)
        
        # 检查服务器
        if not self.check_server():
            print("无法连接到服务器，请检查服务器是否运行")
            return
        
        print("\n可用命令:")
        print("  talk     - 连续语音对话（推荐，支持音色选择）")
        print("  chat     - 语音对话（麦克风输入）")
        print("  dia      - 连续文字对话")
        print("  tchat    - TTS+ASR测试（文字转语音后发服务器）")
        print("  register - 注册声纹（同时注册音色克隆）")
        print("  speakers - 查看已注册的说话人（需要服务器支持）")
        print("  quit     - 退出")
        print()
        
        while True:
            try:
                command = input("\n请输入命令: ").strip().lower()
                
                if command == 'quit' or command == 'q':
                    print("再见！")
                    break
                
                elif command == 'talk' or command == 't':
                    self.voice_chat_loop()
                    
                elif command == 'chat' or command == 'c':
                    self.voice_chat_loop()
                
                elif command == 'dia' or command == 'd':
                    self.text_chat_loop()
                
                elif command == 'tchat' or command == 'tc':
                    self.tts_then_asr_chat()
                    
                elif command == 'register' or command == 'r':
                    speaker_id = input("请输入说话人ID: ").strip()
                    if speaker_id:
                        self.register_speaker(speaker_id)
                    else:
                        print("说话人ID不能为空")
                
                elif command == 'speakers' or command == 's':
                    self.list_speakers()
                    
                else:
                    print("未知命令，请重试")
                    
            except KeyboardInterrupt:
                print("\n\n程序被中断")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"发生错误: {e}")
    
    def run_continuous(self):
        """运行连续对话模式（带唤醒词）"""
        print("\n" + "="*50)
        print("语音助手 - 连续模式")
        print(f"唤醒词: {self.wakeword}")
        print("="*50)
        
        # 检查服务器
        if not self.check_server():
            print("无法连接到服务器")
            return
        
        print("\n监听唤醒词中... (按Ctrl+C退出)")
        
        try:
            while True:
                # 简化版：直接监听并识别
                # 实际应用中应该使用专门的唤醒词检测模型
                print("\n说出唤醒词或按Enter开始对话...")
                input()
                
                print("已激活，请说话...")
                self.chat_once(use_vad=True)
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n程序退出")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Voice Assistant Client')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--server', type=str, help='服务器地址')
    parser.add_argument('--mode', type=str, choices=['interactive', 'continuous'],
                       default='interactive', help='运行模式')
    parser.add_argument('--register', type=str, help='注册声纹（指定说话人ID）')
    
    args = parser.parse_args()
    
    # 初始化客户端
    client = VoiceAssistantClient(config_path=args.config)
    
    # 覆盖服务器地址
    if args.server:
        client.server_url = args.server
    
    # 执行操作
    if args.register:
        client.register_speaker(args.register)
    elif args.mode == 'continuous':
        client.run_continuous()
    else:
        client.run_interactive()


if __name__ == "__main__":
    main()
