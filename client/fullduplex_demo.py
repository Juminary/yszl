"""
全双工语音交互示例
演示如何集成各模块实现全双工对话

使用方法:
    python client/fullduplex_demo.py

功能演示:
1. 持续监听麦克风
2. 唤醒词触发对话
3. VAD自动检测语音边界
4. TTS播放支持打断
5. 状态机管理交互流程
"""

import asyncio
import logging
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from event_bus import EventBus, EventType, Event, get_event_bus
from fullduplex_controller import FullDuplexController, InteractionState
from audio_capture import AsyncAudioCapture
from audio_player import InterruptibleAudioPlayer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullDuplexDemo:
    """
    全双工语音交互演示类
    
    整合所有模块，展示完整的全双工交互流程
    """
    
    def __init__(self, server_url: str = "http://localhost:6007"):
        """
        初始化全双工演示
        
        Args:
            server_url: 服务器地址
        """
        self.server_url = server_url
        
        # 事件总线
        self.event_bus = get_event_bus()
        
        # 全双工控制器
        self.controller = FullDuplexController(event_bus=self.event_bus)
        
        # 音频采集器（持续监听模式）
        self.audio_capture = AsyncAudioCapture(
            event_bus=self.event_bus,
            energy_threshold=1500
        )
        
        # 可中断播放器
        self.audio_player = InterruptibleAudioPlayer(
            event_bus=self.event_bus,
            sample_rate=22050
        )
        
        # 设置回调
        self._setup_callbacks()
        
        # 运行状态
        self._running = False
        self._loop = None  # 主事件循环引用
        
        logger.info("FullDuplexDemo initialized")
    
    def _setup_callbacks(self):
        """设置回调函数"""
        
        # 控制器回调
        self.controller.on_state_change(self._on_state_change)
        self.controller.on_interrupt(self._on_interrupt)
        
        # 音频采集回调
        self.audio_capture.on_audio_chunk(self._on_audio_chunk)
        self.audio_capture.on_speech_end(self._on_speech_end)
        
        # 播放器回调
        self.audio_player.on_reference_chunk(self._on_reference_chunk)
        
        # 事件订阅
        self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._handle_wake_word)
        self.event_bus.subscribe(EventType.BARGE_IN, self._handle_barge_in)
    
    def _on_state_change(self, old_state, new_state, reason):
        """状态变更回调"""
        logger.info(f"[State] {old_state.value} -> {new_state.value}: {reason}")
        
        # 可以在这里添加LED指示灯控制
        if new_state == InteractionState.LISTENING:
            print("🎤 正在聆听...")
        elif new_state == InteractionState.SPEAKING:
            print("🔊 正在播放...")
        elif new_state == InteractionState.PROCESSING:
            print("⏳ 正在处理...")
        elif new_state == InteractionState.IDLE:
            print("💤 空闲状态")
    
    def _on_interrupt(self, data):
        """打断回调"""
        logger.info(f"[Interrupt] 用户打断了播放: {data}")
        print("⚡ 检测到打断!")
    
    def _on_audio_chunk(self, chunk, is_speech, confidence, energy):
        """音频块回调 - 传递给控制器处理"""
        self.controller.process_audio_chunk(
            chunk, 
            is_speech=is_speech,
            energy=energy,
            vad_confidence=confidence
        )
    
    def _on_speech_end(self, audio_data, duration):
        """语音结束回调 - 发送到服务器处理"""
        logger.info(f"[Speech] 语音结束: {duration:.2f}秒, {len(audio_data)} samples")
        
        if len(audio_data) > 0:
            # 从后台线程提交异步任务到主事件循环
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._process_speech(audio_data), self._loop)
            else:
                # 回退：同步处理
                import threading
                threading.Thread(target=self._process_speech_sync, args=(audio_data,), daemon=True).start()
    
    def _on_reference_chunk(self, chunk):
        """TTS参考信号回调 - 用于AEC"""
        # 在实际应用中，这个信号会传给AEC模块
        pass
    
    async def _process_speech(self, audio_data: np.ndarray):
        """
        处理用户语音
        
        在实际应用中，这里会：
        1. 调用ASR服务获取文本
        2. 调用对话服务获取回复
        3. 调用TTS服务获取音频
        4. 播放回复音频
        """
        try:
            import requests
            import io
            import wave
            
            # 保存为临时WAV文件
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data.tobytes())
            wav_buffer.seek(0)
            
            # 调用完整对话API
            logger.info("发送到服务器处理...")
            try:
                response = requests.post(
                    f"{self.server_url}/chat",
                    files={"audio": ("audio.wav", wav_buffer, "audio/wav")},
                    data={"session_id": "fullduplex_demo"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    # 检查响应头获取文本（服务器将文本编码在响应头中）
                    from urllib.parse import unquote
                    asr_text = unquote(response.headers.get('X-ASR-Text', ''))
                    response_text = unquote(response.headers.get('X-Response-Text', ''))
                    
                    if asr_text:
                        print(f"👤 识别: {asr_text}")
                    if response_text:
                        print(f"🤖 回复: {response_text}")
                        logger.info(f"服务器回复: {response_text[:50]}...")
                    
                    # 播放音频回复
                    if response.headers.get('Content-Type', '').startswith('audio'):
                        import soundfile as sf
                        audio_bytes = io.BytesIO(response.content)
                        audio_array_resp, sample_rate = sf.read(audio_bytes, dtype='int16')
                        
                        # 播放回复（支持打断）
                        completed = self.audio_player.play(
                            audio_array_resp, 
                            sample_rate, 
                            text=response_text
                        )
                        
                        if not completed:
                            logger.info("回复被用户打断")
                else:
                    logger.error(f"服务器错误: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                logger.warning("无法连接到服务器，使用本地模拟回复")
                # 模拟回复
                await self._simulate_response()
                
        except Exception as e:
            logger.error(f"处理语音时出错: {e}")
    
    def _process_speech_sync(self, audio_data: np.ndarray):
        """同步版本的语音处理 - 使用流式TTS"""
        try:
            import requests
            import io
            import wave
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data.tobytes())
            wav_buffer.seek(0)
            
            # 步骤1：调用ASR识别
            logger.info("发送到服务器进行语音识别...")
            asr_response = requests.post(
                f"{self.server_url}/asr",
                files={"audio": ("audio.wav", wav_buffer, "audio/wav")},
                timeout=15
            )
            
            if asr_response.status_code != 200:
                logger.error(f"ASR错误: {asr_response.status_code}")
                return
            
            asr_result = asr_response.json()
            asr_text = asr_result.get('text', '')
            
            if not asr_text:
                logger.warning("未识别到语音内容")
                return
            
            print(f"👤 识别: {asr_text}")
            
            # 步骤2：调用对话接口
            logger.info("获取对话回复...")
            dialogue_response = requests.post(
                f"{self.server_url}/dialogue",
                json={"query": asr_text, "session_id": "fullduplex_demo"},
                timeout=20
            )
            
            if dialogue_response.status_code != 200:
                logger.error(f"对话错误: {dialogue_response.status_code}")
                return
            
            dialogue_result = dialogue_response.json()
            response_text = dialogue_result.get('response', '')
            
            if not response_text:
                logger.warning("对话模块未返回内容")
                return
            
            print(f"🤖 回复: {response_text}")
            
            # 步骤3：调用流式TTS，边接收边播放
            logger.info("开始流式TTS播放...")
            
            def stream_audio():
                """生成器：从流式TTS接口获取音频块"""
                with requests.post(
                    f"{self.server_url}/tts/stream",
                    json={"text": response_text},
                    stream=True,
                    timeout=60
                ) as tts_response:
                    if tts_response.status_code != 200:
                        logger.error(f"TTS错误: {tts_response.status_code}")
                        return
                    
                    # 跳过WAV头（44字节），逐块读取音频数据
                    first_chunk = True
                    for chunk in tts_response.iter_content(chunk_size=4096):
                        if chunk:
                            if first_chunk:
                                # 第一块可能包含WAV头，跳过
                                first_chunk = False
                                if chunk[:4] == b'RIFF':
                                    # 找到data块的位置
                                    chunk = chunk[44:] if len(chunk) > 44 else b''
                            if chunk:
                                yield chunk
            
            # 使用流式播放
            completed = self.audio_player.play_stream(
                stream_audio(),
                sample_rate=22050,
                text=response_text
            )
            
            if not completed:
                logger.info("回复被用户打断")
                
        except requests.exceptions.ConnectionError:
            print("🤖: 无法连接服务器，请确认服务器已启动")
        except Exception as e:
            logger.error(f"处理语音时出错: {e}", exc_info=True)
    
    async def _simulate_response(self):
        """模拟服务器回复（用于演示）"""
        print("🤖: 你好！我是医声智联语音助手。由于服务器未连接，这是模拟回复。")
        
        # 生成简单的提示音
        duration = 0.5
        freq = 440.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), False)
        audio = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)
        
        self.audio_player.play(audio, sr, text="提示音")
    
    def _handle_wake_word(self, event: Event):
        """处理唤醒词事件"""
        logger.info("检测到唤醒词!")
        print("\n✨ 唤醒词触发!")
    
    def _handle_barge_in(self, event: Event):
        """处理打断事件"""
        # 停止当前播放
        self.audio_player.stop()
    
    def simulate_wake_word(self):
        """模拟唤醒词触发（用于测试）"""
        self.event_bus.emit(EventType.WAKE_WORD_DETECTED, source="manual")
    
    async def run(self):
        """运行演示"""
        self._running = True
        
        print("=" * 50)
        print("    医声智联 - 全双工语音交互演示")
        print("=" * 50)
        print()
        print("功能说明:")
        print("  - 按 Enter 模拟唤醒词触发")
        print("  - 说话后自动检测语音边界")
        print("  - TTS播放时可以随时打断")
        print("  - 按 Ctrl+C 退出")
        print()
        print("-" * 50)
        
        # 保存事件循环引用
        self._loop = asyncio.get_event_loop()
        
        # 启动事件总线
        await self.event_bus.start()
        
        # 启动控制器
        await self.controller.start()
        
        # 启动音频采集
        self.audio_capture.start()
        
        print("系统已启动，正在监听...")
        print()
        
        try:
            # 主循环
            while self._running:
                # 检查键盘输入（非阻塞）
                # 注意：这只是演示，实际应用中应使用唤醒词检测
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
    
    async def stop(self):
        """停止演示"""
        self._running = False
        self.audio_capture.stop()
        await self.controller.stop()
        await self.event_bus.stop()
        print("\n演示结束")


async def main():
    """主函数"""
    demo = FullDuplexDemo()
    
    # 创建输入监听任务
    async def input_listener():
        """监听键盘输入"""
        import sys
        
        while demo._running:
            try:
                # 在Windows上使用kbhit
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\r':  # Enter
                            demo.simulate_wake_word()
                        elif key == b'\x03':  # Ctrl+C
                            break
                else:
                    # Unix系统
                    import select
                    if select.select([sys.stdin], [], [], 0.0)[0]:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        demo.simulate_wake_word()
                
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"Input error: {e}")
                await asyncio.sleep(0.5)
    
    # 并行运行主循环和输入监听
    try:
        await asyncio.gather(
            demo.run(),
            input_listener()
        )
    except KeyboardInterrupt:
        pass
    finally:
        await demo.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户取消")
