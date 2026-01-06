"""
全双工控制器模块
实现语音交互的状态机管理和打断机制
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any
from enum import Enum
from dataclasses import dataclass
from collections import deque
import threading
import numpy as np

from event_bus import (
    EventBus, Event, EventType, InterruptController, get_event_bus
)

logger = logging.getLogger(__name__)


class InteractionState(Enum):
    """交互状态枚举"""
    IDLE = "idle"               # 空闲状态，等待唤醒
    LISTENING = "listening"     # 聆听状态，录制用户语音
    PROCESSING = "processing"   # 处理状态，ASR/对话处理中
    SPEAKING = "speaking"       # 播放状态，TTS播放中
    INTERRUPTED = "interrupted" # 被打断状态
    ERROR = "error"             # 错误状态


@dataclass
class AudioChunk:
    """音频数据块"""
    data: np.ndarray
    timestamp: float
    doa_angle: Optional[float] = None  # 声源到达方向角度
    is_speech: bool = False
    energy: float = 0.0
    vad_confidence: float = 0.0


class FullDuplexController:
    """
    全双工交互控制器
    
    核心功能:
    - 状态机管理
    - 打断检测(Barge-in)
    - 异步音频处理
    - 事件驱动架构
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 30,
        interrupt_threshold: float = 0.6,
        min_interrupt_duration: float = 0.15,
        idle_timeout: float = 30.0
    ):
        """
        初始化全双工控制器
        
        Args:
            event_bus: 事件总线实例
            sample_rate: 采样率
            chunk_duration_ms: 音频块时长(毫秒)
            interrupt_threshold: 打断检测阈值
            min_interrupt_duration: 最小打断语音持续时间(秒)
            idle_timeout: 空闲超时时间(秒)
        """
        # 事件总线
        self.event_bus = event_bus or get_event_bus()
        
        # 音频参数
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        # 状态管理
        self._state = InteractionState.IDLE
        self._previous_state = InteractionState.IDLE
        self._state_lock = asyncio.Lock()
        
        # 打断参数
        self.interrupt_threshold = interrupt_threshold
        self.min_interrupt_duration = min_interrupt_duration
        self._interrupt_speech_start: Optional[float] = None
        
        # 超时参数
        self.idle_timeout = idle_timeout
        self._last_activity_time = time.time()
        
        # 音频缓冲区
        self._recording_buffer: deque = deque(maxlen=int(30 * sample_rate / self.chunk_size))
        self._reference_buffer: deque = deque(maxlen=int(2 * sample_rate / self.chunk_size))
        
        # 控制标志
        self._running = False
        self._tts_active = False
        
        # 回调函数
        self._on_speech_complete: Optional[Callable] = None
        self._on_interrupt: Optional[Callable] = None
        self._on_state_change: Optional[Callable] = None
        
        # 打断控制器
        self._interrupt_controller = InterruptController(self.event_bus)
        
        # 订阅事件
        self._setup_event_handlers()
        
        logger.info(f"FullDuplexController initialized: sample_rate={sample_rate}, "
                   f"chunk_size={self.chunk_size}")
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        self.event_bus.subscribe(EventType.WAKE_WORD_DETECTED, self._on_wake_word)
        self.event_bus.subscribe(EventType.SPEECH_END, self._on_speech_end)
        self.event_bus.subscribe(EventType.TTS_START, self._on_tts_start)
        self.event_bus.subscribe(EventType.TTS_END, self._on_tts_end)
        self.event_bus.subscribe(EventType.BARGE_IN, self._on_barge_in)
        self.event_bus.subscribe(EventType.ASR_FINAL, self._on_asr_complete)
        self.event_bus.subscribe(EventType.DIALOGUE_RESPONSE, self._on_dialogue_response)
    
    @property
    def state(self) -> InteractionState:
        """获取当前状态"""
        return self._state
    
    @property
    def is_speaking(self) -> bool:
        """是否正在播放TTS"""
        return self._state == InteractionState.SPEAKING
    
    @property
    def is_listening(self) -> bool:
        """是否正在聆听"""
        return self._state == InteractionState.LISTENING
    
    async def set_state(self, new_state: InteractionState, reason: str = ""):
        """
        设置状态（线程安全）
        
        Args:
            new_state: 新状态
            reason: 状态变更原因
        """
        async with self._state_lock:
            if new_state == self._state:
                return
            
            old_state = self._state
            self._previous_state = old_state
            self._state = new_state
            self._last_activity_time = time.time()
            
            logger.info(f"State change: {old_state.value} -> {new_state.value} ({reason})")
            
            # 发布状态变更事件
            await self.event_bus.emit_async(
                EventType.STATE_CHANGE,
                data={
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "reason": reason
                },
                source="fullduplex_controller"
            )
            
            # 调用状态变更回调
            if self._on_state_change:
                try:
                    if asyncio.iscoroutinefunction(self._on_state_change):
                        await self._on_state_change(old_state, new_state, reason)
                    else:
                        self._on_state_change(old_state, new_state, reason)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
    
    def set_state_sync(self, new_state: InteractionState, reason: str = ""):
        """
        同步设置状态
        
        Args:
            new_state: 新状态
            reason: 状态变更原因
        """
        if new_state == self._state:
            return
        
        old_state = self._state
        self._previous_state = old_state
        self._state = new_state
        self._last_activity_time = time.time()
        
        logger.info(f"State change: {old_state.value} -> {new_state.value} ({reason})")
        
        # 发布状态变更事件
        self.event_bus.emit(
            EventType.STATE_CHANGE,
            data={
                "old_state": old_state.value,
                "new_state": new_state.value,
                "reason": reason
            },
            source="fullduplex_controller"
        )
        
        # 调用状态变更回调
        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state, reason)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    # ==================== 事件处理器 ====================
    
    def _on_wake_word(self, event: Event):
        """唤醒词检测事件处理"""
        if self._state == InteractionState.IDLE:
            self.set_state_sync(InteractionState.LISTENING, "wake word detected")
            self._recording_buffer.clear()
            logger.info("Wake word detected, starting to listen")
    
    def _on_speech_end(self, event: Event):
        """用户语音结束事件处理"""
        if self._state == InteractionState.LISTENING:
            self.set_state_sync(InteractionState.PROCESSING, "user speech ended")
            # 触发录音完成回调
            if self._on_speech_complete:
                audio_data = self.get_recording()
                try:
                    self._on_speech_complete(audio_data)
                except Exception as e:
                    logger.error(f"Speech complete callback error: {e}")
    
    def _on_tts_start(self, event: Event):
        """TTS开始播放事件处理"""
        self._tts_active = True
        if self._state in [InteractionState.PROCESSING, InteractionState.LISTENING]:
            self.set_state_sync(InteractionState.SPEAKING, "TTS started")
    
    def _on_tts_end(self, event: Event):
        """TTS播放结束事件处理"""
        self._tts_active = False
        self._interrupt_speech_start = None
        
        if self._state == InteractionState.SPEAKING:
            # 播放完成后回到聆听状态，等待用户下一句话
            self.set_state_sync(InteractionState.LISTENING, "TTS completed")
    
    def _on_barge_in(self, event: Event):
        """打断事件处理"""
        if self._state == InteractionState.SPEAKING:
            logger.info("Barge-in event received, interrupting TTS")
            self.set_state_sync(InteractionState.INTERRUPTED, "user barge-in")
            
            # 清空录音缓冲区，准备录制新的语音
            self._recording_buffer.clear()
            
            # 触发打断回调
            if self._on_interrupt:
                try:
                    self._on_interrupt(event.data)
                except Exception as e:
                    logger.error(f"Interrupt callback error: {e}")
            
            # 短暂延迟后切换到聆听状态
            self.set_state_sync(InteractionState.LISTENING, "ready, after barge-in")
    
    def _on_asr_complete(self, event: Event):
        """ASR识别完成事件处理"""
        if self._state == InteractionState.PROCESSING:
            text = event.data.get("text", "")
            logger.info(f"ASR complete: {text[:50]}...")
    
    def _on_dialogue_response(self, event: Event):
        """对话响应事件处理"""
        pass  # 由外部处理TTS播放
    
    # ==================== 音频处理 ====================
    
    def process_audio_chunk(
        self, 
        audio_chunk: np.ndarray,
        is_speech: bool = False,
        energy: float = 0.0,
        vad_confidence: float = 0.0,
        doa_angle: Optional[float] = None
    ):
        """
        处理一个音频块
        
        Args:
            audio_chunk: 音频数据
            is_speech: VAD检测结果
            energy: 音频能量
            vad_confidence: VAD置信度
            doa_angle: 声源到达方向角度
        """
        chunk = AudioChunk(
            data=audio_chunk,
            timestamp=time.time(),
            doa_angle=doa_angle,
            is_speech=is_speech,
            energy=energy,
            vad_confidence=vad_confidence
        )
        
        # 聆听状态：录制语音
        if self._state == InteractionState.LISTENING:
            if is_speech:
                self._recording_buffer.append(chunk)
                
                # 发布VAD活动事件
                self.event_bus.emit(
                    EventType.VAD_ACTIVE,
                    data={
                        "energy": energy,
                        "confidence": vad_confidence,
                        "doa_angle": doa_angle
                    },
                    source="fullduplex_controller"
                )
        
        # 播放状态：检测打断
        elif self._state == InteractionState.SPEAKING:
            self._check_barge_in(is_speech, energy, vad_confidence)
    
    def _check_barge_in(
        self, 
        is_speech: bool, 
        energy: float, 
        vad_confidence: float,
        doa_angle: float = None
    ):
        """
        DOA辅助打断检测
        
        增强版打断检测:
        1. VAD检测到语音活动
        2. DOA方向与扬声器方向不同 (区分用户vs回声)
        3. 持续时间超过阈值
        
        Args:
            is_speech: VAD语音活动
            energy: 音频能量
            vad_confidence: VAD置信度
            doa_angle: 声源方向 (度)
        """
        # 扬声器方向 (可配置，假设扬声器在设备前方)
        speaker_direction = getattr(self, '_speaker_direction', 0.0)
        doa_threshold = getattr(self, '_doa_threshold', 30.0)
        
        # 基础VAD检查
        if not is_speech or vad_confidence < self.interrupt_threshold:
            self._interrupt_speech_start = None
            self._interrupt_doa_valid = False
            return
        
        # DOA辅助判断：检查声源是否来自用户方向（非扬声器方向）
        doa_is_user = True  # 默认认为是用户
        
        if doa_angle is not None and speaker_direction is not None:
            # 计算与扬声器方向的角度差
            angle_diff = abs(doa_angle - speaker_direction)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # 如果声源接近扬声器方向，可能是回声而非用户
            if angle_diff < doa_threshold:
                doa_is_user = False
                logger.debug(f"DOA {doa_angle:.1f}° close to speaker {speaker_direction:.1f}°, likely echo")
        
        # 如果DOA判断不是用户，重置打断计时
        if not doa_is_user:
            self._interrupt_speech_start = None
            self._interrupt_doa_valid = False
            return
        
        # 记录语音开始时间
        if self._interrupt_speech_start is None:
            self._interrupt_speech_start = time.time()
            self._interrupt_doa_valid = True
            logger.debug(f"Potential barge-in started: DOA={doa_angle}°")
            return
        
        # 检查持续时间
        speech_duration = time.time() - self._interrupt_speech_start
        
        if speech_duration >= self.min_interrupt_duration:
            # 发布打断事件
            self.event_bus.emit(
                EventType.BARGE_IN,
                data={
                    "confidence": vad_confidence,
                    "duration": speech_duration,
                    "energy": energy,
                    "doa_angle": doa_angle,
                    "doa_assisted": True,
                },
                source="fullduplex_controller",
                priority=2  # 高优先级
            )
            logger.info(f"DOA-assisted barge-in triggered: DOA={doa_angle}°, duration={speech_duration:.2f}s")
            self._interrupt_speech_start = None
            self._interrupt_doa_valid = False
    
    def set_speaker_direction(self, direction: float):
        """
        设置扬声器方向
        
        Args:
            direction: 扬声器方向角度 (0-360)
        """
        self._speaker_direction = direction
        logger.info(f"Speaker direction set to {direction}°")
    
    def set_doa_threshold(self, threshold: float):
        """
        设置DOA区分阈值
        
        Args:
            threshold: 角度阈值 (度), 默认30度
        """
        self._doa_threshold = threshold
    
    def set_reference_audio(self, audio_chunk: np.ndarray):
        """
        设置TTS参考信号（用于AEC）
        
        Args:
            audio_chunk: TTS播放的音频数据
        """
        self._reference_buffer.append(audio_chunk)
    
    def get_recording(self) -> np.ndarray:
        """
        获取录制的音频数据
        
        Returns:
            合并后的音频数组
        """
        if not self._recording_buffer:
            return np.array([], dtype=np.int16)
        
        chunks = [chunk.data for chunk in self._recording_buffer]
        return np.concatenate(chunks)
    
    def clear_recording(self):
        """清空录音缓冲区"""
        self._recording_buffer.clear()
    
    # ==================== 回调设置 ====================
    
    def on_speech_complete(self, callback: Callable):
        """
        设置语音录制完成回调
        
        Args:
            callback: 回调函数，接收音频数据 (np.ndarray)
        """
        self._on_speech_complete = callback
    
    def on_interrupt(self, callback: Callable):
        """
        设置打断回调
        
        Args:
            callback: 回调函数，接收打断事件数据
        """
        self._on_interrupt = callback
    
    def on_state_change(self, callback: Callable):
        """
        设置状态变更回调
        
        Args:
            callback: 回调函数 (old_state, new_state, reason)
        """
        self._on_state_change = callback
    
    # ==================== 控制方法 ====================
    
    def start_listening(self):
        """手动开始聆听"""
        self.set_state_sync(InteractionState.LISTENING, "manual start")
        self._recording_buffer.clear()
    
    def stop_listening(self):
        """手动停止聆听"""
        if self._state == InteractionState.LISTENING:
            self.set_state_sync(InteractionState.PROCESSING, "manual stop")
            # 触发语音完成
            self.event_bus.emit(EventType.SPEECH_END, source="fullduplex_controller")
    
    def interrupt_tts(self):
        """手动打断TTS"""
        if self._state == InteractionState.SPEAKING:
            self.event_bus.emit(
                EventType.BARGE_IN,
                data={"confidence": 1.0, "duration": 0.0, "energy": 0.0},
                source="manual_interrupt",
                priority=2
            )
    
    def reset(self):
        """重置控制器状态"""
        self.set_state_sync(InteractionState.IDLE, "reset")
        self._recording_buffer.clear()
        self._reference_buffer.clear()
        self._interrupt_speech_start = None
        self._tts_active = False
        logger.info("FullDuplexController reset")
    
    async def start(self):
        """启动控制器"""
        self._running = True
        await self.event_bus.start()
        logger.info("FullDuplexController started")
    
    async def stop(self):
        """停止控制器"""
        self._running = False
        await self.event_bus.stop()
        logger.info("FullDuplexController stopped")


class TTSPlaybackController:
    """
    TTS播放控制器
    
    负责管理TTS播放和打断
    """
    
    def __init__(
        self, 
        event_bus: Optional[EventBus] = None,
        audio_player = None
    ):
        """
        初始化TTS播放控制器
        
        Args:
            event_bus: 事件总线
            audio_player: 音频播放器实例
        """
        self.event_bus = event_bus or get_event_bus()
        self.audio_player = audio_player
        
        self._playing = False
        self._interrupted = False
        self._play_lock = threading.Lock()
        
        # 订阅打断事件
        self.event_bus.subscribe(EventType.BARGE_IN, self._on_barge_in)
    
    def _on_barge_in(self, event: Event):
        """处理打断事件"""
        if self._playing:
            self._interrupted = True
            logger.info("TTS playback interrupted by barge-in")
    
    def play_with_interrupt_support(
        self, 
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        chunk_size: int = 4096,
        on_reference_chunk: Optional[Callable] = None
    ) -> bool:
        """
        播放音频，支持打断
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            chunk_size: 每次播放的数据块大小
            on_reference_chunk: 每个数据块的回调（用于AEC参考信号）
            
        Returns:
            是否播放完成（False表示被打断）
        """
        import pyaudio
        
        with self._play_lock:
            self._playing = True
            self._interrupted = False
            
            # 发布TTS开始事件
            self.event_bus.emit(EventType.TTS_START, source="tts_controller")
            
            try:
                audio = pyaudio.PyAudio()
                
                # 确保数据格式正确
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=chunk_size
                )
                
                # 分块播放
                total_chunks = len(audio_data) // chunk_size
                
                for i in range(0, len(audio_data), chunk_size):
                    # 检查是否被打断
                    if self._interrupted:
                        logger.info(f"TTS interrupted at chunk {i // chunk_size}/{total_chunks}")
                        break
                    
                    chunk = audio_data[i:i + chunk_size]
                    
                    # 回调AEC参考信号
                    if on_reference_chunk:
                        on_reference_chunk(chunk)
                    
                    stream.write(chunk.tobytes())
                
                stream.stop_stream()
                stream.close()
                audio.terminate()
                
            except Exception as e:
                logger.error(f"TTS playback error: {e}")
            finally:
                self._playing = False
                
                # 发布TTS结束事件
                if self._interrupted:
                    self.event_bus.emit(
                        EventType.TTS_INTERRUPTED, 
                        source="tts_controller"
                    )
                else:
                    self.event_bus.emit(EventType.TTS_END, source="tts_controller")
            
            return not self._interrupted
    
    def stop(self):
        """停止播放"""
        self._interrupted = True


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    logging.basicConfig(level=logging.DEBUG)
    
    async def test():
        # 创建控制器
        controller = FullDuplexController()
        
        # 设置回调
        def on_state(old, new, reason):
            print(f"State: {old.value} -> {new.value} ({reason})")
        
        controller.on_state_change(on_state)
        
        # 启动
        await controller.start()
        
        # 模拟唤醒
        controller.event_bus.emit(EventType.WAKE_WORD_DETECTED)
        await asyncio.sleep(0.1)
        
        # 检查状态
        print(f"Current state: {controller.state.value}")
        assert controller.state == InteractionState.LISTENING
        
        # 模拟语音结束
        controller.event_bus.emit(EventType.SPEECH_END)
        await asyncio.sleep(0.1)
        
        print(f"Current state: {controller.state.value}")
        assert controller.state == InteractionState.PROCESSING
        
        # 停止
        await controller.stop()
        print("Test passed!")
    
    asyncio.run(test())
