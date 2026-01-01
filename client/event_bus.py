"""
事件总线模块
用于全双工交互的异步事件发布/订阅机制
支持本地事件总线和可选的MQTT集成
"""

import asyncio
import logging
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import queue

logger = logging.getLogger(__name__)


class EventType(Enum):
    """系统事件类型枚举"""
    # 唤醒相关
    WAKE_WORD_DETECTED = "wake_word_detected"
    
    # 语音活动
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    VAD_ACTIVE = "vad_active"
    VAD_INACTIVE = "vad_inactive"
    
    # 打断相关
    BARGE_IN = "barge_in"  # 用户打断TTS播放
    INTERRUPT_REQUEST = "interrupt_request"
    
    # ASR相关
    ASR_PARTIAL = "asr_partial"  # 中间结果
    ASR_FINAL = "asr_final"      # 最终结果
    
    # TTS相关
    TTS_START = "tts_start"
    TTS_CHUNK = "tts_chunk"
    TTS_END = "tts_end"
    TTS_INTERRUPTED = "tts_interrupted"
    
    # 对话状态
    DIALOGUE_START = "dialogue_start"
    DIALOGUE_RESPONSE = "dialogue_response"
    DIALOGUE_END = "dialogue_end"
    
    # 副语言事件
    COUGH_DETECTED = "cough_detected"
    WHEEZE_DETECTED = "wheeze_detected"
    PAIN_DETECTED = "pain_detected"
    ANXIETY_DETECTED = "anxiety_detected"
    
    # 系统状态
    STATE_CHANGE = "state_change"
    ERROR = "error"
    
    # 声源定位
    DOA_UPDATE = "doa_update"


@dataclass
class Event:
    """事件数据类"""
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    source: str = "system"
    priority: int = 0  # 0=normal, 1=high, 2=urgent
    
    def __str__(self):
        return f"Event({self.event_type.value}, source={self.source}, priority={self.priority})"


class EventBus:
    """
    事件总线 - 异步事件发布/订阅系统
    
    支持:
    - 异步事件处理
    - 事件优先级
    - 事件过滤
    - 同步和异步回调
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        初始化事件总线
        
        Args:
            max_queue_size: 事件队列最大容量
        """
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._global_subscribers: List[Callable] = []
        self._event_queue: asyncio.Queue = None
        self._running = False
        self._max_queue_size = max_queue_size
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 同步模式支持
        self._sync_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._sync_thread: Optional[threading.Thread] = None
        
        logger.info("EventBus initialized")
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """
        订阅特定类型的事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数，可以是同步或异步函数
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}: {callback.__name__}")
    
    def subscribe_all(self, callback: Callable):
        """
        订阅所有事件
        
        Args:
            callback: 回调函数
        """
        self._global_subscribers.append(callback)
        logger.debug(f"Subscribed to ALL events: {callback.__name__}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}: {callback.__name__}")
            except ValueError:
                pass
    
    def publish(self, event: Event):
        """
        发布事件（同步模式）
        
        Args:
            event: 事件对象
        """
        logger.debug(f"Publishing event: {event}")
        
        # 如果在异步模式下，放入异步队列
        if self._running and self._event_queue is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._event_queue.put_nowait, event
                )
            except asyncio.QueueFull:
                logger.warning(f"Event queue full, dropping event: {event}")
        else:
            # 同步模式直接调用
            self._dispatch_sync(event)
    
    async def publish_async(self, event: Event):
        """
        发布事件（异步模式）
        
        Args:
            event: 事件对象
        """
        if self._event_queue is not None:
            await self._event_queue.put(event)
        else:
            await self._dispatch_async(event)
    
    def emit(self, event_type: EventType, data: Dict[str, Any] = None, 
             source: str = "system", priority: int = 0):
        """
        便捷方法：创建并发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源
            priority: 优先级
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )
        self.publish(event)
    
    async def emit_async(self, event_type: EventType, data: Dict[str, Any] = None,
                         source: str = "system", priority: int = 0):
        """
        便捷方法：创建并发布事件（异步）
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )
        await self.publish_async(event)
    
    def _dispatch_sync(self, event: Event):
        """同步分发事件"""
        callbacks = self._get_callbacks(event.event_type)
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # 异步回调在同步模式下使用新事件循环执行
                    asyncio.run(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event callback {callback.__name__}: {e}")
    
    async def _dispatch_async(self, event: Event):
        """异步分发事件"""
        callbacks = self._get_callbacks(event.event_type)
        
        tasks = []
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(asyncio.create_task(callback(event)))
                else:
                    # 同步回调在线程池中执行
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, callback, event))
            except Exception as e:
                logger.error(f"Error creating task for {callback.__name__}: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _get_callbacks(self, event_type: EventType) -> List[Callable]:
        """获取事件的所有回调"""
        callbacks = list(self._global_subscribers)
        if event_type in self._subscribers:
            callbacks.extend(self._subscribers[event_type])
        return callbacks
    
    async def start(self):
        """启动事件循环（异步模式）"""
        if self._running:
            return
        
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._event_queue = asyncio.Queue(maxsize=self._max_queue_size)
        
        logger.info("EventBus started (async mode)")
        
        # 启动事件处理循环
        asyncio.create_task(self._event_loop())
    
    async def _event_loop(self):
        """事件处理主循环"""
        while self._running:
            try:
                # 带超时获取事件
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                await self._dispatch_async(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event loop: {e}")
    
    async def stop(self):
        """停止事件总线"""
        self._running = False
        logger.info("EventBus stopped")
    
    def clear_subscribers(self):
        """清除所有订阅"""
        self._subscribers.clear()
        self._global_subscribers.clear()
        logger.info("All subscribers cleared")


class InterruptController:
    """
    打断控制器 - 管理Barge-in打断逻辑
    """
    
    def __init__(self, event_bus: EventBus):
        """
        初始化打断控制器
        
        Args:
            event_bus: 事件总线实例
        """
        self.event_bus = event_bus
        self._tts_playing = False
        self._interrupt_enabled = True
        self._interrupt_threshold = 0.6  # VAD置信度阈值
        self._min_speech_duration = 0.15  # 最小语音持续时间（秒）
        self._speech_start_time: Optional[float] = None
        
        # 订阅相关事件
        self.event_bus.subscribe(EventType.TTS_START, self._on_tts_start)
        self.event_bus.subscribe(EventType.TTS_END, self._on_tts_end)
        self.event_bus.subscribe(EventType.VAD_ACTIVE, self._on_vad_active)
        
        logger.info("InterruptController initialized")
    
    def _on_tts_start(self, event: Event):
        """TTS开始播放"""
        self._tts_playing = True
        logger.debug("TTS started, barge-in detection enabled")
    
    def _on_tts_end(self, event: Event):
        """TTS播放结束"""
        self._tts_playing = False
        self._speech_start_time = None
        logger.debug("TTS ended, barge-in detection disabled")
    
    def _on_vad_active(self, event: Event):
        """检测到语音活动"""
        if not self._tts_playing or not self._interrupt_enabled:
            return
        
        confidence = event.data.get("confidence", 0.0)
        energy = event.data.get("energy", 0.0)
        
        # 记录语音开始时间
        if self._speech_start_time is None:
            self._speech_start_time = datetime.now().timestamp()
        
        # 检查是否满足打断条件
        speech_duration = datetime.now().timestamp() - self._speech_start_time
        
        if (confidence >= self._interrupt_threshold and 
            speech_duration >= self._min_speech_duration):
            
            logger.info(f"Barge-in detected! confidence={confidence:.2f}, duration={speech_duration:.2f}s")
            
            # 发布打断事件
            self.event_bus.emit(
                EventType.BARGE_IN,
                data={
                    "confidence": confidence,
                    "duration": speech_duration,
                    "energy": energy
                },
                source="interrupt_controller",
                priority=2  # 高优先级
            )
            
            # 重置状态
            self._speech_start_time = None
    
    def enable_interrupt(self, enabled: bool = True):
        """启用/禁用打断功能"""
        self._interrupt_enabled = enabled
        logger.info(f"Interrupt {'enabled' if enabled else 'disabled'}")
    
    def set_threshold(self, threshold: float):
        """设置打断阈值"""
        self._interrupt_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Interrupt threshold set to {self._interrupt_threshold}")


# 全局事件总线实例
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def reset_event_bus():
    """重置全局事件总线"""
    global _global_event_bus
    if _global_event_bus is not None:
        _global_event_bus.clear_subscribers()
    _global_event_bus = None


# 便捷装饰器
def on_event(event_type: EventType):
    """
    事件处理装饰器
    
    Usage:
        @on_event(EventType.BARGE_IN)
        def handle_barge_in(event: Event):
            print(f"Barge-in detected: {event.data}")
    """
    def decorator(func: Callable):
        get_event_bus().subscribe(event_type, func)
        return func
    return decorator


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    logging.basicConfig(level=logging.DEBUG)
    
    async def test():
        bus = EventBus()
        
        # 测试订阅
        received_events = []
        
        async def handler(event: Event):
            received_events.append(event)
            print(f"Received: {event}")
        
        bus.subscribe(EventType.BARGE_IN, handler)
        
        # 启动事件总线
        await bus.start()
        
        # 发布事件
        await bus.emit_async(
            EventType.BARGE_IN,
            data={"confidence": 0.95},
            source="test"
        )
        
        # 等待处理
        await asyncio.sleep(0.5)
        
        # 停止
        await bus.stop()
        
        print(f"Received {len(received_events)} events")
    
    asyncio.run(test())
