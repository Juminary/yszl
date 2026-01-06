"""
声学前端集成演示 (Acoustic Frontend Integration Demo)

展示 ReSpeaker 6-Mic Circular Array 的全部功能:
1. 多通道音频采集
2. 实时 DOA 声源定位
3. 波束成形信号增强
4. 回声消除
5. LED声源方向指示

使用方法:
    python demo.py [--no-led] [--duration 10]

注意:
    - 需要在已安装 seeed-voicecard 的树莓派上运行
    - 确保 ReSpeaker 6-Mic HAT 正确连接
"""

import argparse
import logging
import time
import sys
import os
import wave
import numpy as np
from datetime import datetime
from typing import Optional

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acoustic_frontend import (
    MicrophoneArray, 
    MicArrayConfig,
    AudioFrame,
    LEDRing, 
    LEDPattern,
    Colors
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AcousticFrontendDemo:
    """
    声学前端集成演示
    
    展示所有声学处理能力的统一演示程序
    """
    
    def __init__(
        self,
        enable_led: bool = True,
        save_audio: bool = False,
        output_dir: str = "recordings",
    ):
        self.enable_led = enable_led
        self.save_audio = save_audio
        self.output_dir = output_dir
        
        # 组件
        self.mic: Optional[MicrophoneArray] = None
        self.led: Optional[LEDRing] = None
        
        # 录音缓冲
        self.recording_buffer = []
        
        # 统计
        self.stats = {
            "frames": 0,
            "speech_events": 0,
            "doa_updates": 0,
            "avg_energy": 0.0,
        }
    
    def setup(self):
        """初始化组件"""
        logger.info("=" * 50)
        logger.info("  医声智联 - 声学前端增强模块 v2.0")
        logger.info("  ReSpeaker 6-Mic Circular Array Demo")
        logger.info("=" * 50)
        
        # 创建麦克风阵列
        config = MicArrayConfig(
            sample_rate=16000,
            chunk_duration_ms=30,
            enable_doa=True,
            enable_beamforming=True,
            enable_aec=True,
            enable_vad=True,
        )
        self.mic = MicrophoneArray(config)
        
        # 注册回调
        self.mic.on("on_doa_update", self._on_doa_update)
        self.mic.on("on_speech_start", self._on_speech_start)
        self.mic.on("on_speech_end", self._on_speech_end)
        
        # 创建LED环
        if self.enable_led:
            self.led = LEDRing()
        
        # 创建录音目录
        if self.save_audio:
            os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Setup complete!")
    
    def _on_doa_update(self, frame: AudioFrame):
        """DOA更新回调"""
        self.stats["doa_updates"] += 1
        if self.led:
            self.led.show_doa(frame.doa_angle)
        logger.info(f"🎯 DOA: {frame.doa_angle:.1f}° (confidence: {frame.doa_confidence:.2f})")
    
    def _on_speech_start(self, frame: AudioFrame):
        """语音开始回调"""
        self.stats["speech_events"] += 1
        if self.led:
            self.led.set_pattern(LEDPattern.LISTENING)
        logger.info("🎤 语音开始")
    
    def _on_speech_end(self, frame: AudioFrame):
        """语音结束回调"""
        if self.led:
            self.led.set_pattern(LEDPattern.BREATHING)
        logger.info("🔇 语音结束")
    
    def run(self, duration: float = 30.0):
        """
        运行演示
        
        Args:
            duration: 运行时长 (秒)
        """
        logger.info(f"\n开始运行 {duration} 秒...")
        logger.info("请在麦克风周围说话并移动，观察DOA检测效果")
        logger.info("按 Ctrl+C 提前退出\n")
        
        # 启动组件
        self.mic.start()
        if self.led:
            self.led.start()
            self.led.set_pattern(LEDPattern.BREATHING)
        
        start_time = time.time()
        energy_sum = 0.0
        
        try:
            while time.time() - start_time < duration:
                # 读取音频帧
                frame = self.mic.read(timeout=0.5)
                
                if frame is None:
                    continue
                
                self.stats["frames"] += 1
                energy_sum += frame.energy
                
                # 保存音频
                if self.save_audio and frame.clean_audio is not None:
                    self.recording_buffer.append(frame.clean_audio)
                
                # 定期打印状态
                if self.stats["frames"] % 30 == 0:  # 约每秒
                    elapsed = time.time() - start_time
                    fps = self.stats["frames"] / elapsed
                    doa_str = f"{frame.doa_angle:.0f}°" if frame.doa_angle else "N/A"
                    
                    print(f"\r⏱ {elapsed:.1f}s | "
                          f"📦 帧: {self.stats['frames']} ({fps:.1f}/s) | "
                          f"🎯 DOA: {doa_str} | "
                          f"📊 能量: {frame.energy:.4f} | "
                          f"🗣 语音: {'✅' if frame.is_speech else '❌'}", 
                          end="", flush=True)
        
        except KeyboardInterrupt:
            logger.info("\n\n用户中断")
        
        finally:
            print()  # 换行
            
            # 停止组件
            if self.led:
                self.led.set_pattern(LEDPattern.SUCCESS)
                time.sleep(0.5)
                self.led.stop()
            self.mic.stop()
            
            # 计算统计
            self.stats["avg_energy"] = energy_sum / max(1, self.stats["frames"])
    
    def save_recording(self):
        """保存录音"""
        if not self.save_audio or not self.recording_buffer:
            return
        
        # 合并音频
        audio = np.concatenate(self.recording_buffer)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
        
        # 保存为WAV
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio.astype(np.int16).tobytes())
        
        logger.info(f"录音已保存: {filename} ({len(audio)/16000:.1f}秒)")
    
    def print_summary(self):
        """打印运行摘要"""
        logger.info("\n" + "=" * 50)
        logger.info("  运行摘要")
        logger.info("=" * 50)
        logger.info(f"  总帧数:      {self.stats['frames']}")
        logger.info(f"  语音事件:    {self.stats['speech_events']}")
        logger.info(f"  DOA更新:     {self.stats['doa_updates']}")
        logger.info(f"  平均能量:    {self.stats['avg_energy']:.4f}")
        
        mic_stats = self.mic.get_stats() if self.mic else {}
        logger.info(f"  组件状态:    DOA={mic_stats.get('components', {}).get('doa', False)}, "
                   f"BF={mic_stats.get('components', {}).get('beamformer', False)}, "
                   f"AEC={mic_stats.get('components', {}).get('aec', False)}")
        logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="医声智联 - 声学前端增强模块演示"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="运行时长 (秒), 默认30秒"
    )
    parser.add_argument(
        "--no-led",
        action="store_true",
        help="禁用LED指示"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="保存录音文件"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="recordings",
        help="录音保存目录"
    )
    
    args = parser.parse_args()
    
    # 创建并运行演示
    demo = AcousticFrontendDemo(
        enable_led=not args.no_led,
        save_audio=args.save,
        output_dir=args.output,
    )
    
    try:
        demo.setup()
        demo.run(duration=args.duration)
        
        if args.save:
            demo.save_recording()
        
        demo.print_summary()
        
    except Exception as e:
        logger.error(f"演示出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
