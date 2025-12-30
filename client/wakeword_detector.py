"""
唤醒词检测模块
使用ASR检测唤醒词"康康"
"""

import requests
import logging
import numpy as np
from pathlib import Path
import time
from typing import Tuple

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """唤醒词检测器"""
    
    def __init__(self, server_url: str, wakeword: str = "康康", 
                 confidence_threshold: float = 0.5):
        """
        初始化唤醒词检测器
        
        Args:
            server_url: 服务器地址
            wakeword: 唤醒词
            confidence_threshold: 置信度阈值（用于模糊匹配）
        """
        self.server_url = server_url
        self.wakeword = wakeword
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"WakeWordDetector initialized: wakeword='{wakeword}', server_url='{server_url}'")
    
    def detect_in_text(self, text: str) -> bool:
        """
        检测文本中是否包含唤醒词
        
        Args:
            text: 识别的文本
            
        Returns:
            是否检测到唤醒词
        """
        if not text:
            return False
        
        # 去除标点符号和空格，进行模糊匹配
        text_clean = text.replace(" ", "").replace("，", "").replace("。", "")
        wakeword_clean = self.wakeword.replace(" ", "")
        
        # 精确匹配
        if wakeword_clean in text_clean:
            return True
        
        # 模糊匹配：检查是否包含唤醒词的每个字（顺序可以不同，但必须都出现）
        wakeword_chars = set(wakeword_clean)
        text_chars = set(text_clean)
        
        # 如果唤醒词的所有字符都在文本中，认为检测到
        if wakeword_chars.issubset(text_chars):
            # 进一步检查顺序（简单版本：检查是否连续出现）
            for i in range(len(text_clean) - len(wakeword_clean) + 1):
                if text_clean[i:i+len(wakeword_clean)] == wakeword_clean:
                    return True
        
        return False
    
    def detect_in_audio(self, audio_path: str) -> Tuple[bool, str]:
        """
        在音频中检测唤醒词
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (是否检测到, 识别的文本)
        """
        try:
            # 发送到ASR接口
            logger.debug(f"Sending audio to ASR endpoint: {self.server_url}/asr")
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(
                    f"{self.server_url}/asr",
                    files=files,
                    timeout=10  # 增加超时时间
                )
            
            logger.debug(f"ASR response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    text = result.get('text', '').strip()
                    logger.info(f"ASR recognized text: '{text}'")
                    
                    # 检测唤醒词
                    detected = self.detect_in_text(text)
                    
                    if detected:
                        logger.info(f"✅ Wake word detected! Text: '{text}'")
                    else:
                        logger.debug(f"Wake word not detected. Text: '{text}'")
                    
                    return detected, text
                except Exception as e:
                    logger.error(f"Failed to parse ASR response: {e}, response: {response.text[:200]}")
                    return False, ""
            else:
                logger.warning(f"ASR request failed: {response.status_code}, response: {response.text[:200]}")
                return False, ""
                
        except requests.exceptions.Timeout:
            logger.error(f"ASR request timeout (>{10}s)")
            return False, ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"ASR connection error: {e}")
            return False, ""
        except Exception as e:
            logger.error(f"Wake word detection failed: {e}", exc_info=True)
            return False, ""
    
    def listen_for_wakeword(self, capture, check_interval: float = 2.0,
                           max_listen_time: float = 300.0) -> Tuple[bool, str]:
        """
        持续监听唤醒词
        
        Args:
            capture: AudioCapture 实例
            check_interval: 每次检测的间隔（秒）
            max_listen_time: 最大监听时间（秒）
            
        Returns:
            (是否检测到, 识别的文本)
        """
        logger.info(f"Listening for wake word: '{self.wakeword}'")
        print(f"\n🎤 正在监听唤醒词: '{self.wakeword}'")
        print("   请说出唤醒词以开始对话...")
        
        start_time = time.time()
        temp_audio = "temp_wakeword_check.wav"
        
        try:
            while time.time() - start_time < max_listen_time:
                # 录制短音频片段用于检测
                audio = capture.record(
                    duration=check_interval,
                    output_path=temp_audio
                )
                
                if len(audio) == 0:
                    logger.debug("No audio recorded, skipping...")
                    continue
                
                logger.debug(f"Checking wake word in audio segment ({len(audio)} samples)...")
                # 检测唤醒词
                detected, text = self.detect_in_audio(temp_audio)
                
                if detected:
                    print(f"\n✅ 检测到唤醒词！识别文本: '{text}'")
                    return True, text
                
                # 显示监听状态（每5秒显示一次）
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    print(f"   监听中... ({int(elapsed)}秒)", end='\r')
                
                # 清理临时文件
                Path(temp_audio).unlink(missing_ok=True)
            
            logger.info("Wake word listening timeout")
            return False, ""
            
        except KeyboardInterrupt:
            logger.info("Wake word listening interrupted")
            return False, ""
        finally:
            # 清理临时文件
            Path(temp_audio).unlink(missing_ok=True)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    detector = WakeWordDetector(
        server_url="http://localhost:6006",
        wakeword="康康"
    )
    
    # 测试文本检测
    test_texts = [
        "康康",
        "你好康康",
        "康康你好",
        "康康，帮我一下",
        "康康医生",
        "医生",
        "你好"
    ]
    
    print("测试文本检测:")
    for text in test_texts:
        detected = detector.detect_in_text(text)
        print(f"  '{text}' -> {detected}")
