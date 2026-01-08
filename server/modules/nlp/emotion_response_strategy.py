"""
情感回复策略模块 (Emotion Response Strategy)

实现官方 Emotional VoiceChat 架构中的情感感知回复生成：
1. 根据用户情绪选择合适的语音风格
2. 构建引导 LLM 输出结构化内容的 prompt
3. 解析 LLM 输出为 {style, content} 格式
"""

import re
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EmotionResponseStrategy:
    """
    情感回复策略器
    
    用于实现 SenseVoice → LLM → CosyVoice 的情感感知对话流程
    """
    
    # 用户情绪到 AI 语音风格的映射（中文版）
    # 注意：不是简单镜像用户情绪，而是选择能支持/安抚用户的风格
    EMOTION_TO_STYLE = {
        # 基础情感（6种）
        "sad": "用温暖关怀的语气，语速稍慢，表达理解和同情",
        "angry": "用平静专业的语气，语速适中，耐心倾听不争辩",
        "fear": "用平稳安心的语气，语速稍慢，给予肯定和支持",
        "happy": "用轻松愉快的语气，语速稍快，保持积极氛围",
        "surprise": "用清晰明了的语气，语速适中，简洁解释情况",
        "neutral": "用专业友好的语气，语速适中，态度亲切",
        
        # 扩展情感（7种）
        "depressed": "用深度共情的语气，语速缓慢，表达理解并鼓励寻求帮助",
        "anxious": "用舒缓镇定的语气，语速平稳，逐步引导放松",
        "pain": "用温和关心的语气，语速稍慢，明确建议就医",
        "confused": "用清晰耐心的语气，语速适中，简洁易懂地解释",
        "grateful": "用温馨友好的语气，语速适中，保持正面互动",
        "frustrated": "用理解包容的语气，语速适中，认可情绪并积极解决",
        "tired": "用关切体贴的语气，语速稍慢，建议休息和关爱自己",
    }
    
    # 用于 CosyVoice instruct 模式的英文风格描述（更适合模型理解）
    EMOTION_TO_INSTRUCT = {
        # 基础情感（6种）
        "sad": "Speak with a warm, comforting tone, slower pace, and gentle emotion",
        "angry": "Speak with a calm, professional tone, medium pace, and patient manner",
        "fear": "Speak with a reassuring, steady tone, slower pace, and supportive emotion",
        "happy": "Speak with a cheerful, upbeat tone, slightly faster pace, and positive emotion",
        "surprise": "Speak with a clear, explanatory tone, medium pace, and composed manner",
        "neutral": "Speak with a friendly, professional tone, medium pace, and welcoming manner",
        
        # 扩展情感（7种）
        "depressed": "Speak with a deeply empathetic tone, slow pace, gentle and encouraging",
        "anxious": "Speak with a soothing, calming tone, steady pace, guiding relaxation",
        "pain": "Speak with a gentle, caring tone, slower pace, expressing concern",
        "confused": "Speak with a clear, patient tone, medium pace, easy to understand",
        "grateful": "Speak with a warm, friendly tone, medium pace, positive and appreciative",
        "frustrated": "Speak with an understanding tone, medium pace, acknowledging and solution-focused",
        "tired": "Speak with a caring, gentle tone, slower pace, suggesting rest and self-care",
    }
    
    def __init__(self):
        """初始化情感回复策略器"""
        logger.info("[EmotionResponseStrategy] 初始化完成")
    
    def get_style_for_emotion(self, emotion: str) -> str:
        """
        获取情绪对应的语音风格描述（中文）
        
        Args:
            emotion: 用户情绪 (sad, angry, fear, happy, surprise, neutral)
            
        Returns:
            风格描述字符串
        """
        emotion = (emotion or "neutral").lower().strip()
        return self.EMOTION_TO_STYLE.get(emotion, self.EMOTION_TO_STYLE["neutral"])
    
    def get_instruct_for_emotion(self, emotion: str) -> str:
        """
        获取情绪对应的 CosyVoice instruct 文本（英文）
        
        Args:
            emotion: 用户情绪
            
        Returns:
            CosyVoice instruct 文本
        """
        emotion = (emotion or "neutral").lower().strip()
        return self.EMOTION_TO_INSTRUCT.get(emotion, self.EMOTION_TO_INSTRUCT["neutral"])
    
    def build_emotional_system_prompt(self, emotion: str, base_prompt: str = None) -> str:
        """
        构建情感感知的系统提示词
        
        引导 LLM 输出格式：[风格描述]<endofprompt>[回复内容]
        
        Args:
            emotion: 用户情绪
            base_prompt: 基础系统提示词（可选）
            
        Returns:
            完整的情感感知系统提示词
        """
        style_hint = self.get_style_for_emotion(emotion)
        
        emotional_instruction = f"""
你是一个富有同理心的医疗语音助手。用户当前的情绪是【{emotion}】。

你需要根据用户情绪调整回复风格。推荐的回复风格是：{style_hint}

【输出格式要求】（必须严格遵守）：
你的回复必须包含两部分，用 <endofprompt> 分隔：
1. 方括号内的风格描述（8-15个中文字，描述语气和语速）
2. 实际回复内容（1-3句话，口语化，不用列表）

示例格式：
[温暖关怀的语气，语速稍慢]<endofprompt>我理解你的感受，身体不舒服确实让人很担心。别着急，我来帮你分析一下。

【内容要求】：
- 回复内容必须是纯中文，适合语音朗读
- 不要使用数字、英文、特殊符号
- 不要使用列表、编号、分点格式
- 用逗号和句号，不用冒号、问号、感叹号
"""
        
        if base_prompt:
            return f"{base_prompt}\n\n{emotional_instruction}"
        else:
            return emotional_instruction
    
    def parse_emotional_response(self, llm_output: str) -> Dict[str, str]:
        """
        解析 LLM 的情感感知输出
        
        将 "[风格描述]<endofprompt>[回复内容]" 格式解析为结构化数据
        
        Args:
            llm_output: LLM 原始输出
            
        Returns:
            {"style": 风格描述, "content": 回复内容, "raw": 原始输出}
        """
        if not llm_output:
            return {"style": "", "content": "", "raw": ""}
        
        raw = llm_output.strip()
        
        # 检查是否包含分隔符
        if "<endofprompt>" not in raw:
            # 没有分隔符，整体作为 content
            logger.warning("[EmotionResponseStrategy] LLM 输出缺少 <endofprompt> 分隔符")
            return {"style": "", "content": raw, "raw": raw}
        
        # 分割
        parts = raw.split("<endofprompt>", maxsplit=1)
        style_part = parts[0].strip()
        content_part = parts[1].strip() if len(parts) > 1 else ""
        
        # 提取方括号内的风格描述
        style_match = re.search(r'\[([^\]]+)\]', style_part)
        if style_match:
            style = style_match.group(1).strip()
        else:
            style = style_part
        
        return {
            "style": style,
            "content": content_part,
            "raw": raw
        }
    
    def get_info(self) -> Dict:
        """获取模块信息"""
        return {
            "name": "EmotionResponseStrategy",
            "supported_emotions": list(self.EMOTION_TO_STYLE.keys()),
            "output_format": "[style]<endofprompt>[content]"
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    strategy = EmotionResponseStrategy()
    
    print("=== 情感回复策略测试 ===\n")
    
    # 测试风格获取
    for emotion in ["sad", "angry", "happy", "neutral"]:
        style = strategy.get_style_for_emotion(emotion)
        instruct = strategy.get_instruct_for_emotion(emotion)
        print(f"情绪: {emotion}")
        print(f"  中文风格: {style}")
        print(f"  Instruct: {instruct}")
        print()
    
    # 测试解析
    test_outputs = [
        "[温暖关怀的语气，语速稍慢]<endofprompt>我理解你的感受，身体不舒服确实让人担心。",
        "没有分隔符的普通回复",
        "[轻松愉快]<endofprompt>太好了，继续保持这种积极的心态。"
    ]
    
    print("=== 解析测试 ===\n")
    for output in test_outputs:
        result = strategy.parse_emotional_response(output)
        print(f"输入: {output[:50]}...")
        print(f"  style: {result['style']}")
        print(f"  content: {result['content']}")
        print()
