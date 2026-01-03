"""
急救模式检测器
基于对话内容和语音特征检测高危症状，触发急救响应
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EmergencyAlert:
    """急救警报"""
    level: str                    # critical, urgent, moderate, low
    score: float                  # 风险评分 0-1
    triggers: List[str]           # 触发关键词
    timestamp: datetime
    message: str
    recommended_actions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "score": self.score,
            "triggers": self.triggers,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "recommended_actions": self.recommended_actions
        }


class EmergencyDetector:
    """
    急救模式检测器
    
    功能：
    1. 检测高危症状（红旗症状）
    2. 分析语音特征（喘息、哭泣等）
    3. 风险分层评估
    4. 触发急救模式响应
    """
    
    # 风险级别定义
    RISK_LEVELS = {
        "critical": {"score": 0.9, "color": "red", "action": "立即急救"},
        "urgent": {"score": 0.7, "color": "orange", "action": "尽快就医"},
        "moderate": {"score": 0.5, "color": "yellow", "action": "限期就医"},
        "low": {"score": 0.3, "color": "green", "action": "常规咨询"}
    }
    
    # 危急症状（红旗症状）- 需要立即处理
    CRITICAL_SYMPTOMS = {
        # 心血管急症
        "压榨性胸痛": {"score": 1.0, "category": "心血管", "action": "疑似心梗，立即拨打120"},
        "心前区剧痛": {"score": 1.0, "category": "心血管", "action": "疑似心梗，立即拨打120"},
        "胸痛放射到左臂": {"score": 0.95, "category": "心血管", "action": "疑似心梗"},
        "濒死感": {"score": 0.95, "category": "心血管", "action": "疑似心梗"},
        "心跳骤停": {"score": 1.0, "category": "心血管", "action": "立即CPR并拨打120"},
        
        # 脑血管急症
        "突发偏瘫": {"score": 1.0, "category": "脑血管", "action": "疑似脑卒中，立即就医"},
        "言语不清": {"score": 0.85, "category": "脑血管", "action": "疑似脑卒中"},
        "口角歪斜": {"score": 0.85, "category": "脑血管", "action": "疑似脑卒中"},
        "剧烈头痛": {"score": 0.8, "category": "脑血管", "action": "排除脑出血"},
        "突然晕倒": {"score": 0.9, "category": "脑血管", "action": "意识丧失，立即就医"},
        "意识不清": {"score": 0.95, "category": "神经", "action": "立即就医"},
        "昏迷": {"score": 1.0, "category": "神经", "action": "立即拨打120"},
        "抽搐": {"score": 0.9, "category": "神经", "action": "癫痫或其他急症"},
        
        # 呼吸急症
        "呼吸困难": {"score": 0.85, "category": "呼吸", "action": "呼吸系统急症"},
        "喘不上气": {"score": 0.85, "category": "呼吸", "action": "呼吸系统急症"},
        "窒息": {"score": 1.0, "category": "呼吸", "action": "立即海姆立克急救"},
        "气道梗阻": {"score": 1.0, "category": "呼吸", "action": "立即急救"},
        
        # 出血
        "大量出血": {"score": 0.95, "category": "外伤", "action": "止血并拨打120"},
        "喷射状出血": {"score": 1.0, "category": "外伤", "action": "动脉出血，立即急救"},
        "咯血": {"score": 0.8, "category": "呼吸", "action": "内科急症"},
        "呕血": {"score": 0.85, "category": "消化", "action": "消化道出血"},
        
        # 过敏
        "过敏性休克": {"score": 1.0, "category": "过敏", "action": "立即使用肾上腺素"},
        "严重过敏": {"score": 0.9, "category": "过敏", "action": "立即就医"},
        "喉头水肿": {"score": 0.95, "category": "过敏", "action": "气道阻塞风险"},
        
        # 其他急症
        "休克": {"score": 1.0, "category": "急症", "action": "立即急救"},
        "高烧不退": {"score": 0.7, "category": "感染", "action": "尽快就医"},
        "剧烈腹痛": {"score": 0.8, "category": "消化", "action": "排除急腹症"},
        
        # 精神急症
        "自杀": {"score": 1.0, "category": "精神", "action": "心理危机干预"},
        "想死": {"score": 0.95, "category": "精神", "action": "心理危机干预"},
        "不想活": {"score": 0.9, "category": "精神", "action": "心理危机干预"},
        "自残": {"score": 0.9, "category": "精神", "action": "心理危机干预"},
    }
    
    # 紧急症状 - 需要尽快处理
    URGENT_SYMPTOMS = {
        "持续高热": {"score": 0.7, "category": "感染"},
        "剧烈疼痛": {"score": 0.7, "category": "疼痛"},
        "心悸": {"score": 0.6, "category": "心血管"},
        "胸闷": {"score": 0.6, "category": "心血管"},
        "头晕严重": {"score": 0.6, "category": "神经"},
        "恶心呕吐": {"score": 0.5, "category": "消化"},
        "血尿": {"score": 0.6, "category": "泌尿"},
        "血便": {"score": 0.7, "category": "消化"},
    }
    
    # 语音特征分析关键词
    AUDIO_DISTRESS_PATTERNS = {
        "喘息": 0.3,
        "哭泣": 0.2,
        "叫喊": 0.3,
        "尖叫": 0.4,
        "呻吟": 0.2,
        "呼救": 0.5
    }
    
    def __init__(self, location_service=None):
        """
        初始化急救检测器
        
        Args:
            location_service: 位置服务（可选，用于查找最近急诊室）
        """
        self.location_service = location_service
        self.alert_history: List[EmergencyAlert] = []
        
        # 编译正则表达式
        self._compile_patterns()
        
        logger.info("[急救检测] 初始化完成")
    
    def _compile_patterns(self):
        """编译症状匹配模式"""
        self.critical_patterns = []
        for symptom, info in self.CRITICAL_SYMPTOMS.items():
            # 创建模式（支持同义词和变体）
            pattern = self._create_pattern(symptom)
            self.critical_patterns.append({
                "pattern": pattern,
                "symptom": symptom,
                "info": info
            })
        
        self.urgent_patterns = []
        for symptom, info in self.URGENT_SYMPTOMS.items():
            pattern = self._create_pattern(symptom)
            self.urgent_patterns.append({
                "pattern": pattern,
                "symptom": symptom,
                "info": info
            })
    
    def _create_pattern(self, symptom: str) -> re.Pattern:
        """为症状创建匹配模式"""
        # 简单实现：允许中间有少量字符
        chars = list(symptom)
        pattern = ".{0,2}".join(chars)
        return re.compile(pattern, re.IGNORECASE)
    
    def assess_risk(self, text: str, audio_features: Dict = None) -> EmergencyAlert:
        """
        评估对话的风险级别
        
        Args:
            text: 对话文本
            audio_features: 副语言特征（可选）
                - cough_detected: bool
                - respiratory_distress: bool
                - crying_detected: bool
                - anxiety_level: str (low/medium/high)
            
        Returns:
            EmergencyAlert 对象
        """
        triggers = []
        max_score = 0.0
        category = None
        action = None
        
        # 检测危急症状
        for item in self.critical_patterns:
            if item["pattern"].search(text):
                triggers.append(item["symptom"])
                if item["info"]["score"] > max_score:
                    max_score = item["info"]["score"]
                    category = item["info"]["category"]
                    action = item["info"].get("action", "立即就医")
        
        # 如果没有危急症状，检测紧急症状
        if max_score < 0.7:
            for item in self.urgent_patterns:
                if item["pattern"].search(text):
                    triggers.append(item["symptom"])
                    if item["info"]["score"] > max_score:
                        max_score = item["info"]["score"]
                        category = item["info"]["category"]
        
        # 考虑语音特征
        if audio_features:
            audio_score = self._analyze_audio_features(audio_features)
            max_score = min(1.0, max_score + audio_score * 0.2)
            
            if audio_features.get("respiratory_distress"):
                triggers.append("呼吸困难（语音检测）")
            if audio_features.get("crying_detected"):
                triggers.append("情绪激动（语音检测）")
        
        # 确定风险级别
        if max_score >= 0.9:
            level = "critical"
        elif max_score >= 0.7:
            level = "urgent"
        elif max_score >= 0.5:
            level = "moderate"
        else:
            level = "low"
        
        # 生成消息
        if level == "critical":
            message = f"⚠️ 检测到危急症状：{', '.join(triggers[:3])}。{action or '请立即拨打急救电话120！'}"
        elif level == "urgent":
            message = f"检测到紧急症状：{', '.join(triggers[:3])}。建议尽快就医。"
        elif level == "moderate":
            message = f"检测到需关注的症状：{', '.join(triggers[:3])}。建议限期就诊。"
        else:
            message = "暂未检测到紧急症状。"
        
        # 生成建议操作
        recommended_actions = self._generate_actions(level, category, triggers)
        
        alert = EmergencyAlert(
            level=level,
            score=max_score,
            triggers=triggers,
            timestamp=datetime.now(),
            message=message,
            recommended_actions=recommended_actions
        )
        
        # 记录历史
        if level in ["critical", "urgent"]:
            self.alert_history.append(alert)
            logger.warning(f"[急救检测] {level.upper()}: {message}")
        
        return alert
    
    def _analyze_audio_features(self, features: Dict) -> float:
        """分析语音特征，返回额外风险分数"""
        score = 0.0
        
        if features.get("respiratory_distress"):
            score += 0.3
        if features.get("crying_detected"):
            score += 0.1
        if features.get("anxiety_level") == "high":
            score += 0.2
        elif features.get("anxiety_level") == "medium":
            score += 0.1
        if features.get("cough_detected"):
            score += 0.05
        
        return min(score, 0.5)
    
    def _generate_actions(self, level: str, category: str, triggers: List[str]) -> List[Dict]:
        """生成建议操作列表"""
        actions = []
        
        if level == "critical":
            actions.append({
                "type": "call",
                "priority": 1,
                "label": "拨打急救电话",
                "number": "120",
                "icon": "🚑"
            })
            actions.append({
                "type": "location",
                "priority": 2,
                "label": "查找最近急诊室",
                "action": "find_nearest_er",
                "icon": "📍"
            })
            actions.append({
                "type": "alert",
                "priority": 3,
                "label": "告知症状",
                "message": f"危急症状: {', '.join(triggers[:2])}",
                "icon": "⚠️"
            })
            
        elif level == "urgent":
            actions.append({
                "type": "location",
                "priority": 1,
                "label": "查找附近医院",
                "action": "find_nearest_hospital",
                "icon": "🏥"
            })
            actions.append({
                "type": "info",
                "priority": 2,
                "label": "就诊建议",
                "message": f"建议科室: {self._suggest_department(category)}",
                "icon": "💡"
            })
            
        elif level == "moderate":
            actions.append({
                "type": "info",
                "priority": 1,
                "label": "就诊建议",
                "message": f"建议预约{self._suggest_department(category)}门诊",
                "icon": "📋"
            })
        
        return actions
    
    def _suggest_department(self, category: str) -> str:
        """根据症状类别建议科室"""
        department_map = {
            "心血管": "心内科/急诊科",
            "脑血管": "神经内科/急诊科",
            "神经": "神经内科",
            "呼吸": "呼吸内科/急诊科",
            "消化": "消化内科",
            "外伤": "外科/急诊外科",
            "过敏": "急诊科/变态反应科",
            "感染": "感染科/发热门诊",
            "泌尿": "泌尿外科",
            "精神": "精神科/心理科",
            "疼痛": "疼痛科/相关专科"
        }
        return department_map.get(category, "急诊科")
    
    def trigger_emergency_mode(self, alert: EmergencyAlert) -> Dict:
        """
        触发急救模式
        
        Args:
            alert: 急救警报
            
        Returns:
            急救模式响应
        """
        logger.critical(f"[急救模式] 已触发！级别: {alert.level}")
        
        response = {
            "mode": "emergency",
            "alert": alert.to_dict(),
            "ui_config": {
                "color_scheme": "emergency_red",
                "sound_alert": True,
                "vibration": True,
                "fullscreen_alert": alert.level == "critical"
            },
            "workflow": {
                "current_step": "assessment",
                "next_steps": ["call_emergency", "find_location", "first_aid_guidance"]
            }
        }
        
        # 如果是危急级别，添加紧急联系人通知
        if alert.level == "critical":
            response["notifications"] = [
                {"type": "emergency_contact", "message": "紧急情况通知"},
                {"type": "medical_history", "request": "准备病史信息"}
            ]
        
        return response
    
    def get_first_aid_guidance(self, symptom_category: str) -> Dict:
        """
        获取急救指导
        
        Args:
            symptom_category: 症状类别
            
        Returns:
            急救指导信息
        """
        guidance = {
            "心血管": {
                "title": "疑似心梗急救",
                "steps": [
                    "1. 立即拨打120急救电话",
                    "2. 让患者平卧或半卧位",
                    "3. 如有硝酸甘油可舌下含服",
                    "4. 解开衣领，保持呼吸顺畅",
                    "5. 如心跳停止，立即进行心肺复苏"
                ],
                "warning": "不要让患者随意走动"
            },
            "脑血管": {
                "title": "疑似脑卒中急救",
                "steps": [
                    "1. 立即拨打120急救电话",
                    "2. 让患者平卧，头稍抬高",
                    "3. 清除口腔异物，保持呼吸道通畅",
                    "4. 记录发病时间（很重要！）",
                    "5. 不要喂水喂药"
                ],
                "warning": "时间就是大脑！"
            },
            "呼吸": {
                "title": "呼吸困难急救",
                "steps": [
                    "1. 帮助患者采取舒适体位",
                    "2. 打开窗户通风",
                    "3. 解开紧身衣物",
                    "4. 如有吸氧设备可辅助吸氧",
                    "5. 持续观察呼吸情况"
                ],
                "warning": "如呼吸停止立即人工呼吸"
            },
            "过敏": {
                "title": "过敏反应急救",
                "steps": [
                    "1. 立即远离过敏原",
                    "2. 如有肾上腺素笔立即使用",
                    "3. 让患者平卧抬高双腿",
                    "4. 观察呼吸和意识",
                    "5. 立即拨打120"
                ],
                "warning": "过敏性休克可能致命"
            },
            "精神": {
                "title": "心理危机干预",
                "steps": [
                    "1. 保持冷静，用平和语气交流",
                    "2. 倾听对方，不要评判",
                    "3. 询问是否有自伤打算",
                    "4. 确保周围没有危险物品",
                    "5. 拨打心理援助热线：010-82951332"
                ],
                "hotlines": [
                    {"name": "北京心理危机研究与干预中心", "number": "010-82951332"},
                    {"name": "全国心理援助热线", "number": "400-161-9995"}
                ]
            }
        }
        
        return guidance.get(symptom_category, {
            "title": "一般急救",
            "steps": [
                "1. 保持冷静",
                "2. 评估情况",
                "3. 如需帮助拨打120"
            ]
        })
    
    def check_text_for_emergency(self, text: str) -> bool:
        """
        快速检查文本是否包含急救关键词
        
        Args:
            text: 输入文本
            
        Returns:
            是否需要触发急救检测
        """
        for item in self.critical_patterns:
            if item["pattern"].search(text):
                return True
        return False
    
    def get_alert_history(self) -> List[Dict]:
        """获取警报历史"""
        return [alert.to_dict() for alert in self.alert_history]


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    detector = EmergencyDetector()
    
    test_cases = [
        "我胸口压榨性疼痛，感觉喘不上气",
        "医生，我突然左边胳膊不能动了，说话也不利索",
        "我头疼了两天，吃了止疼药稍微好点",
        "我最近有点咳嗽，流鼻涕",
        "我不想活了，感觉生活没有意义",
        "孩子吃东西噎住了，脸都憋紫了"
    ]
    
    print("=== 急救检测测试 ===\n")
    
    for text in test_cases:
        alert = detector.assess_risk(text)
        
        level_emoji = {"critical": "🚨", "urgent": "⚠️", "moderate": "⚡", "low": "✓"}
        print(f"{level_emoji.get(alert.level, '?')} [{alert.level.upper():8s}] 分数: {alert.score:.2f}")
        print(f"   输入: {text}")
        print(f"   {alert.message}")
        if alert.triggers:
            print(f"   触发词: {', '.join(alert.triggers)}")
        if alert.recommended_actions:
            print(f"   建议: {alert.recommended_actions[0].get('label', '')}")
        print()
