"""
临床实体抽取模块
基于医学本体和 LLM 从对话中提取药物、症状、疾病等实体
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ClinicalEntity:
    """临床实体"""
    text: str                    # 原始文本
    type: str                    # 实体类型
    normalized: str = None       # 标准化名称
    attributes: Dict = field(default_factory=dict)  # 属性（如剂量、频率）
    source_offset: Tuple[int, int] = None  # 在原文中的位置
    confidence: float = 1.0      # 置信度
    ontology_code: str = None    # 本体编码（如 SNOMED CT）
    speaker_role: str = None     # 提及者角色
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "type": self.type,
            "normalized": self.normalized,
            "attributes": self.attributes,
            "source_offset": self.source_offset,
            "confidence": self.confidence,
            "ontology_code": self.ontology_code,
            "speaker_role": self.speaker_role
        }


class ClinicalEntityExtractor:
    """
    临床实体抽取器
    
    支持的实体类型：
    - symptom: 症状
    - disease: 疾病
    - medication: 药物
    - dosage: 剂量
    - frequency: 频率
    - route: 给药途径
    - procedure: 检查/治疗
    - body_part: 部位
    - time: 时间
    - severity: 严重程度
    """
    
    ENTITY_TYPES = {
        "symptom": "症状",
        "disease": "疾病",
        "medication": "药物",
        "dosage": "剂量",
        "frequency": "频率",
        "route": "给药途径",
        "procedure": "检查/治疗",
        "body_part": "部位",
        "time": "时间",
        "severity": "严重程度"
    }
    
    # 常见症状词
    SYMPTOM_PATTERNS = [
        r"(头痛|头晕|头疼)",
        r"(胸痛|胸闷|心悸|心慌)",
        r"(腹痛|腹泻|便秘|恶心|呕吐)",
        r"(咳嗽|咳痰|气喘|呼吸困难)",
        r"(发热|发烧|高烧|低烧)",
        r"(疼痛|酸痛|刺痛|钝痛|绞痛)",
        r"(乏力|疲劳|无力)",
        r"(失眠|嗜睡|多梦)",
        r"(食欲.*[减退下降不振])",
        r"(水肿|浮肿)",
        r"(出血|便血|咯血|血尿)",
        r"(皮疹|瘙痒|红肿)",
        r"(麻木|刺痛感)",
        r"(视力.*[模糊下降])",
        r"(耳鸣|听力下降)"
    ]
    
    # 常见疾病词
    DISEASE_PATTERNS = [
        r"(感冒|流感)",
        r"(肺炎|支气管炎|哮喘)",
        r"(高血压|低血压)",
        r"(糖尿病|血糖高)",
        r"(冠心病|心肌梗[死塞]|心绞痛)",
        r"(脑梗[死塞]|脑出血|中风)",
        r"(胃炎|胃溃疡|十二指肠溃疡)",
        r"(肝炎|肝硬化|脂肪肝)",
        r"(肾炎|肾结石|尿路感染)",
        r"(关节炎|风湿|类风湿)",
        r"(骨折|扭伤|拉伤)",
        r"(过敏|荨麻疹|湿疹)",
        r"(贫血|白血病)",
        r"(甲[状腺]*亢|甲减)",
        r"(抑郁[症]?|焦虑[症]?)"
    ]
    
    # 常见药物词
    MEDICATION_PATTERNS = [
        r"(阿司匹林|阿斯匹林)",
        r"(布洛芬|芬必得)",
        r"(头孢.*素|青霉素|阿莫西林)",
        r"(降压药|降糖药)",
        r"(二甲双胍|胰岛素)",
        r"(硝苯地平|氨氯地平|缬沙坦)",
        r"(阿托伐他汀|辛伐他汀)",
        r"(氯吡格雷|华法林)",
        r"(奥美拉唑|雷贝拉唑|泮托拉唑)",
        r"(止痛药|止咳药|止泻药)",
        r"(消炎药|抗生素|退烧药)",
        r"(维生素[A-Za-z0-9]*)",
        r"(感冒药|清开灵|连花清瘟)",
        r"(钙片|铁剂)",
        r"(安定|地西泮|艾司唑仑)"
    ]
    
    # 剂量模式
    DOSAGE_PATTERNS = [
        r"(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|颗|支|袋|包)",
        r"(半片|一片|两片|[一二三四五六七八九十]+片)",
        r"(半粒|一粒|两粒|[一二三四五六七八九十]+粒)"
    ]
    
    # 频率模式
    FREQUENCY_PATTERNS = [
        r"(每[天日]|一[天日])\s*([一二三四五六七八九十\d]+)\s*次",
        r"([一二三])\s*日\s*([一二三四五六七八九十\d]+)\s*次",
        r"(每隔\s*\d+\s*小时)",
        r"(饭前|饭后|餐前|餐后|空腹|睡前)",
        r"(早[上晨]|中午|晚[上间]|临睡前)",
        r"(qd|bid|tid|qid|prn)"
    ]
    
    # 给药途径模式
    ROUTE_PATTERNS = [
        r"(口服|内服|外用)",
        r"(静脉注射|肌肉注射|皮下注射)",
        r"(静滴|输液)",
        r"(涂抹|外敷|贴)",
        r"(雾化吸入|吸入)",
        r"(舌下含服|含化)",
        r"(灌肠|栓剂)"
    ]
    
    # 部位模式
    BODY_PART_PATTERNS = [
        r"(头部?|头顶|太阳穴|后脑)",
        r"(眼[睛部]?|耳朵?|鼻子?|嘴[巴唇]?|口腔)",
        r"(颈[部椎]?|咽喉|喉咙)",
        r"(胸[部口腔]?|胸腔|心脏|肺[部]?)",
        r"(腹[部]?|胃[部]?|肝[脏部]?|肾[脏部]?|脾[脏]?)",
        r"(腰[部椎]?|背[部]?|脊[柱椎]?)",
        r"(肩[膀部]?|手[臂腕]?|手指|肘[部关节]?)",
        r"(腿[部]?|膝[盖关节]?|脚[踝腕]?|足[部]?)",
        r"(皮肤|淋巴[结]?|关节)"
    ]
    
    # 时间模式
    TIME_PATTERNS = [
        r"(\d+)\s*(分钟|小时|天|周|月|年)\s*[前后]?",
        r"(昨天|今天|前天|上周|上个?月|去年)",
        r"(最近|近[期来]|刚[刚才])",
        r"(持续|已经)\s*(\d+)\s*(分钟|小时|天|周|月|年)",
        r"(突然|忽然|慢慢)"
    ]
    
    # 严重程度模式
    SEVERITY_PATTERNS = [
        r"(轻微|轻度|微微|有点|稍微)",
        r"(中度|比较|挺)",
        r"(严重|剧烈|非常|特别|很|极|重度)",
        r"(越来越|加重|恶化|好转|缓解|减轻)"
    ]
    
    # 检查/治疗模式
    PROCEDURE_PATTERNS = [
        r"(CT|核磁|MRI|B超|彩超|X光|胸片)",
        r"(心电图|脑电图|肌电图)",
        r"(血[常规检]|尿[常规检]|便[常规检])",
        r"(肝功[能]?|肾功[能]?|血糖|血脂|血压)",
        r"(胃镜|肠镜|支气管镜)",
        r"(活检|穿刺|手术)",
        r"(化疗|放疗|透析)",
        r"(理疗|针灸|推拿|按摩)"
    ]
    
    def __init__(self, knowledge_graph=None, dialogue_module=None):
        """
        初始化实体抽取器
        
        Args:
            knowledge_graph: 知识图谱模块（用于实体链接）
            dialogue_module: 对话模块（用于 LLM 增强抽取）
        """
        self.kg = knowledge_graph
        self.llm = dialogue_module
        
        # 编译正则表达式
        self._compile_patterns()
        
        logger.info("[临床实体抽取] 初始化完成")
    
    def _compile_patterns(self):
        """编译所有正则表达式模式"""
        self.compiled_patterns = {
            "symptom": [re.compile(p, re.IGNORECASE) for p in self.SYMPTOM_PATTERNS],
            "disease": [re.compile(p, re.IGNORECASE) for p in self.DISEASE_PATTERNS],
            "medication": [re.compile(p, re.IGNORECASE) for p in self.MEDICATION_PATTERNS],
            "dosage": [re.compile(p, re.IGNORECASE) for p in self.DOSAGE_PATTERNS],
            "frequency": [re.compile(p, re.IGNORECASE) for p in self.FREQUENCY_PATTERNS],
            "route": [re.compile(p, re.IGNORECASE) for p in self.ROUTE_PATTERNS],
            "body_part": [re.compile(p, re.IGNORECASE) for p in self.BODY_PART_PATTERNS],
            "time": [re.compile(p, re.IGNORECASE) for p in self.TIME_PATTERNS],
            "severity": [re.compile(p, re.IGNORECASE) for p in self.SEVERITY_PATTERNS],
            "procedure": [re.compile(p, re.IGNORECASE) for p in self.PROCEDURE_PATTERNS]
        }
    
    def extract_entities(self, text: str, speaker_role: str = None) -> List[ClinicalEntity]:
        """
        从文本中抽取临床实体
        
        Args:
            text: 对话文本
            speaker_role: 说话人角色（区分主诉 vs 复述）
            
        Returns:
            实体列表
        """
        entities = []
        
        # 对每种实体类型进行匹配
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity_text = match.group(0)
                    
                    # 尝试从知识图谱获取标准化名称
                    normalized = self._normalize_entity(entity_text, entity_type)
                    
                    entity = ClinicalEntity(
                        text=entity_text,
                        type=entity_type,
                        normalized=normalized,
                        source_offset=(match.start(), match.end()),
                        confidence=0.9,  # 基于规则的置信度
                        speaker_role=speaker_role
                    )
                    
                    # 提取相关属性
                    entity.attributes = self._extract_attributes(text, entity_text, entity_type)
                    
                    entities.append(entity)
        
        # 去重（同一位置的实体只保留一个）
        entities = self._deduplicate_entities(entities)
        
        # 关联药物和剂量
        entities = self._associate_medication_dosage(text, entities)
        
        return entities
    
    def _normalize_entity(self, entity_text: str, entity_type: str) -> str:
        """标准化实体名称"""
        # 简单的标准化映射
        normalization_map = {
            "头疼": "头痛",
            "肚子疼": "腹痛",
            "发烧": "发热",
            "高烧": "高热",
            "心慌": "心悸",
            "阿斯匹林": "阿司匹林",
            "芬必得": "布洛芬",
            "降压药": "抗高血压药",
            "消炎药": "抗炎药物",
            "血糖高": "高血糖",
            "血压高": "高血压"
        }
        
        return normalization_map.get(entity_text, entity_text)
    
    def _extract_attributes(self, text: str, entity_text: str, entity_type: str) -> Dict:
        """提取实体相关属性"""
        attributes = {}
        
        # 对于药物，在其附近查找剂量和频率
        if entity_type == "medication":
            # 在实体前后 50 个字符内查找
            entity_pos = text.find(entity_text)
            context_start = max(0, entity_pos - 50)
            context_end = min(len(text), entity_pos + len(entity_text) + 50)
            context = text[context_start:context_end]
            
            # 查找剂量
            for pattern in self.compiled_patterns["dosage"]:
                match = pattern.search(context)
                if match:
                    attributes["dosage"] = match.group(0)
                    break
            
            # 查找频率
            for pattern in self.compiled_patterns["frequency"]:
                match = pattern.search(context)
                if match:
                    attributes["frequency"] = match.group(0)
                    break
            
            # 查找给药途径
            for pattern in self.compiled_patterns["route"]:
                match = pattern.search(context)
                if match:
                    attributes["route"] = match.group(0)
                    break
        
        # 对于症状，查找严重程度和持续时间
        elif entity_type == "symptom":
            entity_pos = text.find(entity_text)
            context_start = max(0, entity_pos - 30)
            context_end = min(len(text), entity_pos + len(entity_text) + 30)
            context = text[context_start:context_end]
            
            # 查找严重程度
            for pattern in self.compiled_patterns["severity"]:
                match = pattern.search(context)
                if match:
                    attributes["severity"] = match.group(0)
                    break
            
            # 查找持续时间
            for pattern in self.compiled_patterns["time"]:
                match = pattern.search(context)
                if match:
                    attributes["duration"] = match.group(0)
                    break
            
            # 查找部位
            for pattern in self.compiled_patterns["body_part"]:
                match = pattern.search(context)
                if match:
                    attributes["body_part"] = match.group(0)
                    break
        
        return attributes
    
    def _deduplicate_entities(self, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """去除重复实体"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.text, entity.type, entity.source_offset)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _associate_medication_dosage(self, text: str, 
                                      entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """关联药物和剂量信息"""
        medications = [e for e in entities if e.type == "medication"]
        dosages = [e for e in entities if e.type == "dosage"]
        frequencies = [e for e in entities if e.type == "frequency"]
        routes = [e for e in entities if e.type == "route"]
        
        for med in medications:
            med_pos = med.source_offset[0] if med.source_offset else 0
            
            # 查找最近的剂量
            nearest_dosage = None
            min_distance = float('inf')
            for dosage in dosages:
                dosage_pos = dosage.source_offset[0] if dosage.source_offset else 0
                distance = abs(dosage_pos - med_pos)
                if distance < min_distance and distance < 50:
                    min_distance = distance
                    nearest_dosage = dosage
            
            if nearest_dosage and "dosage" not in med.attributes:
                med.attributes["dosage"] = nearest_dosage.text
            
            # 类似处理频率和途径
            for freq in frequencies:
                freq_pos = freq.source_offset[0] if freq.source_offset else 0
                if abs(freq_pos - med_pos) < 50 and "frequency" not in med.attributes:
                    med.attributes["frequency"] = freq.text
                    break
            
            for route in routes:
                route_pos = route.source_offset[0] if route.source_offset else 0
                if abs(route_pos - med_pos) < 50 and "route" not in med.attributes:
                    med.attributes["route"] = route.text
                    break
        
        return entities
    
    def extract_with_llm(self, text: str, speaker_role: str = None) -> List[ClinicalEntity]:
        """
        使用 LLM 增强的实体抽取
        
        Args:
            text: 对话文本
            speaker_role: 说话人角色
            
        Returns:
            实体列表
        """
        if not self.llm:
            logger.warning("[临床实体抽取] LLM 不可用，使用规则抽取")
            return self.extract_entities(text, speaker_role)
        
        # 先用规则抽取
        rule_entities = self.extract_entities(text, speaker_role)
        
        # 构建 LLM 提示
        prompt = """请从以下医患对话中提取临床实体，按 JSON 格式返回。

实体类型：
- symptom: 症状
- disease: 疾病  
- medication: 药物（包含剂量、频率、途径）
- procedure: 检查/治疗

对话文本：
{text}

请返回 JSON 数组，格式：
[{{"text": "实体文本", "type": "实体类型", "attributes": {{"属性名": "属性值"}}}}]

只返回 JSON，不要其他内容。"""
        
        try:
            response = self.llm.chat(
                query=prompt.format(text=text),
                session_id="entity_extraction",
                use_rag=False
            )
            
            response_text = response.get("response", "[]")
            
            # 尝试解析 JSON
            import json
            llm_entities_data = json.loads(response_text)
            
            # 转换为 ClinicalEntity 对象
            for entity_data in llm_entities_data:
                entity = ClinicalEntity(
                    text=entity_data.get("text", ""),
                    type=entity_data.get("type", "unknown"),
                    attributes=entity_data.get("attributes", {}),
                    confidence=0.8,  # LLM 抽取的置信度
                    speaker_role=speaker_role
                )
                rule_entities.append(entity)
            
        except Exception as e:
            logger.warning(f"[临床实体抽取] LLM 抽取失败: {e}")
        
        # 去重
        return self._deduplicate_entities(rule_entities)
    
    def link_to_ontology(self, entity: ClinicalEntity) -> Optional[str]:
        """
        将实体链接到医学本体
        
        Args:
            entity: 临床实体
            
        Returns:
            本体编码（如 SNOMED CT 代码）
        """
        if not self.kg:
            return None
        
        # 尝试在知识图谱中查找
        try:
            # 使用知识图谱的智能查询
            result = self.kg.smart_query(f"什么是{entity.text}")
            if result and result.get("results"):
                # 提取实体 ID 或代码
                for r in result["results"]:
                    if "id" in r:
                        return r["id"]
        except Exception as e:
            logger.debug(f"实体链接失败: {e}")
        
        return None
    
    def get_entity_summary(self, entities: List[ClinicalEntity]) -> Dict:
        """
        获取实体统计摘要
        
        Returns:
            按类型分组的实体统计
        """
        summary = {t: [] for t in self.ENTITY_TYPES}
        
        for entity in entities:
            if entity.type in summary:
                summary[entity.type].append(entity.to_dict())
        
        return {k: v for k, v in summary.items() if v}


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    extractor = ClinicalEntityExtractor()
    
    test_texts = [
        "医生您好，我最近胸口有点闷痛，已经两天了，特别是晚上睡觉的时候。",
        "我之前吃的阿司匹林每天一片，现在是不是要加量？",
        "患者有高血压病史五年，目前服用硝苯地平缓释片20mg每日两次。",
        "做了个CT，肺部没什么问题，但是心电图显示有些异常。",
        "剧烈头痛伴有呕吐，持续了三个小时，越来越严重。"
    ]
    
    print("=== 临床实体抽取测试 ===\n")
    
    for text in test_texts:
        print(f"文本: {text}")
        entities = extractor.extract_entities(text, speaker_role="patient")
        
        if entities:
            for entity in entities:
                attrs_str = ", ".join(f"{k}={v}" for k, v in entity.attributes.items()) if entity.attributes else ""
                print(f"  [{entity.type:12s}] {entity.text} {f'({attrs_str})' if attrs_str else ''}")
        else:
            print("  (未提取到实体)")
        print()
