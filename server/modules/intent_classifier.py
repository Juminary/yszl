"""
医学问答意图分类模块
基于规则和关键词模板进行意图识别
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    医学问答意图分类器
    根据用户问题和提取的实体，判断用户的查询意图
    """
    
    # 支持的意图类型
    INTENTS = [
        'disease_symptom',      # 查疾病的症状：感冒有什么症状
        'symptom_disease',      # 根据症状查疾病：头疼是什么病
        'disease_cause',        # 查疾病原因：感冒是怎么引起的
        'disease_prevent',      # 查疾病预防：怎么预防感冒
        'disease_cureway',      # 查治疗方式：感冒怎么治疗
        'disease_lasttime',     # 查治疗周期：感冒多久能好
        'disease_cureprob',     # 查治愈概率：感冒能治好吗
        'disease_easyget',      # 查易感人群：什么人容易感冒
        'disease_desc',         # 查疾病介绍：什么是感冒
        'disease_drug',         # 查疾病用药：感冒吃什么药
        'drug_disease',         # 查药物适应症：布洛芬治什么病
        'disease_food',         # 查饮食建议：感冒吃什么好
        'disease_not_food',     # 查饮食禁忌：感冒不能吃什么
        'disease_check',        # 查检查项目：感冒要做什么检查
        'check_disease',        # 查检查对应疾病：血常规能查什么病
        'disease_department',   # 查就诊科室：感冒挂什么科
        'disease_acompany',     # 查并发症：感冒有什么并发症
        'general_chat',         # 闲聊/无法识别
    ]
    
    def __init__(self):
        """初始化意图分类器"""
        # 意图关键词模板
        self._init_patterns()
        logger.info("IntentClassifier initialized")
    
    def _init_patterns(self):
        """初始化意图匹配模式"""
        
        # 症状相关
        self.symptom_patterns = [
            r'有什么症状', r'症状是什么', r'什么症状', r'会怎样',
            r'有哪些表现', r'表现是什么', r'什么表现', r'临床表现',
            r'有什么反应', r'身体会怎么样'
        ]
        
        # 病因相关
        self.cause_patterns = [
            r'怎么引起', r'什么原因', r'为什么会', r'是怎么得的',
            r'怎么得的', r'病因是什么', r'什么导致', r'怎么患上',
            r'是什么引起', r'发病原因'
        ]
        
        # 预防相关
        self.prevent_patterns = [
            r'怎么预防', r'如何预防', r'怎样预防', r'预防措施',
            r'怎么避免', r'如何避免', r'注意什么', r'注意事项'
        ]
        
        # 治疗方式相关
        self.cureway_patterns = [
            r'怎么治疗', r'如何治疗', r'怎样治疗', r'治疗方法',
            r'怎么治', r'如何治', r'怎么办', r'该怎么办',
            r'治疗方式', r'怎么医治', r'如何医治'
        ]
        
        # 治疗周期相关
        self.lasttime_patterns = [
            r'多久能好', r'多长时间', r'要多久', r'几天能好',
            r'治疗周期', r'恢复时间', r'多久恢复', r'几天恢复'
        ]
        
        # 治愈概率相关
        self.cureprob_patterns = [
            r'能治好吗', r'能治愈吗', r'可以治好', r'治愈率',
            r'能不能治好', r'有救吗', r'严重吗', r'好治吗'
        ]
        
        # 易感人群相关
        self.easyget_patterns = [
            r'什么人容易', r'哪些人容易', r'易感人群', r'高发人群',
            r'谁容易得', r'容易得的人'
        ]
        
        # 疾病介绍相关
        self.desc_patterns = [
            r'什么是', r'是什么病', r'是什么意思', r'介绍一下',
            r'了解一下', r'是啥', r'是什么东西'
        ]
        
        # 用药相关
        self.drug_patterns = [
            r'吃什么药', r'用什么药', r'什么药', r'开什么药',
            r'药物治疗', r'吃啥药', r'用啥药', r'哪些药'
        ]
        
        # 药物适应症相关
        self.drug_disease_patterns = [
            r'治什么病', r'治疗什么', r'能治什么', r'主治什么',
            r'适应症', r'用于什么', r'可以治疗什么'
        ]
        
        # 饮食建议相关
        self.food_patterns = [
            r'吃什么好', r'吃什么食物', r'饮食', r'食疗',
            r'吃啥好', r'宜吃什么', r'适合吃什么'
        ]
        
        # 饮食禁忌相关
        self.not_food_patterns = [
            r'不能吃什么', r'忌口', r'忌什么', r'不宜吃',
            r'不能吃啥', r'禁忌食物', r'不要吃什么'
        ]
        
        # 检查项目相关
        self.check_patterns = [
            r'做什么检查', r'检查什么', r'查什么', r'怎么检查',
            r'需要检查', r'要做什么检查', r'检查项目'
        ]
        
        # 检查对应疾病
        self.check_disease_patterns = [
            r'能查出什么', r'检查什么病', r'查什么病', r'能发现什么'
        ]
        
        # 科室相关
        self.department_patterns = [
            r'挂什么科', r'看什么科', r'去哪个科', r'属于什么科',
            r'哪个科室', r'应该挂', r'要挂什么'
        ]
        
        # 并发症相关
        self.acompany_patterns = [
            r'并发症', r'会引起什么', r'会导致什么', r'有什么并发',
            r'会引发什么', r'伴随什么'
        ]
    
    def _match_patterns(self, text: str, patterns: List[str]) -> bool:
        """检查文本是否匹配任一模式"""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def classify(self, query: str, entities: Dict[str, List[str]] = None) -> Tuple[str, float]:
        """
        分类用户意图
        
        Args:
            query: 用户问题
            entities: 提取的实体 {'disease': [...], 'symptom': [...], ...}
            
        Returns:
            (意图类型, 置信度)
        """
        if not query:
            return 'general_chat', 0.0
        
        if entities is None:
            entities = {}
        
        query = query.lower()
        
        # 获取实体信息
        has_disease = bool(entities.get('disease'))
        has_symptom = bool(entities.get('symptom'))
        has_drug = bool(entities.get('drug'))
        has_check = bool(entities.get('check'))
        has_food = bool(entities.get('food'))
        has_department = bool(entities.get('department'))
        
        # 按优先级匹配意图
        
        # 1. 症状 -> 疾病（根据症状问是什么病）
        if has_symptom and not has_disease:
            if self._match_patterns(query, self.desc_patterns + ['是什么病', '什么病', '得了什么']):
                return 'symptom_disease', 0.9
        
        # 2. 疾病相关查询
        if has_disease:
            # 查症状
            if self._match_patterns(query, self.symptom_patterns):
                return 'disease_symptom', 0.9
            
            # 查病因
            if self._match_patterns(query, self.cause_patterns):
                return 'disease_cause', 0.9
            
            # 查预防
            if self._match_patterns(query, self.prevent_patterns):
                return 'disease_prevent', 0.9
            
            # 查治疗方式
            if self._match_patterns(query, self.cureway_patterns):
                return 'disease_cureway', 0.9
            
            # 查治疗周期
            if self._match_patterns(query, self.lasttime_patterns):
                return 'disease_lasttime', 0.9
            
            # 查治愈概率
            if self._match_patterns(query, self.cureprob_patterns):
                return 'disease_cureprob', 0.85
            
            # 查易感人群
            if self._match_patterns(query, self.easyget_patterns):
                return 'disease_easyget', 0.9
            
            # 查用药
            if self._match_patterns(query, self.drug_patterns):
                return 'disease_drug', 0.9
            
            # 查饮食禁忌
            if self._match_patterns(query, self.not_food_patterns):
                return 'disease_not_food', 0.9
            
            # 查饮食建议
            if self._match_patterns(query, self.food_patterns):
                return 'disease_food', 0.9
            
            # 查检查项目
            if self._match_patterns(query, self.check_patterns):
                return 'disease_check', 0.9
            
            # 查科室
            if self._match_patterns(query, self.department_patterns):
                return 'disease_department', 0.9
            
            # 查并发症
            if self._match_patterns(query, self.acompany_patterns):
                return 'disease_acompany', 0.9
            
            # 查疾病介绍（默认）
            if self._match_patterns(query, self.desc_patterns):
                return 'disease_desc', 0.85
        
        # 3. 药物相关查询
        if has_drug:
            if self._match_patterns(query, self.drug_disease_patterns):
                return 'drug_disease', 0.9
        
        # 4. 检查相关查询
        if has_check:
            if self._match_patterns(query, self.check_disease_patterns):
                return 'check_disease', 0.9
        
        # 5. 纯症状描述（可能在问是什么病）
        if has_symptom:
            # 如果只有症状，默认认为是在问可能是什么病
            return 'symptom_disease', 0.7
        
        # 6. 无法识别
        return 'general_chat', 0.3
    
    def classify_with_details(self, query: str, entities: Dict[str, List[str]] = None) -> Dict:
        """
        分类并返回详细信息
        
        Returns:
            {
                'intent': 意图类型,
                'confidence': 置信度,
                'entities': 实体,
                'query': 原始问题
            }
        """
        intent, confidence = self.classify(query, entities)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities or {},
            'query': query,
            'intent_description': self.get_intent_description(intent)
        }
    
    def get_intent_description(self, intent: str) -> str:
        """获取意图的中文描述"""
        descriptions = {
            'disease_symptom': '查询疾病症状',
            'symptom_disease': '根据症状查疾病',
            'disease_cause': '查询疾病原因',
            'disease_prevent': '查询预防措施',
            'disease_cureway': '查询治疗方式',
            'disease_lasttime': '查询治疗周期',
            'disease_cureprob': '查询治愈概率',
            'disease_easyget': '查询易感人群',
            'disease_desc': '查询疾病介绍',
            'disease_drug': '查询疾病用药',
            'drug_disease': '查询药物适应症',
            'disease_food': '查询饮食建议',
            'disease_not_food': '查询饮食禁忌',
            'disease_check': '查询检查项目',
            'check_disease': '查询检查对应疾病',
            'disease_department': '查询就诊科室',
            'disease_acompany': '查询并发症',
            'general_chat': '闲聊/其他',
        }
        return descriptions.get(intent, '未知意图')


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    classifier = IntentClassifier()
    
    # 测试用例
    test_cases = [
        ("感冒有什么症状", {'disease': ['感冒']}),
        ("头疼发烧是什么病", {'symptom': ['头疼', '发烧']}),
        ("高血压怎么治疗", {'disease': ['高血压']}),
        ("糖尿病吃什么药", {'disease': ['糖尿病']}),
        ("感冒不能吃什么", {'disease': ['感冒']}),
        ("布洛芬治什么病", {'drug': ['布洛芬']}),
        ("感冒挂什么科", {'disease': ['感冒']}),
        ("怎么预防感冒", {'disease': ['感冒']}),
        ("什么是高血压", {'disease': ['高血压']}),
        ("感冒多久能好", {'disease': ['感冒']}),
    ]
    
    for query, entities in test_cases:
        result = classifier.classify_with_details(query, entities)
        print(f"\n问题: {query}")
        print(f"意图: {result['intent']} ({result['intent_description']})")
        print(f"置信度: {result['confidence']}")

