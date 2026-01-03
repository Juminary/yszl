"""
医学词典模块
使用 AC 自动机进行快速医学实体识别
"""

import os
import logging
from typing import Dict, List, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入 ahocorasick
try:
    import ahocorasick
    AC_AVAILABLE = True
except ImportError:
    AC_AVAILABLE = False
    logger.warning("ahocorasick not available. Install with: pip install pyahocorasick")


class MedicalDictionary:
    """
    医学词典模块
    使用 AC 自动机进行高效的多模式匹配
    """
    
    # 实体类型
    ENTITY_TYPES = ['disease', 'symptom', 'drug', 'check', 'food', 'department', 'producer']
    
    def __init__(self, dict_dir: str = None):
        """
        初始化医学词典
        
        Args:
            dict_dir: 词典目录路径
        """
        if dict_dir is None:
            # 默认路径：server/data/dict
            # 使用 resolve() 获取绝对路径，确保在不同启动方式下都能找到
            self.dict_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'dict'
        else:
            self.dict_dir = Path(dict_dir)
        
        print(f"📊 [词典加载] 正在从目录读取: {self.dict_dir}", flush=True)
        
        # 各类实体词典
        self.diseases: Set[str] = set()
        self.symptoms: Set[str] = set()
        self.drugs: Set[str] = set()
        self.checks: Set[str] = set()
        self.foods: Set[str] = set()
        self.departments: Set[str] = set()
        self.producers: Set[str] = set()
        
        # AC 自动机
        self.ac = None
        
        # 加载词典
        self._load_all_dicts()
        
        # 构建 AC 自动机
        if AC_AVAILABLE:
            self._build_automaton()
        
        logger.info(f"MedicalDictionary initialized: {self.get_stats()}")
    
    def _load_dict(self, filename: str) -> Set[str]:
        """加载单个词典文件"""
        filepath = self.dict_dir / filename
        words = set()
        
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        # 过滤掉太短或无效的词
                        if word and len(word) >= 2:
                            words.add(word)
                logger.info(f"Loaded {len(words)} words from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
        else:
            logger.warning(f"Dictionary file not found: {filepath}")
        
        return words
    
    def _load_all_dicts(self):
        """加载所有词典"""
        self.diseases = self._load_dict('disease.txt')
        self.symptoms = self._load_dict('symptom.txt')
        self.drugs = self._load_dict('drug.txt')
        self.checks = self._load_dict('check.txt')
        self.foods = self._load_dict('food.txt')
        self.departments = self._load_dict('department.txt')
        self.producers = self._load_dict('producer.txt')
    
    def _build_automaton(self):
        """构建 AC 自动机"""
        if not AC_AVAILABLE:
            return
        
        self.ac = ahocorasick.Automaton()
        
        # 添加所有词到自动机
        entity_map = {
            'disease': self.diseases,
            'symptom': self.symptoms,
            'drug': self.drugs,
            'check': self.checks,
            'food': self.foods,
            'department': self.departments,
            'producer': self.producers
        }
        
        word_count = 0
        for entity_type, words in entity_map.items():
            for word in words:
                # 存储 (实体类型, 实体词)
                self.ac.add_word(word, (entity_type, word))
                word_count += 1
        
        self.ac.make_automaton()
        logger.info(f"AC automaton built with {word_count} patterns")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        从文本中提取医学实体
        
        Args:
            text: 输入文本
            
        Returns:
            {
                'disease': ['感冒', '发烧'],
                'symptom': ['头疼', '咳嗽'],
                'drug': ['布洛芬'],
                ...
            }
        """
        entities = {
            'disease': [],
            'symptom': [],
            'drug': [],
            'check': [],
            'food': [],
            'department': [],
            'producer': []
        }
        
        if not text:
            return entities
        
        if AC_AVAILABLE and self.ac:
            # 使用 AC 自动机快速匹配
            found = set()  # 用于去重
            for end_idx, (entity_type, word) in self.ac.iter(text):
                if word not in found:
                    entities[entity_type].append(word)
                    found.add(word)
        else:
            # 降级方案：简单的字符串匹配
            entities = self._extract_entities_simple(text)
        
        return entities
    
    def _extract_entities_simple(self, text: str) -> Dict[str, List[str]]:
        """简单的实体提取（无 AC 自动机时的降级方案）"""
        entities = {
            'disease': [],
            'symptom': [],
            'drug': [],
            'check': [],
            'food': [],
            'department': [],
            'producer': []
        }
        
        # 按词典长度降序排列，优先匹配长词
        for word in sorted(self.diseases, key=len, reverse=True):
            if word in text:
                entities['disease'].append(word)
        
        for word in sorted(self.symptoms, key=len, reverse=True):
            if word in text:
                entities['symptom'].append(word)
        
        for word in sorted(self.drugs, key=len, reverse=True):
            if word in text:
                entities['drug'].append(word)
        
        for word in sorted(self.checks, key=len, reverse=True):
            if word in text:
                entities['check'].append(word)
        
        for word in sorted(self.departments, key=len, reverse=True):
            if word in text:
                entities['department'].append(word)
        
        return entities
    
    def extract_entities_with_positions(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        提取实体并返回位置信息
        
        Returns:
            [(start, end, entity_type, word), ...]
        """
        results = []
        
        if AC_AVAILABLE and self.ac:
            for end_idx, (entity_type, word) in self.ac.iter(text):
                start_idx = end_idx - len(word) + 1
                results.append((start_idx, end_idx + 1, entity_type, word))
        
        # 按位置排序
        results.sort(key=lambda x: x[0])
        return results
    
    def is_medical_entity(self, word: str) -> Tuple[bool, str]:
        """
        检查词是否为医学实体
        
        Returns:
            (是否为实体, 实体类型)
        """
        if word in self.diseases:
            return True, 'disease'
        if word in self.symptoms:
            return True, 'symptom'
        if word in self.drugs:
            return True, 'drug'
        if word in self.checks:
            return True, 'check'
        if word in self.foods:
            return True, 'food'
        if word in self.departments:
            return True, 'department'
        if word in self.producers:
            return True, 'producer'
        return False, ''
    
    def get_stats(self) -> Dict[str, int]:
        """获取词典统计信息"""
        return {
            'disease': len(self.diseases),
            'symptom': len(self.symptoms),
            'drug': len(self.drugs),
            'check': len(self.checks),
            'food': len(self.foods),
            'department': len(self.departments),
            'producer': len(self.producers),
            'total': sum([
                len(self.diseases), len(self.symptoms), len(self.drugs),
                len(self.checks), len(self.foods), len(self.departments),
                len(self.producers)
            ])
        }
    
    def add_word(self, word: str, entity_type: str) -> bool:
        """
        动态添加词到词典
        
        Args:
            word: 要添加的词
            entity_type: 实体类型
        """
        if entity_type not in self.ENTITY_TYPES:
            return False
        
        # 添加到对应词典
        getattr(self, f"{entity_type}s" if entity_type != 'disease' else 'diseases').add(word)
        
        # 重建 AC 自动机
        if AC_AVAILABLE:
            self._build_automaton()
        
        return True


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    med_dict = MedicalDictionary()
    print("词典统计:", med_dict.get_stats())
    
    # 测试实体提取
    test_texts = [
        "我最近头疼，还有点发烧，是不是感冒了？",
        "医生说我得了高血压，需要吃降压药",
        "我想挂内科的号",
        "布洛芬可以治疗什么病？"
    ]
    
    for text in test_texts:
        print(f"\n输入: {text}")
        entities = med_dict.extract_entities(text)
        print(f"实体: {entities}")

