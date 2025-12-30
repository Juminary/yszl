"""
用药查询与管理模块
提供药品信息查询、用药建议、药物相互作用检查
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class MedicationModule:
    """用药查询与管理模块"""
    
    def __init__(self, knowledge_path: str = None):
        """
        初始化用药模块
        
        Args:
            knowledge_path: 医疗知识库路径
        """
        self.knowledge_path = knowledge_path or Path(__file__).parent.parent.parent / "config" / "medical_knowledge.json"
        self.medications = {}
        self.interactions = {}
        self.contraindications = {}
        
        self._load_knowledge()
        logger.info("MedicationModule initialized")
    
    def _load_knowledge(self):
        """加载药品知识库"""
        try:
            if Path(self.knowledge_path).exists():
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.medications = {m['id']: m for m in data.get('medications', [])}
                    self.interactions = data.get('drug_interactions', {})
                    logger.info(f"Loaded {len(self.medications)} medications")
            else:
                logger.warning(f"Knowledge base not found: {self.knowledge_path}")
                self._create_default_medications()
        except Exception as e:
            logger.error(f"Failed to load medication knowledge: {e}")
            self._create_default_medications()
    
    def _create_default_medications(self):
        """创建默认药品数据"""
        self.medications = {
            'amoxicillin': {
                'id': 'amoxicillin',
                'name': '阿莫西林',
                'generic_name': '阿莫西林胶囊',
                'category': '抗生素',
                'indications': ['呼吸道感染', '泌尿道感染', '皮肤软组织感染'],
                'dosage': {
                    'adult': '0.5g，每8小时一次',
                    'child': '根据体重计算，20-40mg/kg/日，分3次服用'
                },
                'route': '口服',
                'contraindications': ['青霉素过敏者禁用'],
                'side_effects': ['恶心', '呕吐', '腹泻', '皮疹', '过敏反应'],
                'precautions': [
                    '过敏体质者慎用',
                    '肝肾功能不全者应调整剂量',
                    '孕妇和哺乳期妇女慎用'
                ],
                'storage': '密封，阴凉干燥处保存'
            },
            'ibuprofen': {
                'id': 'ibuprofen',
                'name': '布洛芬',
                'generic_name': '布洛芬片',
                'category': '解热镇痛药',
                'indications': ['发热', '头痛', '关节痛', '牙痛', '肌肉痛', '痛经'],
                'dosage': {
                    'adult': '0.2-0.4g，每4-6小时一次，一日不超过2.4g',
                    'child': '根据体重，5-10mg/kg，每6-8小时一次'
                },
                'route': '口服',
                'contraindications': [
                    '消化性溃疡患者禁用',
                    '严重肝肾功能不全者禁用',
                    '孕妇后期禁用'
                ],
                'side_effects': ['胃肠道不适', '恶心', '呕吐', '头晕', '皮疹'],
                'precautions': [
                    '饭后服用以减少胃肠刺激',
                    '不宜长期大量使用',
                    '老年人慎用'
                ],
                'storage': '遮光，密封保存'
            },
            'metformin': {
                'id': 'metformin',
                'name': '二甲双胍',
                'generic_name': '盐酸二甲双胍片',
                'category': '降糖药',
                'indications': ['2型糖尿病'],
                'dosage': {
                    'adult': '起始剂量0.5g，每日2-3次，餐中或餐后服用，最大剂量2g/日',
                    'child': '10岁以上儿童，起始剂量0.5g，每日一次'
                },
                'route': '口服',
                'contraindications': [
                    '1型糖尿病',
                    '糖尿病酮症酸中毒',
                    '严重肝肾功能不全',
                    '急性心肌梗死',
                    '严重感染'
                ],
                'side_effects': ['胃肠道反应（腹泻、恶心、呕吐）', '乳酸酸中毒（罕见）'],
                'precautions': [
                    '定期监测肾功能',
                    '避免饮酒',
                    '进行影像学检查使用碘造影剂前应停药'
                ],
                'storage': '密封保存'
            },
            'amlodipine': {
                'id': 'amlodipine',
                'name': '氨氯地平',
                'generic_name': '苯磺酸氨氯地平片',
                'category': '降压药（钙通道阻滞剂）',
                'indications': ['高血压', '心绞痛'],
                'dosage': {
                    'adult': '5mg，每日一次，根据病情可增至10mg/日',
                    'child': '6岁以上儿童，2.5-5mg，每日一次'
                },
                'route': '口服',
                'contraindications': ['严重低血压', '主动脉瓣狭窄'],
                'side_effects': ['头痛', '水肿', '面部潮红', '心悸', '头晕'],
                'precautions': [
                    '肝功能不全者慎用并减量',
                    '老年人起始剂量应减半',
                    '不能突然停药'
                ],
                'storage': '密封，30℃以下保存'
            },
            'omeprazole': {
                'id': 'omeprazole',
                'name': '奥美拉唑',
                'generic_name': '奥美拉唑肠溶胶囊',
                'category': '质子泵抑制剂',
                'indications': ['胃溃疡', '十二指肠溃疡', '反流性食管炎', '卓-艾综合征'],
                'dosage': {
                    'adult': '20mg，每日一次，早餐前服用',
                    'child': '体重≥20kg，10-20mg/日'
                },
                'route': '口服',
                'contraindications': ['对本品过敏者'],
                'side_effects': ['头痛', '腹泻', '便秘', '恶心', '腹痛'],
                'precautions': [
                    '应整粒吞服，不能咀嚼或压碎',
                    '长期服用应监测肝功能',
                    '老年人无需调整剂量'
                ],
                'storage': '遮光，密封保存'
            }
        }
        
        # 药物相互作用数据
        self.interactions = {
            'metformin_ibuprofen': {
                'drugs': ['二甲双胍', '布洛芬'],
                'severity': 'moderate',
                'description': '布洛芬等NSAIDs可能影响肾功能，降低二甲双胍清除率',
                'recommendation': '合用时应监测肾功能，注意乳酸酸中毒风险'
            },
            'amlodipine_omeprazole': {
                'drugs': ['氨氯地平', '奥美拉唑'],
                'severity': 'low',
                'description': '奥美拉唑可能轻微影响氨氯地平的代谢',
                'recommendation': '一般无需调整剂量，注意观察降压效果'
            }
        }
    
    def query_medication(self, med_name: str) -> Optional[Dict]:
        """
        查询药品信息
        
        Args:
            med_name: 药品名称
        
        Returns:
            药品信息
        """
        try:
            # 精确匹配
            for med_id, med in self.medications.items():
                if med['name'] == med_name or med.get('generic_name') == med_name:
                    return med
            
            # 模糊匹配
            for med_id, med in self.medications.items():
                if med_name in med['name'] or med_name in med.get('generic_name', ''):
                    return med
            
            return None
            
        except Exception as e:
            logger.error(f"Medication query failed: {e}")
            return None
    
    def check_interactions(self, medications: List[str]) -> List[Dict]:
        """
        检查药物相互作用
        
        Args:
            medications: 药品名称列表
        
        Returns:
            相互作用信息列表
        """
        try:
            warnings = []
            
            # 获取药品信息
            med_objects = []
            for med_name in medications:
                med = self.query_medication(med_name)
                if med:
                    med_objects.append(med)
            
            # 检查两两相互作用
            for i, med1 in enumerate(med_objects):
                for med2 in med_objects[i+1:]:
                    interaction = self._get_interaction(med1['name'], med2['name'])
                    if interaction:
                        warnings.append(interaction)
            
            # 检查同类药物重复
            categories = defaultdict(list)
            for med in med_objects:
                categories[med['category']].append(med['name'])
            
            for category, meds in categories.items():
                if len(meds) > 1:
                    warnings.append({
                        'type': 'duplication',
                        'drugs': meds,
                        'severity': 'moderate',
                        'description': f'同时使用多个{category}，可能存在重复用药',
                        'recommendation': '建议评估是否需要调整用药方案'
                    })
            
            return warnings
            
        except Exception as e:
            logger.error(f"Interaction check failed: {e}")
            return []
    
    def _get_interaction(self, med1: str, med2: str) -> Optional[Dict]:
        """获取两个药物之间的相互作用"""
        # 检查预定义的相互作用
        for interaction_id, interaction in self.interactions.items():
            drugs = interaction['drugs']
            if (med1 in drugs and med2 in drugs) or (med2 in drugs and med1 in drugs):
                return {
                    'type': 'interaction',
                    **interaction
                }
        return None
    
    def get_dosage_recommendation(self, med_name: str, patient_info: Dict) -> Dict:
        """
        获取用药剂量建议
        
        Args:
            med_name: 药品名称
            patient_info: 患者信息（年龄、体重、肝肾功能等）
        
        Returns:
            剂量建议
        """
        try:
            med = self.query_medication(med_name)
            if not med:
                return {
                    'status': 'not_found',
                    'message': f'未找到药品：{med_name}'
                }
            
            age = patient_info.get('age', 0)
            weight = patient_info.get('weight')
            renal_function = patient_info.get('renal_function', 'normal')  # normal/mild/moderate/severe
            hepatic_function = patient_info.get('hepatic_function', 'normal')
            
            # 基础剂量
            if age < 18:
                dosage = med['dosage'].get('child', '请咨询医生')
                note = '儿童用药，需要根据体重计算'
            else:
                dosage = med['dosage'].get('adult', '请咨询医生')
                note = '成人常规剂量'
            
            # 特殊人群调整
            adjustments = []
            
            # 肾功能不全
            if renal_function in ['moderate', 'severe']:
                if '二甲双胍' in med['name']:
                    adjustments.append('⚠️ 严重肾功能不全禁用')
                else:
                    adjustments.append('建议减量或延长给药间隔')
            
            # 肝功能不全
            if hepatic_function in ['moderate', 'severe']:
                adjustments.append('肝功能不全者需要减量')
            
            # 老年人
            if age >= 65:
                if '氨氯地平' in med['name']:
                    adjustments.append('老年人起始剂量应减半')
                else:
                    adjustments.append('老年人用药需谨慎，建议从小剂量开始')
            
            return {
                'status': 'success',
                'medication': med['name'],
                'dosage': dosage,
                'route': med['route'],
                'note': note,
                'adjustments': adjustments,
                'contraindications': med['contraindications'],
                'precautions': med['precautions']
            }
            
        except Exception as e:
            logger.error(f"Dosage recommendation failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def check_contraindications(self, med_name: str, patient_info: Dict) -> Dict:
        """
        检查禁忌症
        
        Args:
            med_name: 药品名称
            patient_info: 患者信息（病史、过敏史等）
        
        Returns:
            禁忌症检查结果
        """
        try:
            med = self.query_medication(med_name)
            if not med:
                return {'status': 'not_found'}
            
            warnings = []
            
            # 检查过敏史
            allergies = patient_info.get('allergies', [])
            for allergy in allergies:
                if allergy in med['name'] or any(allergy in c for c in med.get('contraindications', [])):
                    warnings.append({
                        'severity': 'high',
                        'type': 'allergy',
                        'message': f'患者对{allergy}过敏，可能对{med["name"]}过敏'
                    })
            
            # 检查疾病禁忌
            diseases = patient_info.get('diseases', [])
            for disease in diseases:
                for contraindication in med.get('contraindications', []):
                    if disease in contraindication:
                        warnings.append({
                            'severity': 'high',
                            'type': 'disease',
                            'message': f'{disease}患者禁用{med["name"]}'
                        })
            
            # 检查特殊状态
            if patient_info.get('pregnant'):
                if '孕妇' in ''.join(med.get('contraindications', [])):
                    warnings.append({
                        'severity': 'high',
                        'type': 'pregnancy',
                        'message': f'孕妇禁用或慎用{med["name"]}'
                    })
            
            if patient_info.get('breastfeeding'):
                if '哺乳' in ''.join(med.get('precautions', [])):
                    warnings.append({
                        'severity': 'medium',
                        'type': 'breastfeeding',
                        'message': f'哺乳期妇女使用{med["name"]}需谨慎'
                    })
            
            return {
                'status': 'success',
                'medication': med['name'],
                'warnings': warnings,
                'safe': len(warnings) == 0
            }
            
        except Exception as e:
            logger.error(f"Contraindication check failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def search_by_indication(self, indication: str) -> List[Dict]:
        """
        根据适应症搜索药品
        
        Args:
            indication: 适应症
        
        Returns:
            符合的药品列表
        """
        results = []
        
        for med in self.medications.values():
            if any(indication in ind for ind in med.get('indications', [])):
                results.append({
                    'name': med['name'],
                    'category': med['category'],
                    'indications': med['indications']
                })
        
        return results
    
    def list_medications_by_category(self, category: str = None) -> List[Dict]:
        """
        按类别列出药品
        
        Args:
            category: 药品类别，None表示所有类别
        
        Returns:
            药品列表
        """
        results = []
        
        for med in self.medications.values():
            if category is None or med['category'] == category:
                results.append({
                    'name': med['name'],
                    'category': med['category'],
                    'indications': med['indications']
                })
        
        return results


if __name__ == "__main__":
    # 测试代码
    med_module = MedicationModule()
    
    # 测试查询药品
    print("=" * 50)
    print("测试1: 查询阿莫西林")
    result = med_module.query_medication('阿莫西林')
    if result:
        print(f"药品: {result['name']}")
        print(f"类别: {result['category']}")
        print(f"适应症: {', '.join(result['indications'])}")
    
    # 测试药物相互作用
    print("\n" + "=" * 50)
    print("测试2: 检查药物相互作用")
    interactions = med_module.check_interactions(['二甲双胍', '布洛芬'])
    for interaction in interactions:
        print(f"⚠️ {interaction.get('description', '')}")
    
    # 测试剂量建议
    print("\n" + "=" * 50)
    print("测试3: 获取剂量建议")
    patient = {'age': 70, 'weight': 60, 'renal_function': 'normal'}
    dosage = med_module.get_dosage_recommendation('氨氯地平', patient)
    print(f"剂量: {dosage.get('dosage')}")
    print(f"调整: {dosage.get('adjustments')}")
