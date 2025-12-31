"""
Cypher 查询生成器
根据意图和实体动态生成 Neo4j Cypher 查询语句
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CypherGenerator:
    """
    Cypher 查询生成器
    根据用户意图和提取的实体，生成对应的 Cypher 查询语句
    """
    
    def __init__(self, num_limit: int = 10):
        """
        初始化查询生成器
        
        Args:
            num_limit: 查询结果数量限制
        """
        self.num_limit = num_limit
        logger.info("CypherGenerator initialized")
    
    def generate(self, intent: str, entities: Dict[str, List[str]]) -> List[str]:
        """
        根据意图和实体生成 Cypher 查询
        
        Args:
            intent: 问题意图
            entities: 提取的实体 {'disease': [...], 'symptom': [...], ...}
            
        Returns:
            Cypher 查询语句列表
        """
        queries = []
        
        # 根据意图调用对应的生成方法
        generator_map = {
            'disease_symptom': self._gen_disease_symptom,
            'symptom_disease': self._gen_symptom_disease,
            'disease_cause': self._gen_disease_cause,
            'disease_prevent': self._gen_disease_prevent,
            'disease_cureway': self._gen_disease_cureway,
            'disease_lasttime': self._gen_disease_lasttime,
            'disease_cureprob': self._gen_disease_cureprob,
            'disease_easyget': self._gen_disease_easyget,
            'disease_desc': self._gen_disease_desc,
            'disease_drug': self._gen_disease_drug,
            'drug_disease': self._gen_drug_disease,
            'disease_food': self._gen_disease_food,
            'disease_not_food': self._gen_disease_not_food,
            'disease_check': self._gen_disease_check,
            'check_disease': self._gen_check_disease,
            'disease_department': self._gen_disease_department,
            'disease_acompany': self._gen_disease_acompany,
        }
        
        generator = generator_map.get(intent)
        if generator:
            queries = generator(entities)
        
        return queries
    
    def _gen_disease_symptom(self, entities: Dict) -> List[str]:
        """查询疾病的症状"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)-[r:has_symptom]->(s:Symptom)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, collect(s.name) as symptoms
            """
            queries.append(query)
        
        return queries
    
    def _gen_symptom_disease(self, entities: Dict) -> List[str]:
        """根据症状查询可能的疾病"""
        symptoms = entities.get('symptom', [])
        queries = []
        
        if symptoms:
            # 单个症状查询
            for symptom in symptoms:
                query = f"""
                MATCH (d:Disease)-[r:has_symptom]->(s:Symptom)
                WHERE s.name CONTAINS '{symptom}'
                RETURN d.name as disease, s.name as symptom,
                       d.desc as description
                LIMIT {self.num_limit}
                """
                queries.append(query)
            
            # 多症状联合查询（如果有多个症状）
            if len(symptoms) > 1:
                symptoms_list = str(symptoms)
                query = f"""
                MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
                WHERE s.name IN {symptoms_list}
                WITH d, count(DISTINCT s) as match_count, collect(s.name) as matched_symptoms
                WHERE match_count >= 1
                RETURN d.name as disease, match_count, matched_symptoms,
                       d.desc as description
                ORDER BY match_count DESC
                LIMIT {self.num_limit}
                """
                queries.append(query)
        
        return queries
    
    def _gen_disease_cause(self, entities: Dict) -> List[str]:
        """查询疾病的病因"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.cause as cause
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_prevent(self, entities: Dict) -> List[str]:
        """查询疾病的预防措施"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.prevent as prevention
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_cureway(self, entities: Dict) -> List[str]:
        """查询疾病的治疗方式"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.cure_way as cure_methods
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_lasttime(self, entities: Dict) -> List[str]:
        """查询疾病的治疗周期"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.cure_lasttime as cure_duration
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_cureprob(self, entities: Dict) -> List[str]:
        """查询疾病的治愈概率"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.cured_prob as cure_probability
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_easyget(self, entities: Dict) -> List[str]:
        """查询疾病的易感人群"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.easy_get as susceptible_population
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_desc(self, entities: Dict) -> List[str]:
        """查询疾病的详细介绍"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.desc as description,
                   d.cause as cause, d.prevent as prevention,
                   d.cure_way as cure_methods
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_drug(self, entities: Dict) -> List[str]:
        """查询疾病的常用药物"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            # 常用药
            query1 = f"""
            MATCH (d:Disease)-[r:common_drug]->(drug:Drug)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, 'common' as drug_type, collect(drug.name) as drugs
            """
            queries.append(query1)
            
            # 推荐药
            query2 = f"""
            MATCH (d:Disease)-[r:recommand_drug]->(drug:Drug)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, 'recommended' as drug_type, collect(drug.name) as drugs
            """
            queries.append(query2)
        
        return queries
    
    def _gen_drug_disease(self, entities: Dict) -> List[str]:
        """查询药物能治疗的疾病"""
        drugs = entities.get('drug', [])
        queries = []
        
        for drug in drugs:
            query = f"""
            MATCH (d:Disease)-[:common_drug|recommand_drug]->(drug:Drug)
            WHERE drug.name CONTAINS '{drug}'
            RETURN drug.name as drug, collect(DISTINCT d.name) as diseases
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_food(self, entities: Dict) -> List[str]:
        """查询疾病的饮食建议（宜吃）"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            # 宜吃
            query1 = f"""
            MATCH (d:Disease)-[r:do_eat]->(f:Food)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, 'recommended' as food_type, collect(f.name) as foods
            """
            queries.append(query1)
            
            # 推荐食谱
            query2 = f"""
            MATCH (d:Disease)-[r:recommand_eat]->(f:Food)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, 'recipe' as food_type, collect(f.name) as foods
            """
            queries.append(query2)
        
        return queries
    
    def _gen_disease_not_food(self, entities: Dict) -> List[str]:
        """查询疾病的饮食禁忌（忌吃）"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)-[r:no_eat]->(f:Food)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, collect(f.name) as forbidden_foods
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_check(self, entities: Dict) -> List[str]:
        """查询疾病需要的检查项目"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)-[r:need_check]->(c:Check)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, collect(c.name) as check_items
            """
            queries.append(query)
        
        return queries
    
    def _gen_check_disease(self, entities: Dict) -> List[str]:
        """查询检查项目能检出的疾病"""
        checks = entities.get('check', [])
        queries = []
        
        for check in checks:
            query = f"""
            MATCH (d:Disease)-[r:need_check]->(c:Check)
            WHERE c.name CONTAINS '{check}'
            RETURN c.name as check_item, collect(DISTINCT d.name) as diseases
            LIMIT {self.num_limit}
            """
            queries.append(query)
        
        return queries
    
    def _gen_disease_department(self, entities: Dict) -> List[str]:
        """查询疾病对应的科室"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            query = f"""
            MATCH (d:Disease)-[r:belongs_to]->(dept:Department)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, collect(dept.name) as departments
            """
            queries.append(query)
            
            # 也查询疾病节点上的科室属性
            query2 = f"""
            MATCH (d:Disease)
            WHERE d.name = '{disease}'
            RETURN d.name as disease, d.cure_department as departments
            """
            queries.append(query2)
        
        return queries
    
    def _gen_disease_acompany(self, entities: Dict) -> List[str]:
        """查询疾病的并发症"""
        diseases = entities.get('disease', [])
        queries = []
        
        for disease in diseases:
            # 双向查询并发症关系
            query = f"""
            MATCH (d1:Disease)-[r:acompany_with]-(d2:Disease)
            WHERE d1.name = '{disease}'
            RETURN d1.name as disease, collect(DISTINCT d2.name) as complications
            """
            queries.append(query)
        
        return queries
    
    def generate_comprehensive(self, disease: str) -> str:
        """
        生成综合查询，获取疾病的全部信息
        
        Args:
            disease: 疾病名称
            
        Returns:
            综合查询的 Cypher 语句
        """
        query = f"""
        MATCH (d:Disease)
        WHERE d.name = '{disease}'
        OPTIONAL MATCH (d)-[:has_symptom]->(s:Symptom)
        OPTIONAL MATCH (d)-[:common_drug|recommand_drug]->(drug:Drug)
        OPTIONAL MATCH (d)-[:need_check]->(c:Check)
        OPTIONAL MATCH (d)-[:do_eat|recommand_eat]->(food_good:Food)
        OPTIONAL MATCH (d)-[:no_eat]->(food_bad:Food)
        OPTIONAL MATCH (d)-[:belongs_to]->(dept:Department)
        OPTIONAL MATCH (d)-[:acompany_with]-(comp:Disease)
        RETURN d.name as disease,
               d.desc as description,
               d.cause as cause,
               d.prevent as prevention,
               d.cure_way as cure_methods,
               d.cure_lasttime as cure_duration,
               d.cured_prob as cure_probability,
               d.easy_get as susceptible_population,
               d.cure_department as cure_department,
               collect(DISTINCT s.name) as symptoms,
               collect(DISTINCT drug.name) as drugs,
               collect(DISTINCT c.name) as check_items,
               collect(DISTINCT food_good.name) as recommended_foods,
               collect(DISTINCT food_bad.name) as forbidden_foods,
               collect(DISTINCT dept.name) as departments,
               collect(DISTINCT comp.name) as complications
        """
        return query


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    generator = CypherGenerator()
    
    # 测试用例
    test_cases = [
        ('disease_symptom', {'disease': ['感冒']}),
        ('symptom_disease', {'symptom': ['头疼', '发烧']}),
        ('disease_drug', {'disease': ['高血压']}),
        ('drug_disease', {'drug': ['布洛芬']}),
        ('disease_food', {'disease': ['糖尿病']}),
        ('disease_check', {'disease': ['肺炎']}),
    ]
    
    for intent, entities in test_cases:
        print(f"\n意图: {intent}")
        print(f"实体: {entities}")
        queries = generator.generate(intent, entities)
        for i, q in enumerate(queries):
            print(f"查询 {i+1}:")
            print(q.strip())

