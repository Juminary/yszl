"""
知识图谱模块
连接 Neo4j 图数据库，提供医学知识查询
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class KnowledgeGraphModule:
    """
    医学知识图谱查询模块
    基于 Neo4j 图数据库
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7474,
        user: str = "neo4j",
        password: str = "12345"
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.graph = None
        self.enabled = False
        
        self._connect()
    
    def _connect(self):
        """连接 Neo4j 数据库"""
        try:
            from py2neo import Graph
            
            self.graph = Graph(
                host=self.host,
                http_port=self.port,
                user=self.user,
                password=self.password
            )
            
            # 测试连接
            self.graph.run("RETURN 1")
            self.enabled = True
            
            print("\n" + "="*50)
            print(f"🔗 [知识图谱] 连接成功")
            print(f"   - 地址: {self.host}:{self.port}")
            print("="*50 + "\n")
            
            logger.info(f"Knowledge Graph connected: {self.host}:{self.port}")
            
        except ImportError:
            logger.warning("py2neo not installed. Run: pip install py2neo")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            self.enabled = False
    
    def query(self, cypher: str) -> List[Dict]:
        """执行 Cypher 查询"""
        if not self.enabled or not self.graph:
            return []
        
        try:
            result = self.graph.run(cypher).data()
            return result
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []
    
    def search_by_symptom(self, symptom: str) -> Dict[str, Any]:
        """根据症状查询相关疾病"""
        if not self.enabled:
            return {}
        
        cypher = f"""
        MATCH (d:Disease)-[r:has_symptom]->(s:Symptom)
        WHERE s.name CONTAINS '{symptom}'
        RETURN d.name as disease, s.name as symptom, 
               d.cause as cause, d.cure_way as cure_way
        LIMIT 5
        """
        
        results = self.query(cypher)
        
        if results:
            diseases = list(set([r['disease'] for r in results if r.get('disease')]))
            return {
                'symptom': symptom,
                'possible_diseases': diseases[:5],
                'details': results[:3]
            }
        return {}
    
    def search_by_disease(self, disease: str) -> Dict[str, Any]:
        """根据疾病名查询详细信息"""
        if not self.enabled:
            return {}
        
        # 查询疾病基本信息
        cypher_info = f"""
        MATCH (d:Disease)
        WHERE d.name CONTAINS '{disease}'
        RETURN d.name as name, d.desc as description, 
               d.cause as cause, d.prevent as prevent,
               d.cure_way as cure_way, d.cure_lasttime as cure_time,
               d.cured_prob as cure_prob, d.easy_get as easy_get
        LIMIT 1
        """
        
        info = self.query(cypher_info)
        
        if not info:
            return {}
        
        result = {
            'disease': info[0].get('name', disease),
            'description': info[0].get('description', ''),
            'cause': info[0].get('cause', ''),
            'prevent': info[0].get('prevent', ''),
            'cure_way': info[0].get('cure_way', []),
            'cure_time': info[0].get('cure_time', ''),
            'cure_prob': info[0].get('cure_prob', ''),
            'easy_get': info[0].get('easy_get', '')
        }
        
        # 查询症状
        cypher_symptoms = f"""
        MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
        WHERE d.name = '{result['disease']}'
        RETURN s.name as symptom LIMIT 10
        """
        symptoms = self.query(cypher_symptoms)
        result['symptoms'] = [s['symptom'] for s in symptoms if s.get('symptom')]
        
        # 查询常用药物
        cypher_drugs = f"""
        MATCH (d:Disease)-[:common_drug|recommand_drug]->(drug:Drug)
        WHERE d.name = '{result['disease']}'
        RETURN drug.name as drug LIMIT 10
        """
        drugs = self.query(cypher_drugs)
        result['drugs'] = [d['drug'] for d in drugs if d.get('drug')]
        
        # 查询检查项目
        cypher_checks = f"""
        MATCH (d:Disease)-[:need_check]->(c:Check)
        WHERE d.name = '{result['disease']}'
        RETURN c.name as check_item LIMIT 5
        """
        checks = self.query(cypher_checks)
        result['checks'] = [c['check_item'] for c in checks if c.get('check_item')]
        
        return result
    
    def search_drug_for_disease(self, disease: str) -> List[str]:
        """查询疾病的常用药物"""
        if not self.enabled:
            return []
        
        cypher = f"""
        MATCH (d:Disease)-[:common_drug|recommand_drug]->(drug:Drug)
        WHERE d.name CONTAINS '{disease}'
        RETURN drug.name as drug LIMIT 10
        """
        
        results = self.query(cypher)
        return [r['drug'] for r in results if r.get('drug')]
    
    def search_check_for_disease(self, disease: str) -> List[str]:
        """查询疾病需要的检查项目"""
        if not self.enabled:
            return []
        
        cypher = f"""
        MATCH (d:Disease)-[:need_check]->(c:Check)
        WHERE d.name CONTAINS '{disease}'
        RETURN c.name as check_item LIMIT 10
        """
        
        results = self.query(cypher)
        return [r['check_item'] for r in results if r.get('check_item')]
    
    def search_food_for_disease(self, disease: str) -> Dict[str, List[str]]:
        """查询疾病的饮食建议"""
        if not self.enabled:
            return {}
        
        # 宜吃
        cypher_do = f"""
        MATCH (d:Disease)-[:do_eat|recommand_eat]->(f:Food)
        WHERE d.name CONTAINS '{disease}'
        RETURN f.name as food LIMIT 10
        """
        do_eat = self.query(cypher_do)
        
        # 忌吃
        cypher_not = f"""
        MATCH (d:Disease)-[:no_eat]->(f:Food)
        WHERE d.name CONTAINS '{disease}'
        RETURN f.name as food LIMIT 10
        """
        no_eat = self.query(cypher_not)
        
        return {
            'recommended': [r['food'] for r in do_eat if r.get('food')],
            'avoid': [r['food'] for r in no_eat if r.get('food')]
        }
    
    def build_context_from_query(self, query: str) -> str:
        """
        根据用户查询，从知识图谱中构建上下文
        自动识别查询意图并返回相关信息
        """
        if not self.enabled:
            return ""
        
        context_parts = []
        
        # 尝试从查询中提取症状关键词
        symptom_keywords = ['疼', '痛', '晕', '热', '烧', '咳', '呕', '吐', 
                           '麻', '痒', '肿', '红', '软', '硬', '胀', '闷']
        
        # 检查是否包含症状描述
        has_symptom = any(kw in query for kw in symptom_keywords)
        
        if has_symptom:
            # 尝试症状查询
            for kw in symptom_keywords:
                if kw in query:
                    result = self.search_by_symptom(kw)
                    if result and result.get('possible_diseases'):
                        diseases = result['possible_diseases'][:3]
                        context_parts.append(
                            f"【知识图谱-症状关联】症状'{kw}'可能相关的疾病：{', '.join(diseases)}"
                        )
                        
                        # 对第一个疾病获取详细信息
                        if diseases:
                            detail = self.search_by_disease(diseases[0])
                            if detail:
                                if detail.get('symptoms'):
                                    context_parts.append(
                                        f"【{diseases[0]}的症状】{', '.join(detail['symptoms'][:5])}"
                                    )
                                if detail.get('drugs'):
                                    context_parts.append(
                                        f"【{diseases[0]}常用药物】{', '.join(detail['drugs'][:5])}"
                                    )
                                if detail.get('cure_way'):
                                    ways = detail['cure_way'] if isinstance(detail['cure_way'], list) else [detail['cure_way']]
                                    context_parts.append(
                                        f"【{diseases[0]}治疗方式】{', '.join(ways[:3])}"
                                    )
                        break
        
        # 检查是否询问特定疾病
        disease_markers = ['什么是', '怎么治', '如何治疗', '吃什么药', '做什么检查']
        for marker in disease_markers:
            if marker in query:
                # 提取可能的疾病名
                words = query.replace(marker, ' ').split()
                for word in words:
                    if len(word) >= 2:
                        detail = self.search_by_disease(word)
                        if detail and detail.get('disease'):
                            context_parts.append(
                                f"【知识图谱-{detail['disease']}】"
                            )
                            if detail.get('description'):
                                context_parts.append(f"简介：{detail['description'][:100]}")
                            if detail.get('symptoms'):
                                context_parts.append(f"主要症状：{', '.join(detail['symptoms'][:5])}")
                            if detail.get('drugs'):
                                context_parts.append(f"常用药物：{', '.join(detail['drugs'][:5])}")
                            if detail.get('checks'):
                                context_parts.append(f"检查项目：{', '.join(detail['checks'][:3])}")
                            break
                break
        
        if context_parts:
            graph_context = "\n".join(context_parts)
            print("\n" + "="*50)
            print("📊 [知识图谱] 查询结果:")
            print("-"*50)
            for part in context_parts:
                # 每行最多显示80字符
                if len(part) > 80:
                    print(f"   {part[:77]}...")
                else:
                    print(f"   {part}")
            print("="*50 + "\n")
            return graph_context
        else:
            # 即使没有结果也打印调试信息
            print("\n" + "-"*50)
            print(f"📊 [知识图谱] 未找到匹配: {query[:30]}...")
            print("-"*50)
        
        return ""
    
    def get_info(self) -> Dict:
        """获取知识图谱信息"""
        if not self.enabled:
            return {'enabled': False}
        
        try:
            # 统计节点数量
            result = self.query("MATCH (n) RETURN labels(n)[0] as label, count(*) as count")
            node_counts = {r['label']: r['count'] for r in result}
            
            return {
                'enabled': True,
                'host': f"{self.host}:{self.port}",
                'nodes': node_counts
            }
        except:
            return {'enabled': True, 'host': f"{self.host}:{self.port}"}
