"""
知识图谱模块（改进版）
连接 Neo4j 图数据库，提供医学知识查询
整合：医学词典 + 意图分类 + Cypher生成器
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# 导入新增模块
# 导入新增模块
try:
    from modules.medical.medical_dict import MedicalDictionary
    from modules.medical.intent_classifier import IntentClassifier
    from .cypher_generator import CypherGenerator
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import NLU modules: {e}")
    MODULES_AVAILABLE = False


class KnowledgeGraphModule:
    """
    医学知识图谱查询模块（改进版）
    基于 Neo4j 图数据库
    
    改进点：
    1. 使用医学词典进行精准实体识别
    2. 使用意图分类器理解用户问题
    3. 使用 Cypher 生成器动态构建查询
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7474,
        user: str = "neo4j",
        password: str = "12345",
        dict_dir: str = None
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.graph = None
        self.enabled = False
        
        # 初始化 NLU 组件
        self.medical_dict = None
        self.intent_classifier = None
        self.cypher_generator = None
        
        if MODULES_AVAILABLE:
            try:
                self.medical_dict = MedicalDictionary(dict_dir=dict_dir)
                self.intent_classifier = IntentClassifier()
                self.cypher_generator = CypherGenerator()
                logger.info("NLU modules initialized (医学词典 + 意图分类 + Cypher生成)")
            except Exception as e:
                logger.warning(f"Failed to initialize NLU modules: {e}")
        
        self._connect()
    
    def _connect(self):
        """连接 Neo4j 数据库"""
        try:
            from py2neo import Graph
            
            # Neo4j 5.x 使用 Bolt 协议连接
            # 尝试多种连接方式以兼容不同版本
            bolt_url = f"bolt://{self.host}:7687"
            http_url = f"http://{self.host}:{self.port}"
            
            connected = False
            
            # 优先尝试 Bolt 协议 (Neo4j 4.x/5.x)
            try:
                self.graph = Graph(bolt_url, auth=(self.user, self.password))
                self.graph.run("RETURN 1")
                connected = True
                connection_info = f"bolt://{self.host}:7687"
            except Exception as bolt_e:
                # 尝试 HTTP 协议 (旧版 Neo4j 或 py2neo)
                try:
                    self.graph = Graph(http_url, auth=(self.user, self.password))
                    self.graph.run("RETURN 1")
                    connected = True
                    connection_info = http_url
                except TypeError:
                    # 旧版 py2neo API
                    try:
                        self.graph = Graph(
                            host=self.host,
                            http_port=self.port,
                            user=self.user,
                            password=self.password
                        )
                        self.graph.run("RETURN 1")
                        connected = True
                        connection_info = f"{self.host}:{self.port}"
                    except Exception as old_e:
                        raise Exception(f"All connection methods failed: Bolt={bolt_e}, HTTP={old_e}")
            
            if connected:
                self.enabled = True
                
                print("\n" + "="*50)
                print(f"🔗 [知识图谱] 连接成功")
                print(f"   - 地址: {connection_info}")
                if self.medical_dict:
                    stats = self.medical_dict.get_stats()
                    print(f"   - 词典: {stats['total']} 词条")
                print("="*50 + "\n")
                
                logger.info(f"Knowledge Graph connected: {connection_info}")
            
        except ImportError:
            logger.warning("py2neo not installed. Run: pip install py2neo")
            self.enabled = False
        except Exception as e:
            import traceback
            logger.warning(f"Failed to connect to Neo4j: {e}")
            print(f"\n[知识图谱] ✗ 连接失败: {e}", flush=True)
            print(f"   详细信息: {traceback.format_exc()}", flush=True)
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
    
    # ==================== 新增：智能问答接口 ====================
    
    def smart_query(self, query: str) -> Dict[str, Any]:
        """
        智能问答接口（新增）
        
        完整流程：
        1. 实体识别（医学词典）
        2. 意图分类
        3. 生成 Cypher 查询
        4. 执行查询
        5. 整理结果
        
        Args:
            query: 用户问题
            
        Returns:
            {
                'entities': 识别的实体,
                'intent': 识别的意图,
                'intent_desc': 意图描述,
                'results': 查询结果,
                'context': 格式化的上下文（用于 LLM）
            }
        """
        if not self.enabled:
            return {'error': 'Knowledge graph not connected'}
        
        # 1. 实体识别
        if self.medical_dict:
            entities = self.medical_dict.extract_entities(query)
        else:
            entities = self._extract_entities_legacy(query)
        
        # 2. 意图分类
        if self.intent_classifier:
            intent, confidence = self.intent_classifier.classify(query, entities)
            intent_desc = self.intent_classifier.get_intent_description(intent)
        else:
            intent = 'general_chat'
            confidence = 0.5
            intent_desc = '通用查询'
        
        # 3. 生成 Cypher 查询
        if self.cypher_generator and intent != 'general_chat':
            cypher_queries = self.cypher_generator.generate(intent, entities)
        else:
            cypher_queries = self._generate_legacy_queries(query, entities)
        
        # 4. 执行查询
        all_results = []
        for cypher in cypher_queries:
            results = self.query(cypher)
            if results:
                all_results.extend(results)
        
        # 5. 整理结果并构建上下文
        context = self._build_context_from_results(intent, entities, all_results)
        
        # 打印调试信息
        self._print_debug_info(query, entities, intent, intent_desc, confidence, all_results)
        
        return {
            'entities': entities,
            'intent': intent,
            'intent_desc': intent_desc,
            'confidence': confidence,
            'results': all_results,
            'context': context
        }
    
    def _extract_entities_legacy(self, query: str) -> Dict[str, List[str]]:
        """旧版实体提取（降级方案）"""
        entities = {
            'disease': [],
            'symptom': [],
            'drug': [],
            'check': [],
            'food': [],
            'department': []
        }
        
        # 简单的关键词匹配
        symptom_keywords = ['疼', '痛', '晕', '热', '烧', '咳', '呕', '吐', 
                           '麻', '痒', '肿', '红', '胀', '闷', '头疼', '发烧',
                           '咳嗽', '流鼻涕', '恶心', '腹泻']
        
        for kw in symptom_keywords:
            if kw in query:
                entities['symptom'].append(kw)
        
        return entities
    
    def _generate_legacy_queries(self, query: str, entities: Dict) -> List[str]:
        """旧版查询生成（降级方案）"""
        queries = []
        
        # 根据症状查疾病
        for symptom in entities.get('symptom', []):
            queries.append(f"""
                MATCH (d:Disease)-[r:has_symptom]->(s:Symptom)
                WHERE s.name CONTAINS '{symptom}'
                RETURN d.name as disease, s.name as symptom,
                       d.cause as cause, d.cure_way as cure_way
                LIMIT 5
            """)
        
        # 根据疾病查信息
        for disease in entities.get('disease', []):
            queries.append(f"""
                MATCH (d:Disease)
                WHERE d.name = '{disease}'
                RETURN d.name as disease, d.desc as description,
                       d.cause as cause, d.cure_way as cure_methods
            """)
        
        return queries
    
    def _build_context_from_results(self, intent: str, entities: Dict, results: List[Dict]) -> str:
        """根据查询结果构建 LLM 上下文"""
        if not results:
            return ""
        
        context_parts = []
        
        # 根据不同意图格式化结果
        if intent == 'disease_symptom':
            for r in results:
                if r.get('disease') and r.get('symptoms'):
                    symptoms = r['symptoms'] if isinstance(r['symptoms'], list) else [r['symptoms']]
                    context_parts.append(f"【{r['disease']}的症状】{', '.join(symptoms[:10])}")
        
        elif intent == 'symptom_disease':
            diseases = list(set([r.get('disease') for r in results if r.get('disease')]))
            if diseases:
                context_parts.append(f"【可能的疾病】{', '.join(diseases[:5])}")
            for r in results[:2]:
                if r.get('description'):
                    context_parts.append(f"【{r.get('disease', '疾病')}简介】{r['description'][:100]}")
        
        elif intent == 'disease_drug':
            for r in results:
                if r.get('disease') and r.get('drugs'):
                    drugs = r['drugs'] if isinstance(r['drugs'], list) else [r['drugs']]
                    drug_type = '常用药物' if r.get('drug_type') == 'common' else '推荐药物'
                    context_parts.append(f"【{r['disease']}{drug_type}】{', '.join(drugs[:10])}")
        
        elif intent == 'drug_disease':
            for r in results:
                if r.get('drug') and r.get('diseases'):
                    diseases = r['diseases'] if isinstance(r['diseases'], list) else [r['diseases']]
                    context_parts.append(f"【{r['drug']}可治疗】{', '.join(diseases[:10])}")
        
        elif intent == 'disease_food':
            for r in results:
                if r.get('disease') and r.get('foods'):
                    foods = r['foods'] if isinstance(r['foods'], list) else [r['foods']]
                    food_type = '推荐食谱' if r.get('food_type') == 'recipe' else '宜吃食物'
                    context_parts.append(f"【{r['disease']}{food_type}】{', '.join(foods[:10])}")
        
        elif intent == 'disease_not_food':
            for r in results:
                if r.get('disease') and r.get('forbidden_foods'):
                    foods = r['forbidden_foods'] if isinstance(r['forbidden_foods'], list) else [r['forbidden_foods']]
                    context_parts.append(f"【{r['disease']}忌口食物】{', '.join(foods[:10])}")
        
        elif intent == 'disease_check':
            for r in results:
                if r.get('disease') and r.get('check_items'):
                    checks = r['check_items'] if isinstance(r['check_items'], list) else [r['check_items']]
                    context_parts.append(f"【{r['disease']}检查项目】{', '.join(checks[:10])}")
        
        elif intent == 'disease_cause':
            for r in results:
                if r.get('disease') and r.get('cause'):
                    context_parts.append(f"【{r['disease']}病因】{r['cause'][:200]}")
        
        elif intent == 'disease_prevent':
            for r in results:
                if r.get('disease') and r.get('prevention'):
                    context_parts.append(f"【{r['disease']}预防措施】{r['prevention'][:200]}")
        
        elif intent == 'disease_cureway':
            for r in results:
                if r.get('disease') and r.get('cure_methods'):
                    methods = r['cure_methods']
                    if isinstance(methods, list):
                        methods = ', '.join(methods[:5])
                    context_parts.append(f"【{r['disease']}治疗方式】{methods[:200]}")
        
        elif intent == 'disease_desc':
            for r in results:
                if r.get('disease'):
                    parts = [f"【{r['disease']}】"]
                    if r.get('description'):
                        parts.append(f"简介：{r['description'][:150]}")
                    if r.get('cause'):
                        parts.append(f"病因：{r['cause'][:100]}")
                    if r.get('cure_methods'):
                        methods = r['cure_methods']
                        if isinstance(methods, list):
                            methods = ', '.join(methods[:3])
                        parts.append(f"治疗：{methods[:100]}")
                    context_parts.append('\n'.join(parts))
        
        elif intent == 'disease_department':
            for r in results:
                if r.get('disease') and r.get('departments'):
                    depts = r['departments']
                    if isinstance(depts, list):
                        depts = ', '.join(depts)
                    context_parts.append(f"【{r['disease']}就诊科室】{depts}")
        
        elif intent == 'disease_acompany':
            for r in results:
                if r.get('disease') and r.get('complications'):
                    comps = r['complications'] if isinstance(r['complications'], list) else [r['complications']]
                    context_parts.append(f"【{r['disease']}并发症】{', '.join(comps[:10])}")
        
        else:
            # 通用格式化
            for r in results[:3]:
                if r.get('disease'):
                    info = [f"【{r['disease']}】"]
                    for key in ['description', 'cause', 'cure_way', 'symptoms']:
                        if r.get(key):
                            val = r[key]
                            if isinstance(val, list):
                                val = ', '.join(val[:5])
                            info.append(f"{key}: {str(val)[:100]}")
                    context_parts.append(' | '.join(info))
        
        return '\n'.join(context_parts)
    
    def _print_debug_info(self, query: str, entities: Dict, intent: str, 
                          intent_desc: str, confidence: float, results: List):
        """打印调试信息到控制台"""
        import sys
        
        # 使用 sys.stdout 确保立即输出
        output = []
        output.append("\n" + "="*60)
        output.append("📊 [知识图谱智能查询]")
        output.append("-"*60)
        output.append(f"   问题: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # 显示识别的实体
        entity_str = []
        for etype, elist in entities.items():
            if elist:
                entity_str.append(f"{etype}: {elist}")
        if entity_str:
            output.append(f"   实体: {', '.join(entity_str)}")
        else:
            output.append(f"   实体: (未识别到医学实体)")
        
        output.append(f"   意图: {intent} ({intent_desc}) [置信度: {confidence:.2f}]")
        output.append(f"   结果: {len(results)} 条记录")
        
        # 显示部分结果
        if results:
            output.append("-"*60)
            for i, r in enumerate(results[:3]):
                # 格式化显示结果
                if r.get('disease'):
                    disease = r.get('disease', '')
                    if r.get('symptoms'):
                        symptoms = r['symptoms'][:5] if isinstance(r['symptoms'], list) else [r['symptoms']]
                        output.append(f"   [{i+1}] {disease} - 症状: {', '.join(symptoms)}")
                    elif r.get('drugs'):
                        drugs = r['drugs'][:5] if isinstance(r['drugs'], list) else [r['drugs']]
                        output.append(f"   [{i+1}] {disease} - 药物: {', '.join(drugs)}")
                    elif r.get('description'):
                        desc = str(r['description'])[:60]
                        output.append(f"   [{i+1}] {disease} - {desc}...")
                    else:
                        preview = str(r)[:70]
                        output.append(f"   [{i+1}] {preview}...")
                elif r.get('symptom'):
                    output.append(f"   [{i+1}] 症状: {r.get('symptom')} -> 疾病: {r.get('disease', '未知')}")
                else:
                    preview = str(r)[:70]
                    output.append(f"   [{i+1}] {preview}...")
        else:
            output.append("   (未查询到相关结果)")
        
        output.append("="*60 + "\n")
        
        # 打印到控制台
        print('\n'.join(output), flush=True)
    
    # ==================== 原有接口（保持兼容） ====================
    
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
        根据用户查询，从知识图谱中构建上下文（改进版）
        
        优先使用智能查询，失败时降级到旧版逻辑
        """
        if not self.enabled:
            print("[知识图谱] ✗ 未连接，跳过查询", flush=True)
            return ""
        
        # 使用新的智能查询
        if self.medical_dict and self.intent_classifier:
            result = self.smart_query(query)
            if result.get('context'):
                return result['context']
            else:
                print("[知识图谱] ✗ 智能查询无结果，尝试旧版逻辑", flush=True)
        
        # 降级到旧版逻辑
        return self._build_context_legacy(query)
    
    def _build_context_legacy(self, query: str) -> str:
        """旧版上下文构建（保持兼容）"""
        print("\n" + "-"*50, flush=True)
        print("📊 [知识图谱] 使用旧版查询逻辑", flush=True)
        print(f"   问题: {query[:40]}...", flush=True)
        
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
                        print(f"   匹配症状: {kw} -> 疾病: {', '.join(diseases)}", flush=True)
                        
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
                            print(f"   匹配疾病: {detail['disease']}", flush=True)
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
            print(f"   结果: {len(context_parts)} 条信息", flush=True)
            print("-"*50 + "\n", flush=True)
            return "\n".join(context_parts)
        
        print("   结果: (无匹配)", flush=True)
        print("-"*50 + "\n", flush=True)
        return ""
    
    def get_info(self) -> Dict:
        """获取知识图谱信息"""
        info = {
            'enabled': self.enabled,
            'host': f"{self.host}:{self.port}",
            'nlu_modules': {
                'medical_dict': self.medical_dict is not None,
                'intent_classifier': self.intent_classifier is not None,
                'cypher_generator': self.cypher_generator is not None
            }
        }
        
        if self.medical_dict:
            info['dict_stats'] = self.medical_dict.get_stats()
        
        if self.enabled:
            try:
                # 统计节点数量
                result = self.query("MATCH (n) RETURN labels(n)[0] as label, count(*) as count")
                info['nodes'] = {r['label']: r['count'] for r in result}
            except:
                pass
        
        return info


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化模块
    kg = KnowledgeGraphModule(
        host="172.24.30.243",
        port=7474,
        user="neo4j",
        password="12345"
    )
    
    print("知识图谱信息:", kg.get_info())
    
    # 测试智能查询
    test_queries = [
        "感冒有什么症状",
        "头疼发烧是什么病",
        "高血压吃什么药",
        "糖尿病不能吃什么",
        "肺炎要做什么检查",
        "怎么预防感冒",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"测试: {query}")
        result = kg.smart_query(query)
        print(f"上下文:\n{result.get('context', '(无)')}")
