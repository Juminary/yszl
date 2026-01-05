"""
患者导诊服务 - 重构版
基于 RAG+LLM 进行科室匹配和疾病预测
使用多因素投票推荐医生
"""

import sqlite3
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TriageService:
    """
    导诊服务
    
    功能：
    1. 使用 RAG+LLM 分析症状，匹配科室，预测可能疾病
    2. 使用多因素投票推荐医生（科室、擅长、职称、排队人数）
    """
    
    # 职称权重（用于投票）
    TITLE_WEIGHTS = {
        '主任医师': 1.0,
        '副主任医师': 0.8,
        '主治医师': 0.6,
        '住院医师': 0.4,
    }
    
    def __init__(self, db_path: str, dialogue_module=None, rag_module=None, entity_extractor=None):
        """
        初始化导诊服务
        
        Args:
            db_path: SQLite 数据库路径
            dialogue_module: 对话模块（用于 LLM 生成）
            rag_module: RAG 模块（用于知识检索）
            entity_extractor: 实体提取器（用于提取症状）
        """
        self.db_path = db_path
        self.dialogue = dialogue_module
        self.rag = rag_module
        self.extractor = entity_extractor
        
        # 加载科室关键词映射
        self.dept_keywords = self._load_department_keywords()
        
        logger.info(f"[导诊服务] 初始化完成，加载 {len(self.dept_keywords)} 个科室")
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def _load_department_keywords(self) -> Dict[str, List[str]]:
        """加载科室-关键词映射"""
        result = {}
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, keywords FROM departments')
            for dept_id, name, keywords in cursor.fetchall():
                if keywords:
                    result[dept_id] = {
                        'name': name,
                        'keywords': [k.strip() for k in keywords.split(',')]
                    }
            conn.close()
        except Exception as e:
            logger.error(f"[导诊服务] 加载科室关键词失败: {e}")
        return result
    
    def analyze(self, user_input: str, age: Optional[int] = None, 
                gender: Optional[str] = None) -> Dict:
        """
        分析用户输入，返回导诊结果
        
        Args:
            user_input: 患者症状描述
            age: 年龄
            gender: 性别
        
        Returns:
            {
                'department': {'id': str, 'name': str},
                'diseases': [str],
                'doctors': [{'name', 'title', 'specialty', 'room', 'queue', 'score'}],
                'response': str,
                'symptoms': [str]
            }
        """
        try:
            # 1. 提取症状
            symptoms = self._extract_symptoms(user_input)
            logger.info(f"[导诊服务] 提取症状: {symptoms}")
            
            # 2. 使用 RAG+LLM 匹配科室和预测疾病
            dept_result = self._match_department_with_llm(user_input, symptoms)
            department = dept_result.get('department', {})
            diseases = dept_result.get('diseases', [])
            logger.info(f"[导诊服务] 匹配科室: {department.get('name')}, 可能疾病: {diseases}")
            
            # 3. 多因素投票推荐医生
            doctors = []
            if department.get('id'):
                doctors = self._recommend_doctors_voting(
                    department['id'], 
                    diseases,
                    top_k=3
                )
                logger.info(f"[导诊服务] 推荐医生: {[d['name'] for d in doctors]}")
            
            # 4. 生成导诊回复
            response = self._generate_response(
                user_input, symptoms, department, diseases, doctors
            )
            
            return {
                'department': department,
                'diseases': diseases,
                'doctors': doctors,
                'symptoms': symptoms,
                'response': response,
                'patient_info': {'age': age, 'gender': gender}
            }
            
        except Exception as e:
            logger.error(f"[导诊服务] 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'department': {},
                'diseases': [],
                'doctors': [],
                'symptoms': [],
                'response': '抱歉，导诊服务出现问题，建议您直接前往医院咨询。',
                'error': str(e)
            }
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """提取症状（增强版）"""
        symptoms = []
        
        # 常见症状词直接匹配
        common_symptoms = [
            '头痛', '头疼', '发烧', '发热', '咳嗽', '胸闷', '胸痛', 
            '腹痛', '肚子疼', '腹泻', '拉肚子', '恶心', '呕吐',
            '头晕', '眩晕', '心慌', '心悸', '气短', '呼吸困难',
            '失眠', '乏力', '疲劳', '皮疹', '瘙痒', '水肿',
            '尿频', '尿急', '便秘', '关节痛', '腰痛', '颈椎痛'
        ]
        
        # 直接匹配常见症状
        for symptom in common_symptoms:
            if symptom in text:
                symptoms.append(symptom)
        
        # 如果有实体提取器，也使用它
        if self.extractor and not symptoms:
            try:
                entities = self.extractor.extract_entities(text, speaker_role='patient')
                symptoms = [e.text for e in entities if e.type == 'symptom']
            except Exception as e:
                logger.warning(f"[导诊服务] 实体提取失败: {e}")
        
        # 补充：基于科室关键词匹配
        if not symptoms:
            for dept_id, dept_info in self.dept_keywords.items():
                for keyword in dept_info['keywords']:
                    if keyword in text and keyword not in symptoms:
                        symptoms.append(keyword)
        
        print(f"[DEBUG] 从输入提取症状: {symptoms}")
        return symptoms[:10]  # 最多返回10个
    
    def _match_department_with_llm(self, user_input: str, symptoms: List[str]) -> Dict:
        """
        使用 RAG+LLM 匹配科室并预测疾病
        """
        # 获取科室列表
        dept_list = [f"{d['name']}" for d in self.dept_keywords.values()]
        
        if not self.dialogue:
            print("[DEBUG] 无 LLM 模块，使用关键词匹配")
            return self._match_department_by_keywords(symptoms)
        
        try:
            print("[DEBUG] 使用 LLM 进行科室匹配...")
            
            # 构建 RAG 上下文
            rag_context = ""
            if self.rag:
                rag_context = self.rag.build_context(user_input, top_k=3) or ""
                print(f"[DEBUG] RAG 上下文长度: {len(rag_context)}")
            
            # 构建 LLM 提示
            prompt = f"""你是一位专业的医院导诊护士。根据患者症状，判断应该挂哪个科室，并预测可能的疾病。

患者症状描述：{user_input}
提取的症状：{', '.join(symptoms) if symptoms else '未明确'}

参考医学信息：
{rag_context if rag_context else '无'}

可选科室：{', '.join(dept_list)}

请直接输出以下格式（不要其他内容）：
科室：[科室名称]
可能疾病：[疾病1]，[疾病2]，[疾病3]"""

            result = self.dialogue.chat(
                query=prompt,
                session_id="triage_match",
                reset=True,
                use_rag=False
            )
            
            response_text = result.get('response', '')
            print(f"[DEBUG] LLM 响应: {response_text[:200]}...")
            
            # 解析响应
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            logger.warning(f"[导诊服务] LLM匹配失败: {e}")
            return self._match_department_by_keywords(symptoms)
    
    def _parse_llm_response(self, text: str) -> Dict:
        """解析 LLM 响应（支持多种格式）"""
        import re
        
        result = {'department': {}, 'diseases': []}
        
        print(f"[DEBUG] 解析 LLM 响应: {text[:100]}...")
        
        # 提取科室 - 支持多种格式
        # 格式1: 科室：xxx 或 科室:xxx
        # 格式2: 科室，xxx
        dept_match = re.search(r'科室[：:，,]\s*([^可能疾病\n]+?)(?=可能|疾病|\n|$)', text)
        if dept_match:
            dept_name = dept_match.group(1).strip()
            print(f"[DEBUG] 提取到科室名: {dept_name}")
            # 查找匹配的科室ID
            for dept_id, dept_info in self.dept_keywords.items():
                if dept_info['name'] in dept_name or dept_name in dept_info['name']:
                    result['department'] = {'id': dept_id, 'name': dept_info['name']}
                    print(f"[DEBUG] 匹配到科室: {dept_info['name']}")
                    break
        
        # 如果没有匹配到，直接在文本中搜索科室名
        if not result['department']:
            print("[DEBUG] 未匹配科室，尝试直接搜索...")
            for dept_id, dept_info in self.dept_keywords.items():
                if dept_info['name'] in text:
                    result['department'] = {'id': dept_id, 'name': dept_info['name']}
                    print(f"[DEBUG] 直接搜索匹配到科室: {dept_info['name']}")
                    break
        
        # 提取疾病 - 支持多种格式
        disease_match = re.search(r'疾病[：:，,]\s*(.+?)(?:\n|$)', text)
        if disease_match:
            diseases_text = disease_match.group(1)
            result['diseases'] = [d.strip() for d in re.split(r'[，,、。]', diseases_text) if d.strip() and len(d.strip()) > 1]
        
        return result
    
    def _match_department_by_keywords(self, symptoms: List[str]) -> Dict:
        """基于关键词匹配科室（支持模糊匹配和同义词）"""
        if not symptoms:
            return {'department': {}, 'diseases': []}
        
        # 同义词映射
        synonyms = {
            '头疼': '头痛', '头痛': '头疼',
            '发烧': '发热', '发热': '发烧',
            '肚子疼': '腹痛', '腹痛': '肚子疼',
            '拉肚子': '腹泻', '腹泻': '拉肚子',
            '心慌': '心悸', '心悸': '心慌',
            '胸口闷': '胸闷', '胸闷': '胸口闷',
        }
        
        # 扩展症状列表（加入同义词）
        expanded_symptoms = set(symptoms)
        for symptom in symptoms:
            if symptom in synonyms:
                expanded_symptoms.add(synonyms[symptom])
            # 提取关键部分（如"头痛"中的"头"和"痛"）
            for char in symptom:
                if char in ['头', '胸', '腹', '心', '咳', '发', '痛', '热', '晕']:
                    expanded_symptoms.add(char)
        
        print(f"[DEBUG] 扩展症状: {expanded_symptoms}")
        
        # 计算每个科室的匹配分数
        scores = {}
        for dept_id, dept_info in self.dept_keywords.items():
            score = 0
            matched_keywords = []
            for symptom in expanded_symptoms:
                for keyword in dept_info['keywords']:
                    # 完全匹配
                    if symptom == keyword:
                        score += 2
                        matched_keywords.append(keyword)
                    # 包含匹配
                    elif symptom in keyword or keyword in symptom:
                        score += 1
                        matched_keywords.append(keyword)
                    # 共同字符匹配（至少1个关键字符相同）
                    elif len(set(symptom) & set(keyword)) >= 1 and len(symptom) >= 2:
                        score += 0.5
            if score > 0:
                scores[dept_id] = score
                if matched_keywords:
                    print(f"[DEBUG] 科室 {dept_info['name']} 匹配: {matched_keywords}, 得分: {score}")
        
        if not scores:
            return {'department': {}, 'diseases': []}
        
        # 选择得分最高的科室
        best_dept_id = max(scores, key=scores.get)
        return {
            'department': {
                'id': best_dept_id,
                'name': self.dept_keywords[best_dept_id]['name']
            },
            'diseases': []  # 关键词匹配无法预测疾病
        }
    
    def _recommend_doctors_voting(self, dept_id: str, diseases: List[str], 
                                   top_k: int = 3) -> List[Dict]:
        """
        多因素投票推荐医生
        
        投票因素：
        - 科室匹配：30%（已经是该科室医生，固定得分）
        - 擅长疾病：30%
        - 职称经验：20%
        - 排队人数少：20%
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 获取该科室的所有医生
        cursor.execute('''
            SELECT id, name, title, specialty, room, experience_years, current_queue
            FROM doctors
            WHERE department_id = ?
        ''', (dept_id,))
        
        doctors = []
        for row in cursor.fetchall():
            doc = {
                'id': row[0],
                'name': row[1],
                'title': row[2],
                'specialty': row[3],
                'room': row[4],
                'experience_years': row[5],
                'queue': row[6],
            }
            
            # 计算综合得分
            score = self._calculate_doctor_score(doc, diseases)
            doc['score'] = round(score, 2)
            doctors.append(doc)
        
        conn.close()
        
        # 按得分排序，取前 top_k
        doctors.sort(key=lambda x: x['score'], reverse=True)
        return doctors[:top_k]
    
    def _calculate_doctor_score(self, doctor: Dict, diseases: List[str]) -> float:
        """
        计算医生综合得分（在已匹配科室内部排名）
        
        权重：
        - 擅长疾病匹配：40%
        - 职称经验：30%
        - 排队人数少：30%
        """
        score = 0.0
        
        # 1. 擅长疾病匹配：40%
        specialty = doctor.get('specialty', '')
        disease_score = 0
        if diseases:
            for disease in diseases:
                if disease in specialty or specialty in disease:
                    disease_score = 1.0
                    break
                # 部分匹配（有共同字符）
                common_chars = set(disease) & set(specialty)
                if len(common_chars) >= 2:
                    disease_score = max(disease_score, 0.5)
        score += 0.40 * disease_score
        
        # 2. 职称经验：30%
        title = doctor.get('title', '')
        title_weight = self.TITLE_WEIGHTS.get(title, 0.3)
        score += 0.30 * title_weight
        
        # 3. 排队人数少：30%（排队越少得分越高）
        queue = doctor.get('queue', 10)
        max_queue = 20  # 假设最大排队20人
        queue_score = max(0, 1 - queue / max_queue)
        score += 0.30 * queue_score
        
        return score
    
    def _generate_response(self, user_input: str, symptoms: List[str],
                           department: Dict, diseases: List[str],
                           doctors: List[Dict]) -> str:
        """生成导诊回复"""
        if not department:
            return "根据您的描述，建议您先到医院挂号处咨询，他们会根据您的具体情况推荐合适的科室。"
        
        # 构建回复
        parts = []
        
        # 科室推荐
        dept_name = department.get('name', '综合门诊')
        parts.append(f"根据您描述的症状，建议您到{dept_name}就诊")
        
        # 疾病预测
        if diseases:
            parts.append(f"可能的情况包括{','.join(diseases[:3])}")
        
        # 医生推荐
        if doctors:
            best_doc = doctors[0]
            parts.append(f"推荐您挂{best_doc['name']}{best_doc['title']}的号")
            parts.append(f"诊室在{best_doc['room']}")
            parts.append(f"目前排队{best_doc['queue']}人")
        
        return '，'.join(parts) + '。'
    
    def update_queue(self, doctor_id: int, delta: int = 1):
        """更新医生排队人数"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE doctors 
                SET current_queue = MAX(0, current_queue + ?)
                WHERE id = ?
            ''', (delta, doctor_id))
            conn.commit()
            conn.close()
            logger.info(f"[导诊服务] 更新医生 {doctor_id} 排队人数: {delta:+d}")
        except Exception as e:
            logger.error(f"[导诊服务] 更新排队失败: {e}")
    
    def get_all_departments(self) -> List[Dict]:
        """获取所有科室列表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM departments ORDER BY name')
        result = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]
        conn.close()
        return result
    
    def get_doctors_by_department(self, dept_id: str) -> List[Dict]:
        """获取指定科室的医生列表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, title, specialty, room, current_queue
            FROM doctors
            WHERE department_id = ?
            ORDER BY current_queue ASC
        ''', (dept_id,))
        result = [{
            'id': row[0],
            'name': row[1],
            'title': row[2],
            'specialty': row[3],
            'room': row[4],
            'queue': row[5]
        } for row in cursor.fetchall()]
        conn.close()
        return result


# 保留旧的 TriageModule 接口以保持向后兼容
class TriageModule(TriageService):
    """向后兼容的别名"""
    
    def __init__(self, knowledge_path: str = None, rag_module=None, dialogue_module=None):
        # 默认数据库路径
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'hospital.db'
        )
        super().__init__(
            db_path=db_path,
            dialogue_module=dialogue_module,
            rag_module=rag_module
        )
    
    def triage(self, query: str, age: Optional[int] = None, 
               gender: Optional[str] = None) -> Dict:
        """向后兼容的导诊接口"""
        result = self.analyze(query, age, gender)
        # 转换为旧格式
        return {
            'response': result.get('response', ''),
            'query': query,
            'patient_info': result.get('patient_info', {}),
            'priority': 'normal',
            'rag_used': self.rag is not None,
            'department': result.get('department', {}),
            'diseases': result.get('diseases', []),
            'doctors': result.get('doctors', [])
        }


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'hospital.db')
    
    if os.path.exists(db_path):
        service = TriageService(db_path)
        
        # 测试分析
        test_queries = [
            "我头疼了三天，还有点发烧",
            "最近胸口闷，心跳很快",
            "肚子痛，拉肚子"
        ]
        
        for query in test_queries:
            print(f"\n=== 测试: {query} ===")
            result = service.analyze(query)
            print(f"科室: {result.get('department', {}).get('name', '未匹配')}")
            print(f"疾病: {result.get('diseases', [])}")
            print(f"推荐医生: {[d['name'] for d in result.get('doctors', [])]}")
            print(f"回复: {result.get('response', '')}")
    else:
        print(f"数据库不存在: {db_path}")
        print("请先运行 data/init_hospital_db.py 初始化数据库")
