#!/usr/bin/env python3
"""
医院数据库初始化脚本
创建科室和医生表，填充模拟数据
"""

import sqlite3
import os
import random

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), 'hospital.db')

# 科室数据：id, 名称, 症状关键词
DEPARTMENTS = [
    ('neuro', '神经内科', '头痛,头晕,偏头痛,失眠,手脚麻木,眩晕,记忆力下降,抽搐'),
    ('cardio', '心内科', '胸痛,心悸,心慌,胸闷,气短,心跳快,血压高'),
    ('gastro', '消化内科', '腹痛,腹泻,便秘,恶心,呕吐,胃痛,反酸,烧心'),
    ('resp', '呼吸内科', '咳嗽,咳痰,气喘,呼吸困难,胸闷,喘息'),
    ('endo', '内分泌科', '多饮,多尿,体重下降,肥胖,甲状腺,糖尿病'),
    ('nephro', '肾内科', '水肿,尿血,尿蛋白,腰痛,肾结石'),
    ('rheum', '风湿免疫科', '关节痛,关节肿,晨僵,口干,眼干'),
    ('hema', '血液科', '贫血,出血,淤青,淋巴结肿大'),
    ('derm', '皮肤科', '皮疹,瘙痒,红肿,脱皮,水泡,痤疮'),
    ('ent', '耳鼻喉科', '耳鸣,听力下降,鼻塞,流鼻涕,咽痛,声音嘶哑'),
    ('ophth', '眼科', '视力下降,眼痛,眼红,流泪,眼干'),
    ('ortho', '骨科', '骨折,扭伤,腰痛,颈椎痛,关节疼'),
    ('uro', '泌尿外科', '尿频,尿急,尿痛,血尿,排尿困难'),
    ('gensurg', '普外科', '腹痛,包块,疝气,阑尾炎'),
    ('obgyn', '妇产科', '月经不调,痛经,白带异常,孕期检查'),
    ('pedi', '儿科', '小儿发热,小儿咳嗽,小儿腹泻,小儿皮疹'),
    ('psych', '精神科', '焦虑,抑郁,失眠,幻觉,情绪低落'),
    ('tcm', '中医科', '体质调理,慢性病,亚健康'),
    ('emergency', '急诊科', '剧烈疼痛,大出血,意识不清,呼吸困难,高烧'),
    ('oncology', '肿瘤科', '肿块,消瘦,持续疼痛,淋巴结肿大'),
]

# 医生姓氏和名字
SURNAMES = ['张', '王', '李', '刘', '陈', '杨', '黄', '周', '吴', '赵', '徐', '孙', '马', '朱', '胡']
GIVEN_NAMES = ['明', '华', '伟', '芳', '娜', '静', '强', '磊', '洋', '勇', '敏', '军', '辉', '平', '涛']

# 职称
TITLES = [
    ('主任医师', 20, 30),  # 职称, 最少年限, 最多年限
    ('副主任医师', 12, 20),
    ('主治医师', 5, 12),
    ('住院医师', 1, 5),
]

# 每个科室的擅长疾病
SPECIALTIES = {
    'neuro': ['脑血管疾病', '癫痫', '帕金森', '头痛', '眩晕症'],
    'cardio': ['冠心病', '心律失常', '高血压', '心力衰竭', '心肌病'],
    'gastro': ['胃炎', '肠炎', '肝病', '胰腺炎', '消化道出血'],
    'resp': ['肺炎', '哮喘', '慢阻肺', '肺结核', '呼吸衰竭'],
    'endo': ['糖尿病', '甲亢', '甲减', '骨质疏松', '肥胖症'],
    'nephro': ['肾炎', '肾衰竭', '尿路感染', '肾结石', '透析'],
    'rheum': ['类风湿', '红斑狼疮', '痛风', '强直性脊柱炎', '干燥综合征'],
    'hema': ['贫血', '白血病', '淋巴瘤', '血小板减少', '凝血障碍'],
    'derm': ['湿疹', '银屑病', '荨麻疹', '痤疮', '皮肤感染'],
    'ent': ['中耳炎', '鼻炎', '咽炎', '扁桃体炎', '耳聋'],
    'ophth': ['白内障', '青光眼', '近视', '结膜炎', '眼底病'],
    'ortho': ['骨折', '腰椎病', '颈椎病', '关节炎', '运动损伤'],
    'uro': ['前列腺炎', '肾结石', '膀胱炎', '尿道炎', '泌尿肿瘤'],
    'gensurg': ['阑尾炎', '胆囊炎', '疝气', '甲状腺手术', '乳腺疾病'],
    'obgyn': ['月经病', '不孕症', '妇科肿瘤', '产前检查', '分娩'],
    'pedi': ['小儿感冒', '小儿肺炎', '小儿腹泻', '生长发育', '儿童保健'],
    'psych': ['抑郁症', '焦虑症', '失眠症', '精神分裂', '强迫症'],
    'tcm': ['中医调理', '针灸推拿', '慢病管理', '体质调养', '亚健康'],
    'emergency': ['急性心梗', '脑卒中', '创伤', '中毒', '休克'],
    'oncology': ['肺癌', '胃癌', '肝癌', '乳腺癌', '化疗'],
}


def generate_doctor_name():
    """生成随机医生姓名"""
    return random.choice(SURNAMES) + random.choice(GIVEN_NAMES)


def generate_room(dept_id, index):
    """生成诊室号"""
    building = random.choice(['A', 'B', 'C'])
    floor = random.randint(1, 5)
    room = index + 1
    return f"{building}{floor}{room:02d}"


def init_database():
    """初始化数据库"""
    # 删除旧数据库
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"已删除旧数据库: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建科室表
    cursor.execute('''
        CREATE TABLE departments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            keywords TEXT
        )
    ''')
    
    # 创建医生表
    cursor.execute('''
        CREATE TABLE doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department_id TEXT,
            title TEXT,
            specialty TEXT,
            room TEXT,
            experience_years INTEGER,
            current_queue INTEGER DEFAULT 0,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    ''')
    
    # 插入科室数据
    for dept_id, name, keywords in DEPARTMENTS:
        cursor.execute(
            'INSERT INTO departments (id, name, keywords) VALUES (?, ?, ?)',
            (dept_id, name, keywords)
        )
    print(f"已插入 {len(DEPARTMENTS)} 个科室")
    
    # 为每个科室生成医生
    doctor_count = 0
    for dept_id, dept_name, _ in DEPARTMENTS:
        # 每个科室 3-5 位医生
        num_doctors = random.randint(3, 5)
        specialties = SPECIALTIES.get(dept_id, ['综合诊疗'])
        
        for i in range(num_doctors):
            name = generate_doctor_name()
            title, min_years, max_years = random.choice(TITLES)
            experience = random.randint(min_years, max_years)
            specialty = random.choice(specialties)
            room = generate_room(dept_id, i)
            queue = random.randint(0, 15)  # 模拟当前排队人数
            
            cursor.execute('''
                INSERT INTO doctors 
                (name, department_id, title, specialty, room, experience_years, current_queue)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, dept_id, title, specialty, room, experience, queue))
            doctor_count += 1
    
    print(f"已插入 {doctor_count} 位医生")
    
    conn.commit()
    conn.close()
    
    print(f"\n数据库初始化完成: {DB_PATH}")
    return DB_PATH


def show_sample_data():
    """显示示例数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n=== 科室列表 ===")
    cursor.execute('SELECT id, name FROM departments LIMIT 5')
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    print("\n=== 医生示例 ===")
    cursor.execute('''
        SELECT d.name, dept.name, d.title, d.specialty, d.room, d.current_queue
        FROM doctors d
        JOIN departments dept ON d.department_id = dept.id
        LIMIT 10
    ''')
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2]} | 擅长{row[3]} | {row[4]}诊室 | 排队{row[5]}人")
    
    conn.close()


if __name__ == '__main__':
    init_database()
    show_sample_data()
