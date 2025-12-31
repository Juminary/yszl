# 医疗语音助手 API 文档

## 🎯 系统概述

本系统提供**患者端**和**医生端**双端服务：
- **患者端**：智能导诊、症状咨询
- **医生端**：辅助诊断、用药查询

---

## 📍 服务器地址

默认：`http://localhost:6007`

---

## 🏥 患者端 API

### 1. 智能导诊

根据患者症状推荐就诊科室。

**请求：**
```http
POST /patient/triage
Content-Type: application/json

{
  "symptoms": ["发烧", "咳嗽", "头痛"],
  "age": 8,
  "gender": "男",
  "severity": "normal"  // normal/urgent/emergency
}
```

**响应：**
```json
{
  "status": "success",
  "result": {
    "recommended_department": {
      "id": "respiratory",
      "name": "呼吸内科",
      "description": "诊治呼吸系统疾病"
    },
    "priority": "normal",  // normal/urgent/emergency
    "reason": "根据您的症状（发烧、咳嗽），推荐就诊科室",
    "alternatives": [...],  // 其他可能的科室
    "matched_symptoms": ["发烧", "咳嗽"]
  }
}
```

**使用示例：**
```python
import requests

response = requests.post(
    "http://localhost:6007/patient/triage",
    json={
        "symptoms": ["胸痛", "呼吸困难"],
        "age": 55,
        "gender": "男",
        "severity": "emergency"
    }
)
print(response.json())
```

### 2. 信息提取

从患者描述中自动提取症状、严重程度等信息。

**请求：**
```http
POST /patient/collect-info
Content-Type: application/json

{
  "query": "我发烧已经三天了，还有头痛和咳嗽"
}
```

**响应：**
```json
{
  "status": "success",
  "info": {
    "symptoms": ["发烧", "头痛", "咳嗽"],
    "duration": "三天",
    "severity": "normal",
    "age": null,
    "gender": null
  }
}
```

---

## 👨‍⚕️ 医生端 API

### 1. 症状分析与疾病推断

基于症状分析，给出可能的疾病列表和诊断建议。

**请求：**
```http
POST /doctor/analyze-symptoms
Content-Type: application/json

{
  "symptoms": ["高热", "头痛", "肌肉酸痛", "乏力"],
  "patient_info": {
    "age": 35,
    "gender": "男",
    "medical_history": []
  }
}
```

**响应：**
```json
{
  "status": "success",
  "analysis": {
    "possible_diseases": [
      {
        "disease_id": "flu",
        "name": "流行性感冒",
        "confidence": 85.5,
        "matched_symptoms": ["高热", "头痛", "肌肉酸痛", "乏力"],
        "typical_matched": 3,
        "severity": "moderate",
        "description": "由流感病毒引起的急性呼吸道传染病",
        "treatment": "抗病毒药物、对症治疗、充分休息",
        "medications": ["奥司他韦", "布洛芬"]
      }
    ],
    "confidence": "high",  // high/medium/low/very_low
    "suggestions": [
      "症状高度符合流行性感冒特征",
      "建议检查项目以确诊"
    ],
    "additional_questions": [
      "症状是什么时候开始的？",
      "有无既往病史或家族史？"
    ],
    "warning": null  // 紧急情况警告
  }
}
```

### 2. 鉴别诊断

对比多个疾病的症状差异。

**请求：**
```http
POST /doctor/differential-diagnosis
Content-Type: application/json

{
  "disease_ids": ["common_cold", "flu"]
}
```

**响应：**
```json
{
  "status": "success",
  "comparison": {
    "diseases": ["普通感冒", "流行性感冒"],
    "comparison": [...],  // 症状对比表
    "key_points": [
      "流行性感冒特有：高热、肌肉酸痛",
      "普通感冒特有：流鼻涕、打喷嚏"
    ]
  }
}
```

---

## 💊 用药查询 API

### 1. 查询药品信息

**请求：**
```http
POST /medication/query
Content-Type: application/json

{
  "medication": "阿莫西林"
}
```

**响应：**
```json
{
  "status": "success",
  "medication": {
    "id": "amoxicillin",
    "name": "阿莫西林",
    "generic_name": "阿莫西林胶囊",
    "category": "抗生素（青霉素类）",
    "indications": ["呼吸道感染", "泌尿道感染"],
    "dosage": {
      "adult": "0.5g，每8小时一次",
      "child": "根据体重计算，20-40mg/kg/日"
    },
    "route": "口服",
    "contraindications": ["青霉素过敏者禁用"],
    "side_effects": ["恶心", "呕吐", "腹泻", "皮疹"],
    "precautions": ["过敏体质者慎用"],
    "storage": "密封，阴凉干燥处保存"
  }
}
```

### 2. 检查药物相互作用

**请求：**
```http
POST /medication/check-interactions
Content-Type: application/json

{
  "medications": ["二甲双胍", "布洛芬", "阿司匹林"]
}
```

**响应：**
```json
{
  "status": "success",
  "medications": ["二甲双胍", "布洛芬", "阿司匹林"],
  "safe": false,
  "warnings": [
    {
      "type": "interaction",
      "drugs": ["二甲双胍", "布洛芬"],
      "severity": "moderate",
      "description": "布洛芬等NSAIDs可能影响肾功能",
      "recommendation": "合用时应监测肾功能"
    },
    {
      "type": "interaction",
      "drugs": ["阿司匹林", "布洛芬"],
      "severity": "moderate",
      "description": "布洛芬可能降低阿司匹林的心血管保护作用",
      "recommendation": "注意服药时间间隔"
    }
  ]
}
```

### 3. 剂量建议

根据患者信息给出个性化剂量建议。

**请求：**
```http
POST /medication/dosage-recommendation
Content-Type: application/json

{
  "medication": "氨氯地平",
  "patient_info": {
    "age": 72,
    "weight": 65,
    "renal_function": "normal",  // normal/mild/moderate/severe
    "hepatic_function": "normal"
  }
}
```

**响应：**
```json
{
  "status": "success",
  "medication": "氨氯地平",
  "dosage": "5mg，每日一次",
  "route": "口服",
  "note": "成人常规剂量",
  "adjustments": [
    "老年人起始剂量应减半"
  ],
  "contraindications": [...],
  "precautions": [...]
}
```

### 4. 禁忌症检查

**请求：**
```http
POST /medication/check-contraindications
Content-Type: application/json

{
  "medication": "阿莫西林",
  "patient_info": {
    "allergies": ["青霉素"],
    "diseases": ["肾功能不全"],
    "pregnant": false,
    "breastfeeding": false
  }
}
```

**响应：**
```json
{
  "status": "success",
  "medication": "阿莫西林",
  "safe": false,
  "warnings": [
    {
      "severity": "high",
      "type": "allergy",
      "message": "患者对青霉素过敏，可能对阿莫西林过敏"
    }
  ]
}
```

### 5. 根据适应症搜索药品

**请求：**
```http
POST /medication/search-by-indication
Content-Type: application/json

{
  "indication": "发热"
}
```

**响应：**
```json
{
  "status": "success",
  "indication": "发热",
  "medications": [
    {
      "name": "布洛芬",
      "category": "解热镇痛药",
      "indications": ["发热", "头痛", "关节痛"]
    }
  ]
}
```

---

## 🏢 科室管理 API

### 查看所有科室

**请求：**
```http
GET /departments/list
```

**响应：**
```json
{
  "status": "success",
  "departments": [
    {
      "id": "emergency",
      "name": "急诊科",
      "description": "处理急危重症",
      "priority": 0
    },
    {
      "id": "internal",
      "name": "内科",
      "description": "诊治内脏器官非手术性疾病",
      "priority": 1
    }
  ]
}
```

---

## 🔧 系统管理 API

### 健康检查

**请求：**
```http
GET /health
```

**响应：**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-26T10:00:00",
  "modules": {
    "asr": true,
    "emotion": true,
    "speaker": true,
    "dialogue": true,
    "tts": true,
    "triage": true,
    "diagnosis": true,
    "medication": true
  }
}
```

---

## 📝 完整使用示例

### Python 示例

```python
import requests

SERVER_URL = "http://localhost:6007"

# 1. 患者导诊
def patient_triage_example():
    response = requests.post(
        f"{SERVER_URL}/patient/triage",
        json={
            "symptoms": ["发烧", "咳嗽", "头痛"],
            "age": 8,
            "gender": "男"
        }
    )
    result = response.json()
    dept = result['result']['recommended_department']
    print(f"推荐科室: {dept['name']}")

# 2. 医生辅助诊断
def doctor_diagnosis_example():
    response = requests.post(
        f"{SERVER_URL}/doctor/analyze-symptoms",
        json={
            "symptoms": ["高热", "肌肉酸痛", "乏力"],
            "patient_info": {"age": 35, "gender": "男"}
        }
    )
    result = response.json()
    diseases = result['analysis']['possible_diseases']
    print(f"可能疾病: {diseases[0]['name']}")
    print(f"置信度: {diseases[0]['confidence']}%")

# 3. 用药查询
def medication_query_example():
    response = requests.post(
        f"{SERVER_URL}/medication/query",
        json={"medication": "布洛芬"}
    )
    result = response.json()
    med = result['medication']
    print(f"药品: {med['name']}")
    print(f"适应症: {', '.join(med['indications'])}")

# 4. 药物相互作用检查
def interaction_check_example():
    response = requests.post(
        f"{SERVER_URL}/medication/check-interactions",
        json={"medications": ["二甲双胍", "布洛芬"]}
    )
    result = response.json()
    if not result['safe']:
        print("⚠️ 存在药物相互作用")
        for warning in result['warnings']:
            print(f"  {warning['description']}")

if __name__ == "__main__":
    patient_triage_example()
    doctor_diagnosis_example()
    medication_query_example()
    interaction_check_example()
```

### cURL 示例

```bash
# 患者导诊
curl -X POST http://localhost:6007/patient/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["发烧", "咳嗽"],
    "age": 8,
    "gender": "男"
  }'

# 症状分析
curl -X POST http://localhost:6007/doctor/analyze-symptoms \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["高热", "头痛", "肌肉酸痛"],
    "patient_info": {"age": 35}
  }'

# 查询药品
curl -X POST http://localhost:6007/medication/query \
  -H "Content-Type: application/json" \
  -d '{"medication": "布洛芬"}'

# 检查相互作用
curl -X POST http://localhost:6007/medication/check-interactions \
  -H "Content-Type: application/json" \
  -d '{"medications": ["二甲双胍", "布洛芬"]}'
```

---

## ⚠️ 注意事项

1. **免责声明**：本系统提供的建议仅供参考，不能替代专业医生的诊断和治疗
2. **数据安全**：患者隐私数据应加密存储和传输
3. **紧急情况**：系统检测到紧急症状时，会提示立即就医
4. **知识库更新**：医疗知识库需要定期更新以保证准确性

---

## 🚀 快速测试

运行测试脚本：

```bash
python test_medical_features.py
```

该脚本会自动测试所有医疗功能并输出结果。
