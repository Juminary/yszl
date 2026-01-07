# 🏥 智能医疗语音助手系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> 基于深度学习的端到端医疗语音对话系统，集成语音识别、情感分析、声纹识别、智能对话、知识图谱和语音合成，专为医疗场景设计。

---

## ✨ 功能特性

### 🎯 核心 AI 能力

| 模块 | 模型 | 功能描述 |
|:---:|:---:|:---|
| **语音识别** | SenseVoice | 支持50+语言/方言，自动语言检测，同时输出情感和音频事件 |
| **情感识别** | SenseVoice | 识别6种情感：中性、开心、悲伤、愤怒、恐惧、惊讶 |
| **声纹识别** | CAM++ | 说话人身份识别、声纹注册、音色克隆 |
| **智能对话** | Qwen2.5-0.5B | 基于大语言模型的医疗问答，支持 RAG 知识增强 |
| **语音合成** | CosyVoice-300M | 高质量中文语音合成，支持流式输出、音色克隆 |
| **知识检索** | BGE + FAISS | 17万+医疗文档向量检索 |
| **知识图谱** | Neo4j | 医学实体关系查询，支持智能问答 |

### 🏥 医疗专业功能

| 功能 | 患者端 | 医生端 | 描述 |
|:---:|:---:|:---:|:---|
| **智能导诊** | ✅ | - | 根据症状推荐科室和医生 |
| **辅助诊断** | - | ✅ | 症状分析、疾病推断、鉴别诊断 |
| **用药查询** | ✅ | ✅ | 药品信息、相互作用、剂量建议 |
| **SOAP病历** | - | ✅ | 自动从对话生成结构化病历 |
| **急救检测** | ✅ | ✅ | 识别危急情况并发出警报 |
| **语音交互** | ✅ | ✅ | 全程语音操作，老年人友好 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      智能医疗语音助手系统                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐                        ┌──────────────────────┐ │
│   │   客户端      │  ◄──── HTTP/SSE ────►  │      服务端           │ │
│   │  (树莓派等)   │                        │   (Flask Server)     │ │
│   └──────────────┘                        └──────────────────────┘ │
│          │                                          │              │
│          ▼                                          ▼              │
│   ┌──────────────┐                        ┌──────────────────────┐ │
│   │ • 音频采集    │                        │    核心 AI 模块       │ │
│   │ • VAD 检测   │                        │  ┌────────────────┐  │ │
│   │ • 唤醒词检测  │                        │  │ ASR (语音识别)  │  │ │
│   │ • 音频播放    │                        │  │ TTS (语音合成)  │  │ │
│   │ • 流式播放    │                        │  │ LLM (对话生成)  │  │ │
│   └──────────────┘                        │  │ 情感/声纹识别   │  │ │
│                                           │  │ RAG 知识检索    │  │ │
│                                           │  └────────────────┘  │ │
│                                           └──────────────────────┘ │
│                                                     │              │
│                                                     ▼              │
│                                           ┌──────────────────────┐ │
│                                           │    医疗专业模块       │ │
│                                           │  ┌────────────────┐  │ │
│                                           │  │ 患者导诊服务    │  │ │
│                                           │  │ 医生辅助诊断    │  │ │
│                                           │  │ 用药查询管理    │  │ │
│                                           │  │ 知识图谱 Neo4j  │  │ │
│                                           │  │ SOAP 病历生成   │  │ │
│                                           │  └────────────────┘  │ │
│                                           └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
yszl/
├── client/                          # 客户端代码
│   ├── main.py                     # 主程序入口
│   ├── audio_capture.py            # 音频采集（含 VAD）
│   ├── audio_player.py             # 音频播放（含流式）
│   ├── wakeword_detector.py        # 唤醒词检测
│   ├── event_bus.py                # 事件总线
│   ├── fullduplex_controller.py    # 全双工控制器
│   └── acoustic_frontend/          # 声学前端（麦克风阵列）
│       ├── mic_array.py            # 麦克风阵列驱动
│       ├── beamformer.py           # 波束成形
│       ├── doa.py                  # 声源定位
│       └── aec.py                  # 回声消除
│
├── server/                          # 服务端代码
│   ├── app.py                      # Flask 主应用（API 入口）
│   ├── modules/                    # 功能模块
│   │   ├── core/                   # 核心 AI 模块
│   │   │   ├── asr.py             # 语音识别 (SenseVoice)
│   │   │   ├── tts.py             # 语音合成 (CosyVoice)
│   │   │   ├── dialogue.py        # 对话生成 (Qwen2.5)
│   │   │   ├── gguf_dialogue.py   # GGUF 量化模型支持
│   │   │   └── rag.py             # RAG 向量检索
│   │   ├── audio/                  # 音频分析模块
│   │   │   ├── emotion.py         # 情感识别
│   │   │   ├── speaker.py         # 声纹识别 (CAM++)
│   │   │   ├── paralinguistic.py  # 副语言分析
│   │   │   └── sound_event.py     # 音频事件检测
│   │   ├── medical/                # 医疗业务模块
│   │   │   ├── triage.py          # 患者导诊服务
│   │   │   ├── diagnosis_assistant.py  # 辅助诊断
│   │   │   ├── medication.py      # 用药查询管理
│   │   │   ├── medical_dict.py    # 医学词典
│   │   │   └── intent_classifier.py    # 意图分类
│   │   ├── knowledge/              # 知识模块
│   │   │   ├── knowledge_graph.py # Neo4j 知识图谱
│   │   │   └── cypher_generator.py # Cypher 查询生成
│   │   └── aci/                    # 临床智能模块 (ACI)
│   │       ├── consultation_session.py  # 会诊会话管理
│   │       ├── clinical_entity_extractor.py  # 临床实体提取
│   │       ├── soap_generator.py  # SOAP 病历生成
│   │       ├── hallucination_detector.py  # 幻觉检测
│   │       ├── emergency_detector.py  # 急救检测
│   │       └── speaker_diarization.py  # 说话人分离
│   ├── data/                       # 数据文件
│   │   ├── rag_index/             # RAG 向量索引
│   │   │   ├── index.faiss        # FAISS 索引文件
│   │   │   └── documents.json     # 文档内容
│   │   ├── dict/                  # 医学词典
│   │   │   ├── disease.txt        # 疾病词典
│   │   │   ├── symptom.txt        # 症状词典
│   │   │   ├── drug.txt           # 药物词典
│   │   │   └── ...
│   │   ├── voice_clones/          # 音色克隆文件
│   │   ├── hospital.db            # 医院数据库 (SQLite)
│   │   └── speaker_db.pkl         # 声纹数据库
│   ├── models/                     # AI 模型文件
│   │   ├── asr/                   # 语音识别模型
│   │   ├── tts/                   # 语音合成模型
│   │   ├── dialogue/              # 对话模型
│   │   ├── embedding/             # 向量模型
│   │   └── speaker/               # 声纹模型
│   ├── libs/                       # 第三方库
│   │   └── cosyvoice/             # CosyVoice 源码
│   ├── static/                     # Web 静态文件
│   │   ├── index.html             # 主页面
│   │   ├── consultation.html      # 会诊页面
│   │   ├── app.js                 # 前端逻辑
│   │   └── style.css              # 样式
│   ├── utils/                      # 工具函数
│   ├── logs/                       # 日志文件
│   └── temp/                       # 临时文件
│
├── config/
│   └── config.yaml                 # 系统配置文件
│
├── docs/                           # 项目文档
│   ├── API文档_医疗功能.md
│   ├── 医疗功能实现总结.md
│   └── 演示指南.md
│
├── requirements.txt                # Python 依赖（精简版）
├── requirements_full.txt           # Python 依赖（完整版）
├── Dockerfile                      # Docker 构建文件
├── docker-compose.yml              # Docker Compose 配置
├── start_server.sh                 # 服务端启动脚本
├── start_client.sh                 # 客户端启动脚本
└── README.md                       # 项目说明文档
```

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
|:---|:---|
| **Python** | 3.10 - 3.12（推荐 3.12） |
| **操作系统** | Linux / macOS / Windows |
| **内存** | 最低 8GB，推荐 16GB+ |
| **显存** | 推荐 8GB+ GPU（可选，支持 CPU 运行） |
| **磁盘空间** | 约 10GB（模型和索引文件） |

### 步骤 1：克隆项目

```bash
git clone <项目地址>
cd yszl
```

### 步骤 2：创建虚拟环境

```bash
# 创建虚拟环境
python3.12 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 确认 Python 版本
python --version
```

### 步骤 3：安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装依赖（使用国内镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<details>
<summary>💡 常见问题解决</summary>

**PyAudio 安装失败（macOS）**
```bash
brew install portaudio
pip install pyaudio
```

**PyAudio 安装失败（Ubuntu/Debian）**
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
pip install pyaudio
```

**HuggingFace 下载慢**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

</details>

### 步骤 4：配置系统

编辑 `config/config.yaml`，根据需要修改配置：

```yaml
# 服务器配置
server:
  host: "0.0.0.0"
  port: 6008

# 设备配置（cuda/cpu/mps）
asr:
  device: "cuda"    # GPU 加速
dialogue:
  device: "cpu"     # 对话模型
tts:
  device: "cuda"    # TTS 模型

# 知识图谱（可选）
knowledge_graph:
  enabled: true
  host: "localhost"
  port: 7474
  user: "neo4j"
  password: "your_password"
```

### 步骤 5：启动服务

**方式一：使用启动脚本（推荐）**

```bash
# 终端 1：启动服务器
./start_server.sh

# 终端 2：启动客户端
./start_client.sh
```

**方式二：手动启动**

```bash
# 终端 1：启动服务器
cd server
python app.py

# 终端 2：启动客户端
cd client
python main.py
```

### 步骤 6：验证启动

服务启动成功后，您将看到：

```
📚 [RAG] 成功加载索引
   - 文档数量: 177703
   - 向量数量: 177703

🔗 [知识图谱] 连接成功
   - 地址: bolt://localhost:7687
   - 词典: 44093 词条

 * Running on http://0.0.0.0:6008
```

访问 `http://localhost:6008` 可打开 Web 界面。

---

## 📡 API 接口

### 基础接口

| 接口 | 方法 | 描述 |
|:---|:---:|:---|
| `/health` | GET | 健康检查，返回各模块状态 |
| `/info` | GET | 获取系统信息 |
| `/events` | GET | SSE 事件流（实时消息推送） |

### 语音处理接口

| 接口 | 方法 | 描述 |
|:---|:---:|:---|
| `/asr` | POST | 语音识别，返回文本和情感 |
| `/tts` | POST | 语音合成，返回音频文件 |
| `/tts/stream` | POST | 流式语音合成 |
| `/emotion` | POST | 情感识别 |

### 声纹接口

| 接口 | 方法 | 描述 |
|:---|:---:|:---|
| `/speaker/register` | POST | 注册声纹（同时注册音色克隆） |
| `/speaker/recognize` | POST | 声纹识别 |
| `/speaker/list` | GET | 列出已注册说话人 |
| `/voice-clone/list` | GET | 列出可用音色克隆 |

### 对话接口

| 接口 | 方法 | 描述 |
|:---|:---:|:---|
| `/dialogue` | POST | 文本对话（支持模式切换） |
| `/chat` | POST | 完整对话流程（ASR→情感→声纹→对话→TTS） |

### 医疗接口

| 接口 | 方法 | 描述 |
|:---|:---:|:---|
| `/patient/triage` | POST | 患者智能导诊 |
| `/doctor/analyze-symptoms` | POST | 医生辅助诊断 |
| `/medication/query` | POST | 药品信息查询 |
| `/medication/check-interactions` | POST | 药物相互作用检查 |
| `/departments/list` | GET | 获取科室列表 |

### ACI 临床智能接口

| 接口 | 方法 | 描述 |
|:---|:---:|:---|
| `/consultation/start` | POST | 开始会诊会话 |
| `/consultation/<id>/utterance` | POST | 添加对话记录 |
| `/consultation/<id>/soap` | GET | 获取 SOAP 病历 |
| `/consultation/<id>/end` | POST | 结束会诊 |
| `/aci/generate-soap` | POST | 从对话生成 SOAP 病历 |
| `/emergency/assess` | POST | 急救风险评估 |

### 使用示例

<details>
<summary>📝 完整对话流程</summary>

```bash
# 发送音频进行完整对话
curl -X POST http://localhost:6008/chat \
  -F "audio=@test.wav" \
  -F "session_id=user001" \
  -F "mode=patient"
```

</details>

<details>
<summary>🏥 患者导诊</summary>

```bash
curl -X POST http://localhost:6008/patient/triage \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我头疼了三天，还有点发烧",
    "age": 35,
    "gender": "男"
  }'
```

</details>

<details>
<summary>💊 药品查询</summary>

```bash
curl -X POST http://localhost:6008/medication/query \
  -H "Content-Type: application/json" \
  -d '{"medication": "阿莫西林"}'
```

</details>

<details>
<summary>📋 生成 SOAP 病历</summary>

```bash
curl -X POST http://localhost:6008/aci/generate-soap \
  -H "Content-Type: application/json" \
  -d '{
    "utterances": [
      {"speaker": "医生", "text": "您哪里不舒服？"},
      {"speaker": "患者", "text": "我头疼了三天，还有点发烧。"},
      {"speaker": "医生", "text": "体温多少度？"},
      {"speaker": "患者", "text": "三十八度五。"}
    ]
  }'
```

</details>

---

## 🎤 客户端使用

启动客户端后，可使用以下命令：

| 命令 | 功能 | 说明 |
|:---|:---|:---|
| `talk` / `t` | 连续语音对话 | 推荐使用，支持音色选择 |
| `chat` / `c` | 单次语音对话 | 说一句话后返回 |
| `dia` / `d` | 连续文字对话 | 文字输入模式 |
| `tchat` / `tc` | TTS+ASR 测试 | 文字转语音后发送服务器 |
| `register` / `r` | 注册声纹 | 同时注册音色克隆 |
| `speakers` / `s` | 查看说话人 | 列出已注册的声纹 |
| `quit` / `q` | 退出 | 退出客户端 |

### 语音命令

在对话过程中，可以使用语音命令切换模式：

| 语音命令 | 功能 |
|:---|:---|
| "切换到患者模式" / "我是患者" | 切换到患者模式（导诊） |
| "切换到医生模式" / "我是医生" | 切换到医生模式（辅助诊断） |
| "切换到会诊模式" / "开始会诊" | 切换到会诊模式（病历生成） |
| "结束会诊" / "生成病历" | 结束会诊并生成 SOAP 病历 |
| "切换音色" | 选择不同的音色克隆 |

---

## 🔧 高级配置

### 知识图谱配置（Neo4j）

1. **安装 Neo4j**

```bash
# macOS
brew install neo4j

# Ubuntu/Debian
sudo apt-get install neo4j

# 或下载 Neo4j Desktop
# https://neo4j.com/download/
```

2. **启动 Neo4j**

```bash
neo4j start
```

3. **导入医学知识图谱**

```bash
cd server
python build_medicalgraph.py
```

4. **配置连接信息**

```yaml
# config/config.yaml
knowledge_graph:
  enabled: true
  host: "localhost"
  port: 7474
  user: "neo4j"
  password: "your_password"
```

### RAG 索引构建

如果需要重新构建 RAG 索引：

```bash
cd server
python build_rag_index.py
```

### 声学前端配置（麦克风阵列）

支持 ReSpeaker 6-Mic 环形麦克风阵列：

```yaml
# config/config.yaml
acoustic_frontend:
  enabled: true
  device:
    type: "respeaker_6mic"
    sample_rate: 16000
  doa:
    enabled: true
    algorithm: "gcc_phat"
  beamforming:
    enabled: true
    method: "das"
  aec:
    enabled: true
```

---

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t medical-voice-assistant .

# 启动服务
docker-compose up -d
```

---

## 📊 性能指标

| 指标 | 数值 | 说明 |
|:---|:---|:---|
| ASR 延迟 | < 500ms | 语音识别响应时间 |
| TTS 首音频延迟 | < 1s | 流式模式下首音频延迟 |
| 对话响应 | < 2s | 完整对话流程响应时间 |
| RAG 检索 | < 100ms | 向量检索响应时间 |
| 知识图谱查询 | < 200ms | Neo4j 查询响应时间 |

---

## 🛠️ 技术栈

| 类别 | 技术 |
|:---|:---|
| **后端框架** | Flask, Flask-CORS |
| **深度学习** | PyTorch, Transformers |
| **语音识别** | FunASR (SenseVoice) |
| **语音合成** | CosyVoice (ModelScope) |
| **大语言模型** | Qwen2.5-0.5B-Instruct |
| **向量检索** | FAISS, Sentence-Transformers (BGE) |
| **知识图谱** | Neo4j, py2neo |
| **音频处理** | librosa, soundfile, PyAudio |
| **数据库** | SQLite, Neo4j |
| **实体识别** | AC自动机 (pyahocorasick) |

---

## 📝 更新日志

### v1.0.0 (2025-01-07)
- ✅ 完整的语音对话流程
- ✅ 患者导诊服务
- ✅ 医生辅助诊断
- ✅ 用药查询管理
- ✅ Neo4j 知识图谱集成
- ✅ SOAP 病历自动生成
- ✅ 音色克隆支持
- ✅ 流式 TTS 输出
- ✅ Web 界面

---

## 🗺️ 路线图

### 短期计划
- [ ] 多轮问诊对话优化
- [ ] 健康档案管理
- [ ] 用药提醒功能
- [ ] 更多疾病和药品数据

### 中期计划
- [ ] 影像学报告解读
- [ ] 检验结果分析
- [ ] 慢性病管理
- [ ] 移动端 App

### 长期计划
- [ ] 电子病历系统对接
- [ ] 远程医生咨询
- [ ] 多模态诊断（图像+语音）
- [ ] 数据可视化分析

---

## ⚠️ 免责声明

1. **医疗免责**：本系统仅供辅助参考，不能替代专业医生的诊断和治疗。
2. **数据安全**：实际部署时需要加密存储患者数据，符合医疗数据保护法规。
3. **紧急情况**：系统会识别危急症状并提示立即就医，但不能替代急救服务。
4. **持续更新**：医疗知识库需要定期更新以保证准确性。

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

---

<p align="center">
  <b>Made with ❤️ for Medical AI</b>
</p>
