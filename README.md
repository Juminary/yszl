# 🎙️ 智能语音助手系统

> 基于深度学习的端到端语音对话系统，支持语音识别、情感分析、声纹识别、智能对话和语音合成

## ✨ 功能特性

| 模块 | 模型 | 功能描述 |
|------|------|----------|
| **语音识别 (ASR)** | Paraformer-Large | 高精度中文语音转文本 |
| **情感识别** | SenseVoice | 识别6种情感：中性、开心、悲伤、愤怒、恐惧、惊讶 |
| **声纹识别** | CAM++ | 说话人身份识别与注册 |
| **智能对话** | Qwen2.5-0.5B-Instruct | 基于大语言模型的智能问答 |
| **语音合成 (TTS)** | CosyVoice | 高质量中文语音合成 |

## 🏗️ 系统架构

```
voice_assistant/
├── server/                 # 服务端
│   ├── app.py             # Flask 主应用
│   └── modules/           # 功能模块
│       ├── asr.py         # 语音识别模块
│       ├── emotion.py     # 情感识别模块
│       ├── speaker.py     # 声纹识别模块
│       ├── dialogue.py    # 对话系统模块
│       └── tts.py         # 语音合成模块
├── client/                 # 客户端
│   ├── main.py            # 客户端主程序
│   ├── audio_capture.py   # 音频采集
│   └── audio_player.py    # 音频播放
├── config/
│   └── config.yaml        # 配置文件
├── models/                 # 模型存储目录
├── data/                   # 数据目录（声纹数据库等）
├── logs/                   # 日志目录
└── temp/                   # 临时文件目录
```

## 🚀 部署指南

### 环境要求

| 项目 | 要求 |
|------|------|
| **Python** | 3.10 - 3.12（推荐 3.12） |
| **操作系统** | macOS / Linux / Windows |
| **内存** | 最低 8GB，推荐 16GB+ |
| **磁盘空间** | 约 5GB（模型和索引文件） |
| **网络** | 首次运行需要下载模型（约 3GB） |

---

### 步骤 1：克隆项目

```bash
git clone <项目地址>
cd voice_assistant
```

---

### 步骤 2：创建 Python 虚拟环境

**macOS / Linux：**
```bash
# 使用 Python 3.12 创建虚拟环境
python3.12 -m venv venv312

# 激活虚拟环境
source venv312/bin/activate

# 确认 Python 版本
python --version  # 应显示 Python 3.12.x
```

**Windows：**
```powershell
# 创建虚拟环境
python -m venv venv312

# 激活虚拟环境
venv312\Scripts\activate

# 确认 Python 版本
python --version
```

---

### 步骤 3：安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

**⚠️ 常见问题：**

1. **PyAudio 安装失败（macOS）**：
   ```bash
   brew install portaudio
   pip install pyaudio
   ```

2. **PyAudio 安装失败（Ubuntu/Debian）**：
   ```bash
   sudo apt-get install python3-pyaudio portaudio19-dev
   pip install pyaudio
   ```

3. **网络问题导致下载慢**：使用国内镜像源
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

---

### 步骤 4：下载模型

模型会在首次运行时自动下载，也可以预先下载：

```bash
cd server
python download_models.py
```

按提示选择：
- `s` - 仅下载缺失的模型
- `a` - 下载所有模型

**模型列表：**

| 模型 | 大小 | 来源 | 用途 |
|------|------|------|------|
| Paraformer-Large | ~1GB | ModelScope | 语音识别 (ASR) |
| SenseVoice | ~200MB | ModelScope | 情感识别 |
| CAM++ | ~100MB | ModelScope | 声纹识别 |
| Qwen2.5-0.5B | ~1GB | HuggingFace | 对话生成 (LLM) |
| bge-small-zh | ~100MB | HuggingFace | RAG 文本向量化 |

**⚠️ 如果 HuggingFace 下载慢**：设置镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

### 步骤 5：构建 RAG 索引（可选）

如果项目中未包含 RAG 索引文件，需要手动构建：

```bash
cd server
python build_rag_index.py
```

索引构建完成后会在 `server/data/rag_index/` 目录下生成：
- `index.faiss` - FAISS 向量索引
- `documents.json` - 文档内容

---

### 步骤 6：配置知识图谱（可选）

知识图谱功能需要 Neo4j 数据库支持：

**安装 Neo4j：**

```bash
# macOS (Homebrew)
brew install neo4j

# Ubuntu/Debian
sudo apt-get install neo4j

# 或下载 Neo4j Desktop
# https://neo4j.com/download/
```

**启动 Neo4j：**
```bash
neo4j start
```

**配置连接信息**（编辑 `config/config.yaml`）：
```yaml
knowledge_graph:
  enabled: true
  host: "localhost"
  port: 7474
  user: "neo4j"
  password: "your_password"  # 修改为你的密码
```

**导入医学知识图谱数据：**
```bash
python build_medicalgraph.py
```

---

### 步骤 7：启动服务

**方式一：使用启动脚本（推荐）**

```bash
# 终端1：启动服务器
./start_server.sh

# 终端2：启动客户端
./start_client.sh
```

**方式二：手动启动**

```bash
# 终端1：启动服务器
source venv312/bin/activate
cd server
python app.py

# 终端2：启动客户端
source venv312/bin/activate
cd client
python main.py
```

**✅ 启动成功标志：**
```
📚 [RAG] 成功加载索引
   - 文档数量: 177703
   - 向量数量: 177703

🔗 [知识图谱] 连接成功
   - 地址: bolt://localhost:7687
   - 词典: 44093 词条

 * Running on http://127.0.0.1:6007
```

---

### 步骤 8：使用客户端

客户端启动后，可使用以下命令：

| 命令 | 功能 | 示例 |
|------|------|------|
| `talk` | 开始语音对话 | 输入后对着麦克风说话 |
| `register <ID>` | 注册声纹 | `register 张三` |
| `list` | 列出已注册声纹 | - |
| `history` | 查看对话历史 | - |
| `clear` | 清除对话历史 | - |
| `help` | 显示帮助信息 | - |
| `quit` | 退出客户端 | - |

---

### 目录结构说明

```
voice_assistant/
├── server/                    # 服务端代码
│   ├── app.py                # Flask 主应用
│   ├── modules/              # 功能模块
│   ├── models/               # 下载的模型文件
│   ├── data/                 # 数据文件
│   │   ├── rag_index/       # RAG 向量索引
│   │   ├── dict/            # 医学词典
│   │   └── speaker_db.pkl   # 声纹数据库
│   └── logs/                 # 日志文件
├── client/                    # 客户端代码
├── config/
│   └── config.yaml           # 配置文件
├── requirements.txt           # Python 依赖
└── README.md
```

## 📡 API 接口

### 健康检查
```bash
GET /health
```

### 语音识别
```bash
POST /asr
Content-Type: multipart/form-data
Body: audio=<音频文件>
```

### 情感识别
```bash
POST /emotion
Content-Type: multipart/form-data
Body: audio=<音频文件>
```

### 声纹注册
```bash
POST /speaker/register
Content-Type: multipart/form-data
Body: audio=<音频文件>, speaker_id=<说话人ID>
```

### 声纹识别
```bash
POST /speaker/recognize
Content-Type: multipart/form-data
Body: audio=<音频文件>
```

### 智能对话
```bash
POST /dialogue
Content-Type: application/json
Body: {"query": "你好", "session_id": "user1"}
```

### 语音合成
```bash
POST /tts
Content-Type: application/json
Body: {"text": "你好，很高兴认识你"}
```

### 完整对话流程
```bash
POST /chat
Content-Type: multipart/form-data
Body: audio=<音频文件>, session_id=<会话ID>
```

一次请求完成：语音识别 → 情感识别 → 声纹识别 → 对话生成 → 语音合成

## ⚙️ 配置说明

编辑 `config/config.yaml` 自定义配置：

```yaml
# 服务器配置
server:
  host: "0.0.0.0"
  port: 5001

# 计算设备：cpu / cuda / mps
asr:
  device: "cpu"

# 对话系统提示词
dialogue:
  system_prompt: |
    你是一个智能语音助手...
```

## 📦 依赖说明

| 依赖 | 用途 |
|------|------|
| Flask | Web 服务框架 |
| FunASR | 语音识别引擎 (Paraformer + SenseVoice) |
| ModelScope | 模型下载 (CosyVoice + CAM++) |
| Transformers | 大语言模型 (Qwen2.5) |
| PyTorch | 深度学习框架 |
| librosa | 音频处理 |

## 🔧 常见问题

**Q: 模型下载失败？**  
A: 检查网络连接，或设置镜像源：
```bash
export MODELSCOPE_CACHE=./models
```

**Q: macOS 上没有声音输出？**  
A: 系统会自动回退到 macOS 内置的 `say` 命令进行语音合成

**Q: 如何使用 GPU 加速？**  
A: 修改 `config.yaml` 中的 `device` 为 `cuda`（NVIDIA）或 `mps`（Apple Silicon）

## 📄 许可证

本项目仅供学习研究使用。

---

**Made with ❤️ for Voice AI**
# yszl
