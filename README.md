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

## 🚀 快速开始

### 环境要求

- Python 3.10+
- macOS / Linux / Windows
- 推荐 8GB+ 内存

### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动服务器

```bash
# 方式一：使用启动脚本
./start_server.sh

# 方式二：手动启动
source venv/bin/activate
cd server && python app.py
```

服务器默认运行在 `http://localhost:5001`

### 3. 启动客户端

```bash
# 方式一：使用启动脚本
./start_client.sh

# 方式二：手动启动
source venv/bin/activate
cd client && python main.py
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
