# =============================================
# 智能语音助手系统 - 开发环境镜像
# 只包含环境，代码通过 volume 挂载
# =============================================

FROM python:3.12-slim

WORKDIR /app

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 创建必要目录
RUN mkdir -p logs server/logs server/temp server/data

EXPOSE 6007

# 工作目录设置为 server
WORKDIR /app/server

CMD ["python", "app.py"]
