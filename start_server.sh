#!/bin/bash
# 服务器启动脚本

cd "$(dirname "$0")/server"

echo "========================================="
echo "  医疗语音助手 - 服务器"
echo "========================================="
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

echo "启动服务器..."
echo "端口: 6007"
echo ""

python app.py
