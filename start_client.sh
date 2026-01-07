#!/bin/bash
# 客户端启动脚本

cd "$(dirname "$0")/client"

echo "========================================="
echo "  医疗语音助手 - 客户端"
echo "========================================="
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 检查服务器是否运行
echo "检查服务器连接..."
if curl -s http://localhost:6008/health > /dev/null 2>&1; then
    echo "✓ 服务器已连接"
else
    echo "⚠ 警告: 服务器未响应，请确保服务器已启动"
    echo "  运行: ./start_server.sh"
    echo ""
fi

echo ""
echo "启动客户端..."
echo "输入 'help' 查看可用命令"
echo ""

python main.py
