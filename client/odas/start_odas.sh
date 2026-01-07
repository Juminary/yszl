#!/bin/bash
# ODAS 安装与启动脚本 for Raspberry Pi 5
# 
# 此脚本用于:
# 1. 检查/安装 ODAS 依赖
# 2. 编译 ODAS (如果需要)
# 3. 启动 odaslive 进程
#
# 使用方法:
#   chmod +x start_odas.sh
#   ./start_odas.sh [install|start|stop|status]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ODAS_DIR="$SCRIPT_DIR/odas_build"
CONFIG_FILE="$SCRIPT_DIR/respeaker_6mic.cfg"
PID_FILE="/tmp/odas.pid"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# 安装依赖
# ============================================================
install_dependencies() {
    log_info "Installing ODAS dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        cmake \
        libfftw3-dev \
        libconfig-dev \
        libasound2-dev \
        git
    
    log_info "Dependencies installed."
}

# ============================================================
# 编译 ODAS
# ============================================================
build_odas() {
    log_info "Building ODAS..."
    
    if [ -d "$ODAS_DIR" ]; then
        log_warn "ODAS directory exists, skipping clone."
    else
        git clone https://github.com/introlab/odas.git "$ODAS_DIR"
    fi
    
    cd "$ODAS_DIR"
    mkdir -p build && cd build
    
    # 配置编译选项 (ARM NEON 优化)
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    # 编译 (使用多核)
    make -j$(nproc)
    
    log_info "ODAS built successfully!"
    log_info "Binary location: $ODAS_DIR/build/bin/odaslive"
}

# ============================================================
# 启动 ODAS
# ============================================================
start_odas() {
    ODASLIVE="$ODAS_DIR/build/bin/odaslive"
    
    if [ ! -f "$ODASLIVE" ]; then
        log_error "odaslive not found at $ODASLIVE"
        log_error "Run '$0 install' first."
        exit 1
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # 检查是否已在运行
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            log_warn "ODAS already running with PID $OLD_PID"
            return
        fi
    fi
    
    # 确定 ALSA 设备号
    CARD_ID=$(arecord -l | grep -i "seeed" | sed 's/card \([0-9]\+\):.*/\1/' | head -1)
    if [ -z "$CARD_ID" ]; then
        log_error "ReSpeaker sound card not found!"
        log_error "Run 'arecord -l' to check available devices."
        exit 1
    fi
    
    log_info "Detected ReSpeaker card ID: $CARD_ID"
    
    # 动态更新配置文件中的 card ID (临时文件)
    RUNTIME_CONFIG="/tmp/odas_runtime.cfg"
    sed "s/card = [0-9]\+;/card = $CARD_ID;/" "$CONFIG_FILE" > "$RUNTIME_CONFIG"
    
    # 后台启动 ODAS
    log_info "Starting ODAS..."
    nohup "$ODASLIVE" -c "$RUNTIME_CONFIG" > /tmp/odas.log 2>&1 &
    echo $! > "$PID_FILE"
    
    sleep 2
    
    if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        log_info "ODAS started with PID $(cat $PID_FILE)"
        log_info "SST output: tcp://127.0.0.1:9000 (Waiting for Python client to listen)"
        log_info "SSS output: tcp://127.0.0.1:9001"
        log_info "Log file: /tmp/odas.log"
    else
        log_error "ODAS failed to start. Check /tmp/odas.log"
        # 尝试显示最后几行日志帮助诊断
        if [ -f /tmp/odas.log ]; then
            log_error "Last lines of /tmp/odas.log:"
            tail -n 5 /tmp/odas.log
        fi
        exit 1
    fi
}

# ============================================================
# 停止 ODAS
# ============================================================
stop_odas() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_info "Stopping ODAS (PID $PID)..."
            kill "$PID"
            rm "$PID_FILE"
            log_info "ODAS stopped."
        else
            log_warn "ODAS not running."
            rm "$PID_FILE"
        fi
    else
        log_warn "PID file not found."
    fi
}

# ============================================================
# 状态检查
# ============================================================
status_odas() {
    echo "=== ODAS Status ==="
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "Status: ${GREEN}Running${NC} (PID $PID)"
        else
            echo -e "Status: ${RED}Not running${NC} (stale PID file)"
        fi
    else
        echo -e "Status: ${YELLOW}Not started${NC}"
    fi
    
    echo ""
    echo "=== ReSpeaker Sound Card ==="
    arecord -l | grep -i seeed || echo "Not detected"
    
    echo ""
    echo "=== Socket Ports ==="
    netstat -tlnp 2>/dev/null | grep -E "9000|9001" || echo "No listeners"
}

# ============================================================
# 主入口
# ============================================================
case "${1:-help}" in
    install)
        install_dependencies
        build_odas
        ;;
    start)
        start_odas
        ;;
    stop)
        stop_odas
        ;;
    restart)
        stop_odas
        sleep 1
        start_odas
        ;;
    status)
        status_odas
        ;;
    *)
        echo "Usage: $0 {install|start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  install  - Install dependencies and build ODAS"
        echo "  start    - Start ODAS daemon"
        echo "  stop     - Stop ODAS daemon"
        echo "  restart  - Restart ODAS daemon"
        echo "  status   - Check ODAS status"
        exit 1
        ;;
esac
