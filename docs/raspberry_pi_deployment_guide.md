# 树莓派部署与测试指南

> 医声智联声学前端 - Raspberry Pi 5 + ReSpeaker 6-Mic 完整部署手册

---

## 一、硬件准备

### 1.1 必需硬件

| 硬件 | 型号 | 备注 |
|-----|------|-----|
| 主板 | Raspberry Pi 5 | 推荐 8GB 内存版 |
| 麦克风阵列 | ReSpeaker 6-Mic Circular Array Kit | Seeed Studio |
| SD卡 | 32GB+ Class 10 | 推荐 64GB |
| 电源 | 5V/5A USB-C | 官方电源适配器 |
| 扬声器 | 3.5mm 或 JST 接口 | 用于TTS播放 |

### 1.2 硬件连接

```
┌──────────────────────────────────────┐
│           ReSpeaker HAT              │
│  ┌──────────────────────────────┐    │
│  │     ● ● ● ● ● ●              │    │──── 排线连接到麦克风阵列板
│  │     6麦克风 LED环             │    │
│  └──────────────────────────────┘    │
│                                      │
│  40-Pin GPIO ─────────────────────   │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│         Raspberry Pi 5               │
│                                      │
│  [USB-C 电源] [HDMI] [USB] [网口]    │
└──────────────────────────────────────┘
```

**安装步骤**:
1. 关闭树莓派电源
2. 将 ReSpeaker HAT 对准 40-Pin GPIO 排针
3. 轻轻按压直到完全插入
4. 连接排线到圆形麦克风阵列板
5. 接入电源启动

---

## 二、系统安装

### 2.1 烧录系统

```bash
# 在电脑上使用 Raspberry Pi Imager
# 选择: Raspberry Pi OS (64-bit) Bookworm
# 高级设置中启用 SSH 和 WiFi
```

### 2.2 首次启动配置

```bash
# SSH 连接
ssh pi@raspberrypi.local

# 更新系统
sudo apt update && sudo apt upgrade -y

# 启用 I2C 和 SPI (ReSpeaker 需要)
sudo raspi-config
# -> Interface Options -> I2C -> Enable
# -> Interface Options -> SPI -> Enable
# -> 重启
```

---

## 三、ReSpeaker 驱动安装

### 3.1 安装依赖

```bash
sudo apt install -y \
    git \
    build-essential \
    raspberrypi-kernel-headers \
    dkms \
    i2c-tools \
    libasound2-plugins
```

### 3.2 安装驱动 (HinTak 分支)

```bash
# 专为 Kernel 6.6 和 Pi 5 适配的分支
git clone -b v6.6 https://github.com/HinTak/seeed-voicecard/
cd seeed-voicecard
sudo ./install.sh
```

### 3.3 重启并验证

```bash
sudo reboot

# 重启后验证
arecord -l
```

**预期输出**:
```
**** List of CAPTURE Hardware Devices ****
card 3: seeed8micvoicec [seeed-8mic-voicecard], device 0: ...
```

### 3.4 测试录音

```bash
# 录制 5 秒测试音频
arecord -D plughw:3,0 -c 8 -r 16000 -f S32_LE -d 5 test.wav

# 播放测试 (需要连接扬声器)
aplay test.wav
```

### 3.5 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 设备未识别 | 驱动未加载 | `dmesg \| grep ac108` 检查日志 |
| 录音全零 | I2C 通信失败 | `i2cdetect -y 1` 检查地址 |
| Card ID 错误 | 多声卡冲突 | 修改 ODAS 配置中的 card 值 |

---

## 四、项目部署

### 4.1 克隆项目

```bash
cd ~
git clone <你的仓库地址> yszl
cd yszl
```

### 4.2 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3 安装 Python 依赖

```bash
pip install --upgrade pip

# 核心依赖
pip install numpy scipy pyaudio pyyaml

# 可选: 声纹模型 (需要较大空间)
# pip install modelscope funasr
```

### 4.4 安装树莓派专用依赖

```bash
# GPIO 控制
pip install RPi.GPIO

# SPI (LED控制)
pip install spidev

# pixel_ring (可选, 更简单的LED控制)
pip install pixel_ring
```

---

## 五、ODAS 安装 (可选增强)

### 5.1 安装编译依赖

```bash
sudo apt install -y \
    cmake \
    libfftw3-dev \
    libconfig-dev \
    libasound2-dev
```

### 5.2 使用项目脚本安装

```bash
cd ~/yszl/client/odas
chmod +x start_odas.sh
./start_odas.sh install
```

### 5.3 启动 ODAS

```bash
./start_odas.sh start

# 检查状态
./start_odas.sh status
```

**预期输出**:
```
=== ODAS Status ===
Status: Running (PID 1234)

=== Socket Ports ===
tcp  0.0.0.0:9000  LISTEN
tcp  0.0.0.0:9001  LISTEN
```

---

## 六、功能测试

### 6.1 测试脚本概览

```bash
~/yszl/
├── client/
│   ├── acoustic_frontend/
│   │   ├── demo.py           # 综合演示 (DOA+波束+AEC)
│   │   └── test_*.py         # 各模块单元测试
│   └── odas/
│       └── start_odas.sh     # ODAS 管理
```

### 6.2 测试 1: 基础音频采集

```python
# test_audio_capture.py
import numpy as np
import pyaudio

CHANNELS = 8
RATE = 16000
CHUNK = 1024

p = pyaudio.PyAudio()

# 查找 ReSpeaker 设备
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if "seeed" in info["name"].lower():
        device_index = i
        print(f"Found ReSpeaker: {info['name']}")
        break

stream = p.open(
    format=pyaudio.paInt32,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK
)

print("Recording 3 seconds...")
frames = []
for _ in range(int(RATE / CHUNK * 3)):
    data = stream.read(CHUNK)
    frames.append(np.frombuffer(data, dtype=np.int32))

stream.stop_stream()
stream.close()
p.terminate()

audio = np.concatenate(frames).reshape(-1, CHANNELS)
print(f"Captured: {audio.shape}")
print(f"Max amplitude: {np.abs(audio).max()}")
```

### 6.3 测试 2: DOA 估计

```python
# test_doa.py
import sys
sys.path.insert(0, "/home/pi/yszl/client")

from acoustic_frontend import EnhancedDOAEstimator
import numpy as np

doa = EnhancedDOAEstimator(backend="auto")
print(f"Backend: {doa.get_backend().value}")

# 模拟数据测试
test_audio = np.random.randn(1600, 6).astype(np.float32) * 1000
result = doa.estimate(test_audio)

print(f"DOA: {result.angle:.1f}°")
print(f"Confidence: {result.confidence:.3f}")
print(f"Backend used: {result.backend.value}")
```

### 6.4 测试 3: LED 控制

```python
# test_led.py
import sys
sys.path.insert(0, "/home/pi/yszl/client")

from acoustic_frontend import LEDRing, LEDPattern
import time

led = LEDRing()
led.start()

print("Test 1: DOA indicator at 90°")
led.show_doa(90)
time.sleep(2)

print("Test 2: Speaker DOA (doctor)")
led.show_speaker_doa(0, "doctor")
time.sleep(2)

print("Test 3: Multi-speaker")
led.show_multi_speakers([
    {"angle": 0, "role": "doctor"},
    {"angle": 120, "role": "patient"}
])
time.sleep(2)

print("Test 4: Listening pattern")
led.set_pattern(LEDPattern.LISTENING)
time.sleep(2)

print("Turning off...")
led.stop()
```

### 6.5 测试 4: 综合演示

```bash
cd ~/yszl/client/acoustic_frontend
python demo.py
```

**演示功能**:
- 实时 DOA 估计并显示在 LED 上
- 波束成形增强
- VAD 检测
- 录音保存

### 6.6 测试 5: ODAS 集成

```python
# test_odas_client.py
import sys
sys.path.insert(0, "/home/pi/yszl/client")

from acoustic_frontend import ODASClient
import time

client = ODASClient(sst_port=9000)
client.start()

print("Monitoring ODAS for 10 seconds...")
for _ in range(20):
    sources = client.get_tracked_sources()
    if sources:
        for s in sources:
            print(f"Source {s.id}: {s.azimuth:.1f}° (energy: {s.energy:.3f})")
    else:
        print("No active sources")
    time.sleep(0.5)

client.stop()
```

---

## 七、自动启动配置

### 7.1 创建 Systemd 服务

```bash
sudo nano /etc/systemd/system/yszl-acoustic.service
```

```ini
[Unit]
Description=YiShengZhiLian Acoustic Frontend
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yszl/client
ExecStartPre=/home/pi/yszl/client/odas/start_odas.sh start
ExecStart=/home/pi/yszl/venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 7.2 启用服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable yszl-acoustic
sudo systemctl start yszl-acoustic

# 查看日志
journalctl -u yszl-acoustic -f
```

---

## 八、性能优化

### 8.1 CPU 亲和性

将音频处理绑定到特定核心:
```python
import os
os.sched_setaffinity(0, {2, 3})  # 使用 CPU 2 和 3
```

### 8.2 实时调度优先级

```bash
# 提升 Python 进程优先级
sudo renice -n -10 -p $(pgrep -f "python.*main.py")
```

### 8.3 禁用不必要服务

```bash
# 减少系统负载
sudo systemctl disable bluetooth
sudo systemctl disable ModemManager
```

---

## 九、故障排查清单

### 9.1 快速诊断命令

```bash
# 1. 检查声卡
arecord -l

# 2. 检查 I2C
i2cdetect -y 1

# 3. 检查 ODAS
~/yszl/client/odas/start_odas.sh status

# 4. 检查 Python 环境
source ~/yszl/venv/bin/activate
python -c "from acoustic_frontend import *; print('OK')"

# 5. 检查 GPIO 权限
ls -la /dev/spidev*
ls -la /dev/gpiomem
```

### 9.2 常见错误

| 错误 | 原因 | 解决 |
|-----|------|-----|
| `ALSA: cannot open` | 声卡未识别 | 重新安装驱动 |
| `Permission denied: /dev/spidev` | SPI 权限 | `sudo usermod -aG spi pi` |
| `ODAS connection refused` | ODAS 未启动 | `start_odas.sh start` |
| `No module named 'pixel_ring'` | 依赖缺失 | `pip install pixel_ring` |

---

## 十、测试报告模板

```markdown
# 声学前端测试报告

**测试日期**: YYYY-MM-DD
**测试环境**: Raspberry Pi 5 (8GB) + ReSpeaker 6-Mic

## 测试结果

| 测试项 | 状态 | 备注 |
|-------|------|-----|
| 驱动安装 | ✓/✗ | |
| 8通道录音 | ✓/✗ | |
| DOA 估计 | ✓/✗ | 精度: __° |
| LED 控制 | ✓/✗ | |
| ODAS 集成 | ✓/✗ | |
| 波束成形 | ✓/✗ | SNR提升: __dB |

## 问题记录

1. 问题描述
   - 现象: 
   - 原因: 
   - 解决: 
```

---

*文档版本: 1.0 | 更新日期: 2026-01-07*
