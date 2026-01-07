# ODAS 声源检测问题排查与修复文档

## 问题描述

使用 ReSpeaker 6-Mic Circular Array 配合 ODAS (Open embeddeD Audition System) 进行声源定位时，遇到以下问题：

- 运行 `test_connection.py` 显示 **"Connected, but no active sources..."**
- 对着麦克风说话后仍然没有反应
- ODAS 和 Python 客户端已经连接成功（SST 9000 和 SSL 9001 端口都显示 connected）

## 排查过程

### 第一阶段：确认连接状态

运行测试脚本后，两个端口（SST 9000 和 SSS 9001）都显示 connected，说明底层通信链路已经打通。

```bash
python client/odas/test_connection.py
```

输出显示 "Connected, but no active sources..." 说明：
1. ODAS 进程正在运行
2. Python 客户端已经连接到 ODAS
3. 但是没有检测到活跃的声源

### 第二阶段：检查麦克风硬件

#### 2.1 确认音频设备

```bash
arecord -l
```

输出显示 `card 3: seeed8telecom [seeed-8mic-voicecard]`，确认 ReSpeaker 6-Mic Array 已识别。

#### 2.2 测试麦克风录音

```bash
arecord -D hw:3,0 -f S32_LE -r 16000 -c 8 -d 5 test.wav
```

**注意**：ReSpeaker 6-Mic Array 只支持 **S32_LE** 格式，不支持 S16_LE！

#### 2.3 检查麦克风增益

**关键发现**：麦克风增益被重置为 0！

```bash
amixer -c 3 cget name='ADC1 PGA gain'
# 输出: values=0  <- 这是问题所在！
```

#### 2.4 设置正确的增益

```bash
# 设置所有 ADC (模拟) 增益到 8dB
for i in {1..8}; do 
    amixer -c 3 cset name="ADC${i} PGA gain" 8
done

# 设置所有 CH (数字) 音量到 160
for i in {1..8}; do 
    amixer -c 3 cset name="CH${i} digital volume" 160
done
```

**增益参数说明**：
- ADC PGA gain: 范围 0-31，建议 8（太高会饱和）
- CH digital volume: 范围 0-255，建议 160

### 第三阶段：验证麦克风工作

创建直接测试脚本 `test_alsa_direct.py` 来验证麦克风是否正常工作：

```bash
python3 test_alsa_direct.py
```

输出显示：
```
[帧   100] 能量: [0.09297 0.07541 0.10026 0.14601 0.11536 0.11021]
           最大: [0.28215 0.27259 0.32084 0.33580 0.28609 0.27047]
✅ 检测到音频信号！
```

**结论**：麦克风硬件工作正常，问题在 ODAS 或 Python 客户端。

### 第四阶段：分析 ODAS 原始输出

创建 `dump_raw_json.py` 脚本直接查看 ODAS 的原始 JSON 输出：

```bash
python3 dump_raw_json.py
# 在另一个终端启动 ODAS
```

**关键发现**：ODAS 输出的是**多行 JSON** 格式！

```json
{
    "timeStamp": 1,
    "src": [
        { "id": 0, "tag": "", "x": 0.000, "y": 0.000, "z": 0.000, "activity": 0.000 },
        { "id": 9, "tag": "dynamic", "x": 0.486, "y": 0.793, "z": 0.366, "activity": 0.639 }
    ]
}
```

看到 `activity: 0.639`！说明 **ODAS 实际上是工作的**！

### 第五阶段：定位根本原因

原来的 Python 客户端 `odas_client.py` 按**单行**解析 JSON：

```python
# 错误的解析方式
while '\n' in buffer:
    line, buffer = buffer.split('\n', 1)
    self._parse_sst_json(line)  # 每行单独解析，但 JSON 是多行的！
```

这导致每次只解析到 JSON 的一部分，`json.loads()` 失败，无法获取 activity 数据。

## 修复方案

### 修复 1：正确解析多行 JSON

修改 `client/acoustic_frontend/odas_client.py` 的 `_sst_receiver` 方法：

```python
def _sst_receiver(self):
    """SST (跟踪) 数据接收线程"""
    # ... 初始化代码 ...
    
    buffer = ""
    brace_count = 0
    json_start = -1
    
    while self._running:
        data = self._sst_socket.recv(4096).decode('utf-8')
        buffer += data
        
        # 解析多行 JSON 对象 (通过大括号匹配)
        i = 0
        while i < len(buffer):
            c = buffer[i]
            if c == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0 and json_start >= 0:
                    # 找到完整的 JSON 对象
                    json_str = buffer[json_start:i+1]
                    self._parse_sst_json(json_str)
                    buffer = buffer[i+1:]
                    i = -1
                    json_start = -1
            i += 1
```

### 修复 2：设置麦克风增益

每次启动 ODAS 前，需要设置麦克风增益（因为系统重启或设备重新初始化后增益会被重置）：

```bash
# 添加到 start_odas.sh 或单独执行
for i in {1..8}; do 
    amixer -c 3 cset name="ADC${i} PGA gain" 8
    amixer -c 3 cset name="CH${i} digital volume" 160
done
```

## 验证修复

修复后运行测试：

```bash
# 终端 1: 启动 Python 客户端
python3 client/odas/test_connection.py

# 终端 2: 设置增益并启动 ODAS
for i in {1..8}; do amixer -c 3 cset name="ADC${i} PGA gain" 8; amixer -c 3 cset name="CH${i} digital volume" 160; done
/home/k-means/yszl/client/odas/odas_build/build/bin/odaslive -c /home/k-means/yszl/client/odas/respeaker_6mic.cfg
```

**成功输出**：
```
🎤 检测到 1 个活跃声源!
   声源 0: 方位角=294.7°, activity=0.990

🎤 检测到 1 个活跃声源!
   声源 0: 方位角=295.8°, activity=0.892

历史最大 activity: 1.000
```

## 问题总结

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 无法录音 | 格式错误 | 使用 S32_LE 而非 S16_LE |
| 录音无声 | 增益为 0 | 设置 ADC=8, Digital=160 |
| ODAS 无输出 | JSON 解析错误 | 改用大括号匹配解析多行 JSON |
| activity 始终为 0 | 同上 | 同上 |

## 相关文件

- `client/acoustic_frontend/odas_client.py` - ODAS Python 客户端（已修复）
- `client/odas/respeaker_6mic.cfg` - ODAS 配置文件
- `client/odas/start_odas.sh` - ODAS 启动脚本
- `client/odas/test_connection.py` - 连接测试脚本
- `client/odas/test_alsa_direct.py` - 麦克风直接测试脚本
- `client/odas/dump_raw_json.py` - ODAS 原始 JSON 输出查看器

## 常用命令

```bash
# 检查增益
amixer -c 3 cget name='ADC1 PGA gain'

# 设置增益
for i in {1..8}; do amixer -c 3 cset name="ADC${i} PGA gain" 8; amixer -c 3 cset name="CH${i} digital volume" 160; done

# 测试麦克风
python3 client/odas/test_alsa_direct.py

# 启动完整系统
# 终端 1:
python3 client/odas/test_connection.py
# 终端 2:
cd client/odas && ./start_odas.sh start

# 查看 ODAS 日志
cat /tmp/odas.log
```

## 注意事项

1. **增益会被重置**：系统重启、设备重新初始化、甚至某些程序打开音频设备后，增益可能被重置为 0。建议在 `start_odas.sh` 中自动设置增益。

2. **ODAS JSON 格式**：ODAS 输出的是多行 JSON，不是单行。任何解析 ODAS 输出的代码都需要处理这一点。

3. **音频格式**：ReSpeaker 6-Mic Array 只支持 S32_LE 格式，配置文件中 `nBits = 32` 是必须的。

4. **通道映射**：配置文件中 `map: (1, 2, 3, 4, 5, 6)` 是 1-indexed 的，对应 ALSA 的通道 0-5。

---

*文档创建日期：2026-01-08*
*最后更新：问题已完全修复*

