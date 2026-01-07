# 声学前端增强模块技术文档

> DOA辅助说话人分离系统 - 原理、功能与工作流程详解

---

## 一、系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         声学前端增强系统 v2.4                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐                 │
│   │ ReSpeaker│───▶│   DOA    │───▶│ Spatio-Spectral  │                 │
│   │  6-Mic   │    │ Estimator│    │     Fusion       │                 │
│   └──────────┘    └──────────┘    └────────┬─────────┘                 │
│        │              │                     │                           │
│        ▼              ▼                     ▼                           │
│   ┌──────────┐   ┌──────────┐       ┌──────────┐                        │
│   │Beamformer│   │   LED    │       │ Speaker  │                        │
│   │   AEC    │   │   Ring   │       │ Embedder │                        │
│   └──────────┘   └──────────┘       └──────────┘                        │
│        │                                  │                             │
│        ▼                                  ▼                             │
│   ┌──────────────────────────────────────────────┐                      │
│   │              FullDuplexController            │                      │
│   │         (DOA辅助打断 + 状态机管理)            │                      │
│   └──────────────────────────────────────────────┘                      │
│                          │                                              │
│                          ▼                                              │
│                    ┌──────────┐                                         │
│                    │   ASR    │                                         │
│                    │ Emotion  │                                         │
│                    └──────────┘                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心模块详解

### 2.1 ODAS客户端 (`odas_client.py`)

#### 原理

**ODAS (Open embeddeD Audition System)** 是一个开源的实时声源定位与分离引擎。

核心算法 **SRP-PHAT (Steered Response Power with Phase Transform)**:

```
           ┌─────────────────────────────────────────────┐
           │              SRP-PHAT 算法流程               │
           ├─────────────────────────────────────────────┤
           │                                             │
           │  1. 多通道音频输入 (6 mic)                   │
           │           ↓                                 │
           │  2. 计算麦克风对的 GCC-PHAT                  │
           │     R_ij(τ) = FFT⁻¹[ X_i·X_j* / |X_i·X_j*| ] │
           │           ↓                                 │
           │  3. 空间搜索: 对每个候选方向 θ               │
           │     P(θ) = Σ R_ij(τ_ij(θ))                  │
           │           ↓                                 │
           │  4. 找到功率最大的方向作为 DOA 估计          │
           │           ↓                                 │
           │  5. 卡尔曼滤波器平滑跟踪                     │
           │                                             │
           └─────────────────────────────────────────────┘
```

#### 关键类

```python
class ODASClient:
    """通过 Socket 接收 ODAS 输出"""
    
    def get_tracked_sources(self) -> List[TrackedSource]:
        """获取当前跟踪的声源列表"""
        
    def get_primary_doa(self) -> float:
        """获取主声源方向"""

@dataclass
class TrackedSource:
    id: int           # 声源ID (ODAS分配)
    azimuth: float    # 方位角 (0-360°)
    energy: float     # 能量
    activity: float   # 活跃度
```

#### 数据流

```
ODAS进程 (C++) ──[Socket 9000]──▶ ODASClient (Python)
     │                                  │
     │  JSON格式:                       │
     │  {"src": [                       │
     │    {"x": 0.5, "y": 0.8,         │
     │     "activity": 0.9}             │
     │  ]}                              │
     │                                  ▼
     │                          TrackedSource对象
     │                                  │
     └──[Socket 9001]──▶ 分离后音频流 ──┘
```

---

### 2.2 增强DOA估计器 (`doa.py`)

#### 原理

双后端设计，自动选择最佳算法:

| 后端 | 算法 | 精度 | 计算量 |
|-----|------|-----|-------|
| ODAS | SRP-PHAT + Kalman | 高 | 高 (C++优化) |
| 内置 | GCC-PHAT | 中 | 低 (Python) |

#### GCC-PHAT 时延估计

```
信号1: x₁(t) ──────────────────────────┐
                                       ├──▶ 互相关 ──▶ 峰值位置 = τ
信号2: x₂(t) = x₁(t - τ) ─────────────┘

GCC-PHAT 公式:
R(τ) = ∫ [X₁(ω)X₂*(ω) / |X₁(ω)X₂*(ω)|] e^(jωτ) dω

其中 PHAT 加权 "白化" 频谱，使峰值更尖锐
```

#### 关键代码

```python
class EnhancedDOAEstimator:
    def estimate(self, mic_data: np.ndarray) -> DOAResult:
        # 1. 优先使用 ODAS (如果可用)
        if self._odas_available:
            return self._estimate_from_odas()
        
        # 2. 降级到内置 GCC-PHAT
        return self._estimate_gcc_phat(mic_data)
```

---

### 2.3 声纹嵌入提取器 (`speaker_embedder.py`)

#### 原理

声纹嵌入是将变长语音映射为固定长度向量的技术:

```
语音信号 (0.5s~10s) ──▶ 神经网络 ──▶ 192维向量 (声纹)
                         ↑
                 ECAPA-TDNN / CAM++
```

**ECAPA-TDNN 架构**:
- SE-Res2Block: 多尺度特征 + 注意力
- Attentive Statistics Pooling: 加权聚合时间维度
- 输出: 192维 L2归一化向量

#### 相似度计算

```python
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (||emb1|| * ||emb2||)

# 同一人: similarity > 0.7
# 不同人: similarity < 0.5
```

#### 使用示例

```python
embedder = SpeakerEmbedder(model_type="ecapa")  # 或 "cam++", "dummy"

emb1 = embedder.extract(audio_segment_1)
emb2 = embedder.extract(audio_segment_2)

similarity = emb1.cosine_similarity(emb2)
if similarity > 0.7:
    print("同一说话人")
```

---

### 2.4 空间-声纹融合 (`spatio_spectral_fusion.py`)

#### 核心创新点 ⭐

传统说话人分离仅依赖声纹，在短语音或多人重叠时失效。我们创新性地**融合空间和频谱特征**:

```
┌─────────────────────────────────────────────────────────┐
│                Spatio-Spectral Fusion                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   D_total = α × D_spatial + (1-α) × D_spectral         │
│                                                         │
│   其中:                                                 │
│   - D_spatial: DOA角度差 (归一化到0-1)                  │
│   - D_spectral: 声纹余弦距离                            │
│   - α: 动态权重                                        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    动态 α 调整策略                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   语音时长     α值      原因                            │
│   ─────────   ────     ─────────────────────────        │
│   < 0.5s      0.8      短语音声纹不稳定，多依赖位置      │
│   0.5s~3s     线性     过渡区间                         │
│   > 3.0s      0.2      长语音声纹可靠，多依赖声纹        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 工作流程

```python
fusion = SpatioSpectralFusion(alpha=0.5)

# 每次语音到来时
speaker_id = fusion.assign_speaker(
    doa=120.0,           # DOA角度
    embedding=emb_vec,   # 声纹向量
    duration=2.0         # 语音时长
)

# 内部流程:
# 1. 计算与所有已知说话人的融合距离
# 2. 如果最小距离 < 阈值 → 归入该说话人
# 3. 否则 → 创建新说话人
# 4. 更新说话人簇中心 (EMA)
```

#### 解决"鸡尾酒会"问题

```
场景: 医生 (0°方向) 和 患者 (120°方向) 同时说话

传统方法: 声纹混叠，无法区分
我们的方法:
  1. ODAS输出两个分离音频流
  2. Stream1 (0°) → 提取声纹 → 识别为医生
  3. Stream2 (120°) → 提取声纹 → 识别为患者
```

---

### 2.5 延迟校准器 (`latency_calibrator.py`)

#### 原理

AEC性能依赖参考信号与麦克风信号的精确对齐。使用 **Chirp信号** 测量系统延迟:

```
Chirp信号: 频率从 200Hz 线性扫到 4000Hz

           播放                     录制
           ────▶   【系统延迟】   ──────▶
                        ↓
            互相关峰值位置 = 延迟采样点数
```

#### 使用流程

```python
calibrator = LatencyCalibrator(sample_rate=16000)

# 1. 获取测试信号
test_signal = calibrator.generate_test_signal()

# 2. 播放并同时录制
recorded = play_and_record(test_signal)

# 3. 测量延迟
result = calibrator.measure_delay(recorded)
print(f"系统延迟: {result.delay_ms} ms")

# 4. 应用到AEC
aec.set_reference_delay(result.delay_samples)
```

---

### 2.6 情感解析器 (`emotion_parser.py`)

#### 原理

**SenseVoice** 模型在ASR转录时自动插入情感和事件标签:

```
输入音频 ──▶ SenseVoice ──▶ "今天心情不好<|SAD|>，咳咳<|COUGH|>"
```

#### 支持的标签

| 类型 | 标签 | 含义 |
|-----|------|-----|
| 情感 | `<|HAPPY|>` | 开心 |
| 情感 | `<|SAD|>` | 悲伤 |
| 情感 | `<|ANGRY|>` | 愤怒 |
| 情感 | `<|NEUTRAL|>` | 中性 |
| 事件 | `<|LAUGHTER|>` | 笑声 |
| 事件 | `<|COUGH|>` | 咳嗽 |

#### 使用示例

```python
parser = SenseVoiceEmotionParser()

result = parser.parse("我头疼<|SAD|>，很难受")

print(result.clean_text)       # "我头疼，很难受"
print(result.dominant_emotion) # "sad"

# 为TTS准备情感指令
text, instruction = parser.extract_for_tts(asr_text)
# instruction = "用温柔、安慰的语气"
```

---

### 2.7 LED环多说话人可视化

#### 新增功能

```python
led = LEDRing()

# 单说话人DOA显示
led.show_doa(90)  # 90°方向亮起

# 带角色颜色的DOA显示 (新)
led.show_speaker_doa(angle=0, speaker_role="doctor")   # 蓝色
led.show_speaker_doa(angle=120, speaker_role="patient") # 绿色

# 多说话人同时显示 (新)
led.show_multi_speakers([
    {"angle": 0, "role": "doctor"},
    {"angle": 120, "role": "patient"}
])
```

#### 颜色映射

```
医生 (doctor):  蓝色 RGB(0, 150, 255)
患者 (patient): 绿色 RGB(0, 255, 120)
家属 (family):  橙色 RGB(255, 180, 0)
未知 (unknown): 灰白 RGB(200, 200, 200)
```

---

### 2.8 DOA辅助打断检测

#### 原理

在TTS播放时，VAD可能误检扬声器回声为用户语音。利用DOA区分:

```
扬声器方向: 0° (前方)
用户方向: 通常在侧面或后方

判断逻辑:
if |DOA - 扬声器方向| < 30°:
    → 可能是回声，不触发打断
else:
    → 用户语音，触发打断
```

#### 使用方法

```python
controller = FullDuplexController()

# 设置扬声器方向和区分阈值
controller.set_speaker_direction(0.0)   # 扬声器在0°
controller.set_doa_threshold(30.0)      # 30°容差

# 处理音频时传入DOA
controller.process_audio_chunk(
    audio_chunk=audio,
    is_speech=True,
    doa_angle=120.0  # 用户在120°方向
)
# → 触发打断
```

---

## 三、完整工作流程

```
┌────────────────────────────────────────────────────────────────────────┐
│                        完整处理流程                                     │
└────────────────────────────────────────────────────────────────────────┘

Step 1: 音频采集
        ReSpeaker 6-Mic ──▶ 8通道PCM (6 mic + 2 echo)

Step 2: DOA估计
        6通道 ──▶ ODAS/GCC-PHAT ──▶ 角度θ + 能量

Step 3: 波束成形
        6通道 + θ ──▶ Beamformer ──▶ 增强音频

Step 4: 回声消除
        增强音频 + echo_ref ──▶ AEC ──▶ 干净音频

Step 5: 说话人分离
        干净音频 + θ ──▶ Spatio-Spectral Fusion ──▶ Speaker_ID

Step 6: LED可视化
        θ + Speaker_ID ──▶ LED Ring ──▶ 颜色指示

Step 7: TTS打断检测
        干净音频 + θ ──▶ DOA辅助判断 ──▶ 打断/继续

Step 8: ASR + 情感
        干净音频 ──▶ SenseVoice ──▶ 文本 + 情感标签
```

---

## 四、API快速参考

```python
from acoustic_frontend import (
    # DOA
    EnhancedDOAEstimator, DOAResult,
    
    # ODAS
    ODASClient, TrackedSource,
    
    # 说话人分离
    SpeakerEmbedder, SpatioSpectralFusion,
    
    # 情感
    SenseVoiceEmotionParser,
    
    # LED
    LEDRing,
    
    # 其他
    LatencyCalibrator, MicrophoneArray
)

# 初始化
doa = EnhancedDOAEstimator(backend="auto")
embedder = SpeakerEmbedder(model_type="dummy")
fusion = SpatioSpectralFusion(alpha=0.5)
parser = SenseVoiceEmotionParser()
led = LEDRing()

# 使用
result = doa.estimate(mic_data)
emb = embedder.extract(audio)
speaker_id = fusion.assign_speaker(doa=result.angle, embedding=emb.vector)
emotion = parser.parse(asr_text)
led.show_speaker_doa(result.angle, speaker_role="patient")
```

---

## 五、性能指标

| 指标 | 目标值 | 说明 |
|-----|-------|-----|
| DOA精度 | < 10° | SRP-PHAT在混响环境 |
| 说话人分离DER | < 15% | 双人交替说话场景 |
| 打断误触发率 | 降低60% | DOA辅助 vs 纯能量 |
| 处理延迟 | < 100ms | 实时交互要求 |

---

*文档版本: 2.4.0 | 更新日期: 2026-01-07*
