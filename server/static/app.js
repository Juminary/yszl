/**
 * 智能医疗语音助手 - 前端交互逻辑
 */

// API 基础地址
const API_BASE = window.location.origin;

// 状态管理
const state = {
    currentMode: 'patient',  // patient | doctor | consultation
    sessionId: `session_${Date.now()}`,
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    isProcessing: false
};

// DOM 元素
const elements = {
    // 模式切换
    modeBtns: document.querySelectorAll('.mode-btn'),

    // 对话面板
    chatPanel: document.getElementById('chat-panel'),
    chatMessages: document.getElementById('chat-messages'),
    textInput: document.getElementById('text-input'),
    sendBtn: document.getElementById('send-btn'),
    recordBtn: document.getElementById('record-btn'),
    recordingStatus: document.getElementById('recording-status'),
    modeDescription: document.getElementById('mode-description'),

    // 会诊面板
    consultationPanel: document.getElementById('consultation-panel'),
    dialogueInput: document.getElementById('dialogue-input'),
    generateSoapBtn: document.getElementById('generate-soap-btn'),
    soapResult: document.getElementById('soap-result'),
    copySoapBtn: document.getElementById('copy-soap-btn'),

    // 状态栏
    modeStatus: document.getElementById('mode-status'),
    connectionStatus: document.getElementById('connection-status'),

    // 音频播放器
    audioPlayer: document.getElementById('audio-player')
};

// 模式描述
const modeDescriptions = {
    patient: '请描述您的症状，我会为您提供初步的导诊建议。',
    doctor: '我将为您提供专业的辅助诊断建议。',
    consultation: '粘贴医患对话，生成 SOAP 格式病历。'
};

const modeStatusTexts = {
    patient: '🟢 患者模式',
    doctor: '🟡 医生模式',
    consultation: '🔵 会诊模式'
};

// ==================== 初始化 ====================

document.addEventListener('DOMContentLoaded', () => {
    initModeSwitch();
    initTextInput();
    initVoiceInput();
    initConsultation();
    checkServerConnection();
});

// ==================== 模式切换 ====================

function initModeSwitch() {
    elements.modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            switchMode(mode);
        });
    });
}

function switchMode(mode) {
    state.currentMode = mode;

    // 更新按钮状态
    elements.modeBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // 切换面板显示
    if (mode === 'consultation') {
        elements.chatPanel.style.display = 'none';
        elements.consultationPanel.style.display = 'grid';
    } else {
        elements.chatPanel.style.display = 'flex';
        elements.consultationPanel.style.display = 'none';
        elements.modeDescription.textContent = modeDescriptions[mode];
    }

    // 更新状态栏
    elements.modeStatus.textContent = modeStatusTexts[mode];
}

// ==================== 文字输入 ====================

function initTextInput() {
    // 发送按钮点击
    elements.sendBtn.addEventListener('click', sendTextMessage);

    // 回车发送（Shift+Enter 换行）
    elements.textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage();
        }
    });

    // 自动调整高度
    elements.textInput.addEventListener('input', () => {
        elements.textInput.style.height = 'auto';
        elements.textInput.style.height = Math.min(elements.textInput.scrollHeight, 120) + 'px';
    });
}

async function sendTextMessage() {
    const text = elements.textInput.value.trim();
    if (!text || state.isProcessing) return;

    state.isProcessing = true;

    // 清空输入框
    elements.textInput.value = '';
    elements.textInput.style.height = 'auto';

    // 添加用户消息
    addMessage(text, 'user');

    // 显示加载状态
    const loadingId = addLoadingMessage();

    try {
        // 调用对话接口
        const response = await fetch(`${API_BASE}/dialogue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: text,
                session_id: state.sessionId,
                mode: state.currentMode
            })
        });

        if (!response.ok) {
            throw new Error(`请求失败: ${response.status}`);
        }

        const data = await response.json();

        // 移除加载消息
        removeMessage(loadingId);

        // 处理模式切换
        if (data.mode_switched) {
            switchMode(data.mode);
            addMessage(`已切换到${modeDescriptions[data.mode]}`, 'assistant');
        }

        // 添加助手回复
        const responseText = data.response || data.text || '';
        if (responseText) {
            addMessage(responseText, 'assistant');

            // 播放 TTS
            playTTS(responseText);
        }

    } catch (error) {
        console.error('发送消息失败:', error);
        removeMessage(loadingId);
        addMessage(`❌ ${error.message}`, 'assistant', true);
    }

    state.isProcessing = false;
}

// ==================== 语音输入 ====================

function initVoiceInput() {
    elements.recordBtn.addEventListener('click', toggleRecording);
}

async function toggleRecording() {
    if (state.isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                state.audioChunks.push(e.data);
            }
        };

        state.mediaRecorder.onstop = async () => {
            // 停止所有音轨
            stream.getTracks().forEach(track => track.stop());

            // 处理录音
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            await sendAudioMessage(audioBlob);
        };

        state.mediaRecorder.start();
        state.isRecording = true;

        // 更新 UI
        elements.recordBtn.classList.add('recording');
        elements.recordBtn.querySelector('.record-text').textContent = '点击停止';
        elements.recordingStatus.classList.add('active');

    } catch (error) {
        console.error('无法访问麦克风:', error);
        addMessage('❌ 无法访问麦克风，请检查权限设置', 'assistant', true);
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.mediaRecorder.stop();
        state.isRecording = false;

        // 更新 UI
        elements.recordBtn.classList.remove('recording');
        elements.recordBtn.querySelector('.record-text').textContent = '点击录音';
        elements.recordingStatus.classList.remove('active');
    }
}

async function sendAudioMessage(audioBlob) {
    state.isProcessing = true;

    // 显示加载状态
    const loadingId = addLoadingMessage();

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        formData.append('session_id', state.sessionId);

        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `请求失败: ${response.status}`);
        }

        // 获取响应头信息
        const asrText = decodeURIComponent(response.headers.get('X-ASR-Text') || '');
        const responseText = decodeURIComponent(response.headers.get('X-Response-Text') || '');

        // 移除加载消息
        removeMessage(loadingId);

        // 添加用户消息（ASR 结果）
        if (asrText) {
            addMessage(asrText, 'user');
        }

        // 添加助手回复
        if (responseText) {
            addMessage(responseText, 'assistant');
        }

        // 播放音频回复
        const audioData = await response.blob();
        if (audioData.size > 0) {
            const audioUrl = URL.createObjectURL(audioData);
            elements.audioPlayer.src = audioUrl;
            elements.audioPlayer.play();
        }

    } catch (error) {
        console.error('语音处理失败:', error);
        removeMessage(loadingId);
        addMessage(`❌ ${error.message}`, 'assistant', true);
    }

    state.isProcessing = false;
}

// ==================== 会诊模式 ====================

function initConsultation() {
    elements.generateSoapBtn.addEventListener('click', generateSOAP);
    elements.copySoapBtn.addEventListener('click', copySOAP);
    document.getElementById('export-html-btn').addEventListener('click', exportHTML);
}

async function generateSOAP() {
    const dialogueText = elements.dialogueInput.value.trim();
    if (!dialogueText) {
        alert('请先粘贴医患对话内容');
        return;
    }

    elements.generateSoapBtn.disabled = true;
    elements.generateSoapBtn.innerHTML = '<span class="loading"><span></span><span></span><span></span></span> 生成中...';

    try {
        const response = await fetch(`${API_BASE}/aci/generate-soap`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dialogue_text: dialogueText })
        });

        if (!response.ok) {
            throw new Error(`请求失败: ${response.status}`);
        }

        const data = await response.json();

        // 显示结果
        displaySOAPResult(data);

    } catch (error) {
        console.error('生成病历失败:', error);
        elements.soapResult.innerHTML = `<p style="color: #ff6666;">❌ 生成失败: ${error.message}</p>`;
    }

    elements.generateSoapBtn.disabled = false;
    elements.generateSoapBtn.innerHTML = '<span>📋</span> 生成 SOAP 病历';
}

function displaySOAPResult(data) {
    const soap = data.soap || data;

    // 存储原始数据供导出使用
    state.lastSoapData = soap;

    let html = '';

    // 处理 subjective（可能是对象或字符串）
    const subjective = soap.subjective || soap.S;
    if (subjective) {
        let content = '';
        if (typeof subjective === 'object') {
            if (subjective.chief_complaint) content += `<p><strong>主诉：</strong>${subjective.chief_complaint}</p>`;
            if (subjective.history) content += `<p><strong>现病史：</strong>${subjective.history}</p>`;
            if (!content) content = `<p>${JSON.stringify(subjective)}</p>`;
        } else {
            content = `<p>${subjective}</p>`;
        }
        html += `<div class="soap-section"><h4>S - 主诉 (Subjective)</h4>${content}</div>`;
    }

    // 处理 objective
    const objective = soap.objective || soap.O;
    if (objective) {
        let content = '';
        if (typeof objective === 'object') {
            if (objective.vital_signs) content += `<p><strong>生命体征：</strong>${objective.vital_signs}</p>`;
            if (objective.content) content += `<p><strong>体格检查：</strong>${objective.content}</p>`;
            if (!content) content = `<p>${JSON.stringify(objective)}</p>`;
        } else {
            content = `<p>${objective}</p>`;
        }
        html += `<div class="soap-section"><h4>O - 客观检查 (Objective)</h4>${content}</div>`;
    }

    // 处理 assessment
    const assessment = soap.assessment || soap.A;
    if (assessment) {
        let content = '';
        if (typeof assessment === 'object') {
            if (assessment.diagnosis) content += `<p><strong>诊断：</strong>${assessment.diagnosis}</p>`;
            if (assessment.content) content += `<p><strong>评估：</strong>${assessment.content}</p>`;
            if (!content) content = `<p>${JSON.stringify(assessment)}</p>`;
        } else {
            content = `<p>${assessment}</p>`;
        }
        html += `<div class="soap-section"><h4>A - 评估诊断 (Assessment)</h4>${content}</div>`;
    }

    // 处理 plan
    const plan = soap.plan || soap.P;
    if (plan) {
        let content = '';
        if (typeof plan === 'object') {
            if (plan.treatment) content += `<p><strong>治疗方案：</strong>${plan.treatment}</p>`;
            if (plan.content) content += `<p><strong>医嘱：</strong>${plan.content}</p>`;
            if (!content) content = `<p>${JSON.stringify(plan)}</p>`;
        } else {
            content = `<p>${plan}</p>`;
        }
        html += `<div class="soap-section"><h4>P - 治疗计划 (Plan)</h4>${content}</div>`;
    }

    // 处理提取的实体
    if (soap.entities && soap.entities.length > 0) {
        const symptoms = soap.entities.filter(e => e.type === 'symptom').map(e => e.text || e.value);
        const diseases = soap.entities.filter(e => e.type === 'disease').map(e => e.text || e.value);
        const drugs = soap.entities.filter(e => e.type === 'drug').map(e => e.text || e.value);

        if (symptoms.length || diseases.length || drugs.length) {
            html += '<div class="soap-section"><h4>📊 提取的医学实体</h4>';
            if (symptoms.length) html += `<p><strong>症状：</strong>${symptoms.join('、')}</p>`;
            if (diseases.length) html += `<p><strong>疾病：</strong>${diseases.join('、')}</p>`;
            if (drugs.length) html += `<p><strong>药物：</strong>${drugs.join('、')}</p>`;
            html += '</div>';
        }
    }

    if (!html) {
        html = `<pre style="white-space: pre-wrap; color: #888;">${JSON.stringify(data, null, 2)}</pre>`;
    }

    elements.soapResult.innerHTML = html;
    document.getElementById('soap-actions').style.display = 'flex';
}

function copySOAP() {
    const text = elements.soapResult.innerText;
    navigator.clipboard.writeText(text).then(() => {
        elements.copySoapBtn.innerHTML = '<span>✓</span> 已复制';
        setTimeout(() => {
            elements.copySoapBtn.innerHTML = '<span>📋</span> 复制病历';
        }, 2000);
    });
}

function exportHTML() {
    const now = new Date();
    const dateStr = now.toLocaleDateString('zh-CN');
    const timeStr = now.toLocaleTimeString('zh-CN');
    const recordId = 'JKS' + Math.random().toString(36).substring(2, 8).toUpperCase();

    // 从 SOAP 数据中提取各部分内容
    const soap = state.lastSoapData || {};
    const subjective = soap.subjective || {};
    const objective = soap.objective || {};
    const assessment = soap.assessment || {};
    const plan = soap.plan || {};

    // 提取内容
    const chiefComplaint = typeof subjective === 'string' ? subjective : (subjective.chief_complaint || '待记录');
    const history = typeof subjective === 'string' ? '' : (subjective.history || '待记录');
    const vitalSigns = typeof objective === 'string' ? objective : (objective.vital_signs || '待检查');
    const physicalExam = typeof objective === 'string' ? '' : (objective.content || '待检查');
    const diagnosis = typeof assessment === 'string' ? assessment : (assessment.diagnosis || '待诊断');
    const assessmentContent = typeof assessment === 'string' ? '' : (assessment.content || '待评估');
    const treatment = typeof plan === 'string' ? plan : (plan.treatment || '待制定');
    const advice = typeof plan === 'string' ? '' : (plan.content || '待记录');

    // 提取症状和疾病标签
    const symptoms = (soap.entities || []).filter(e => e.type === 'symptom').map(e => e.text || e.value);
    const diseases = (soap.entities || []).filter(e => e.type === 'disease').map(e => e.text || e.value);

    const htmlContent = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电子病历 - ${dateStr}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: "SimSun", "宋体", serif;
            font-size: 14px;
            line-height: 1.8;
            color: #000;
            background: #fff;
            padding: 20px;
        }
        .medical-record {
            max-width: 800px;
            margin: 0 auto;
            border: 2px solid #000;
            padding: 30px;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #000;
            padding-bottom: 20px;
            margin-bottom: 20px;
        }
        .hospital-name {
            font-size: 24px;
            font-weight: bold;
            letter-spacing: 4px;
            margin-bottom: 10px;
        }
        .record-title {
            font-size: 20px;
            font-weight: bold;
            border: 1px solid #000;
            display: inline-block;
            padding: 5px 30px;
            margin-top: 10px;
        }
        .patient-info {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            border-bottom: 1px solid #000;
            padding: 15px 0;
            margin-bottom: 20px;
        }
        .info-item {
            display: flex;
        }
        .info-label {
            font-weight: bold;
            min-width: 70px;
        }
        .info-value {
            border-bottom: 1px solid #000;
            flex: 1;
            min-width: 80px;
            padding: 0 5px;
        }
        .section {
            margin-bottom: 20px;
            page-break-inside: avoid;
        }
        .section-title {
            font-weight: bold;
            font-size: 15px;
            background: #f0f0f0;
            padding: 8px 15px;
            border-left: 4px solid #1a5f7a;
            margin-bottom: 10px;
        }
        .section-content {
            padding: 10px 15px;
            min-height: 60px;
            border: 1px solid #ddd;
            background: #fafafa;
        }
        .content-row {
            margin-bottom: 8px;
        }
        .content-label {
            font-weight: bold;
            color: #333;
        }
        .entity-tags {
            margin-top: 10px;
        }
        .entity-tag {
            display: inline-block;
            padding: 2px 10px;
            margin: 2px;
            border-radius: 3px;
            font-size: 12px;
        }
        .entity-tag.symptom { background: #fff3cd; color: #856404; border: 1px solid #ffc107; }
        .entity-tag.disease { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .entity-tag.medication { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .signature-area {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #000;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        .signature-item {
            display: flex;
            align-items: flex-end;
        }
        .signature-label {
            font-weight: bold;
            white-space: nowrap;
        }
        .signature-line {
            flex: 1;
            border-bottom: 1px solid #000;
            margin-left: 10px;
            min-width: 120px;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 12px;
            color: #666;
            border-top: 1px dashed #ccc;
            padding-top: 15px;
        }
        @media print {
            body { padding: 0; }
            .medical-record { border: none; }
            .footer { display: none; }
        }
    </style>
</head>
<body>
    <div class="medical-record">
        <div class="header">
            <div class="hospital-name">智 能 医 疗 助 手</div>
            <div style="font-size: 14px; color: #666;">AI-Powered Medical Assistant</div>
            <div class="record-title">门 诊 病 历</div>
        </div>
        
        <div class="patient-info">
            <div class="info-item">
                <span class="info-label">就诊日期：</span>
                <span class="info-value">${dateStr}</span>
            </div>
            <div class="info-item">
                <span class="info-label">就诊时间：</span>
                <span class="info-value">${now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>
            </div>
            <div class="info-item">
                <span class="info-label">病历号：</span>
                <span class="info-value">${recordId}</span>
            </div>
            <div class="info-item">
                <span class="info-label">姓　　名：</span>
                <span class="info-value"></span>
            </div>
            <div class="info-item">
                <span class="info-label">性　　别：</span>
                <span class="info-value"></span>
            </div>
            <div class="info-item">
                <span class="info-label">年　　龄：</span>
                <span class="info-value"></span>
            </div>
        </div>

        <div class="section">
            <div class="section-title">一、主诉及现病史 (Subjective)</div>
            <div class="section-content">
                <div class="content-row">
                    <span class="content-label">主　诉：</span>
                    ${chiefComplaint}
                </div>
                <div class="content-row">
                    <span class="content-label">现病史：</span>
                    ${history}
                </div>
                ${symptoms.length > 0 ? `
                <div class="entity-tags">
                    <span class="content-label">症状标签：</span>
                    ${symptoms.map(s => '<span class="entity-tag symptom">' + s + '</span>').join('')}
                </div>
                ` : ''}
            </div>
        </div>

        <div class="section">
            <div class="section-title">二、体格检查 (Objective)</div>
            <div class="section-content">
                <div class="content-row">
                    <span class="content-label">生命体征：</span>
                    ${vitalSigns}
                </div>
                <div class="content-row">
                    <span class="content-label">体格检查：</span>
                    ${physicalExam}
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">三、诊断意见 (Assessment)</div>
            <div class="section-content">
                <div class="content-row">
                    <span class="content-label">初步诊断：</span>
                    ${diagnosis}
                </div>
                <div class="content-row">
                    <span class="content-label">病情评估：</span>
                    ${assessmentContent}
                </div>
                ${diseases.length > 0 ? `
                <div class="entity-tags">
                    <span class="content-label">疾病标签：</span>
                    ${diseases.map(d => '<span class="entity-tag disease">' + d + '</span>').join('')}
                </div>
                ` : ''}
            </div>
        </div>

        <div class="section">
            <div class="section-title">四、治疗方案 (Plan)</div>
            <div class="section-content">
                <div class="content-row">
                    <span class="content-label">治疗方案：</span>
                    ${treatment}
                </div>
                <div class="content-row">
                    <span class="content-label">医　　嘱：</span>
                    ${advice}
                </div>
            </div>
        </div>

        <div class="signature-area">
            <div class="signature-item">
                <span class="signature-label">主治医师：</span>
                <span class="signature-line"></span>
            </div>
            <div class="signature-item">
                <span class="signature-label">日　　期：</span>
                <span class="signature-line">${dateStr}</span>
            </div>
        </div>

        <div class="footer">
            <p>本病历由 AI 智能医疗助手辅助生成，仅供参考，不作为最终诊断依据</p>
            <p>如有疑问请咨询专业医生 | 生成时间：${dateStr} ${timeStr}</p>
        </div>
    </div>
</body>
</html>`;

    // 创建下载链接
    const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `电子病历_${dateStr.replace(/\//g, '-')}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // 更新按钮状态
    const exportBtn = document.getElementById('export-html-btn');
    exportBtn.innerHTML = '<span>✓</span> 已导出';
    setTimeout(() => {
        exportBtn.innerHTML = '<span>📥</span> 导出 HTML';
    }, 2000);
}

// ==================== 消息管理 ====================

function addMessage(text, type, isError = false) {
    const id = `msg_${Date.now()}`;

    // 移除欢迎消息
    const welcome = elements.chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const messageDiv = document.createElement('div');
    messageDiv.id = id;
    messageDiv.className = `message ${type}${isError ? ' error' : ''}`;

    messageDiv.innerHTML = `
        <div class="message-content">${escapeHtml(text)}</div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    return id;
}

function addLoadingMessage() {
    const id = `loading_${Date.now()}`;

    const messageDiv = document.createElement('div');
    messageDiv.id = id;
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <span class="loading"><span></span><span></span><span></span></span>
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    return id;
}

function removeMessage(id) {
    const msg = document.getElementById(id);
    if (msg) msg.remove();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== TTS 播放 ====================

async function playTTS(text) {
    try {
        const response = await fetch(`${API_BASE}/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            elements.audioPlayer.src = audioUrl;
            elements.audioPlayer.play();
        }
    } catch (error) {
        console.error('TTS 播放失败:', error);
    }
}

// ==================== 服务器连接检查 ====================

async function checkServerConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            elements.connectionStatus.textContent = '🟢 已连接';
        } else {
            elements.connectionStatus.textContent = '🔴 连接异常';
        }
    } catch (error) {
        elements.connectionStatus.textContent = '🔴 未连接';
    }
}

// 定期检查连接
setInterval(checkServerConnection, 30000);
