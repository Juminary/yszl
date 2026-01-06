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
            body: JSON.stringify({ text: dialogueText })
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

    let html = '';

    if (soap.S || soap.subjective) {
        html += `
            <div class="soap-section">
                <h4>S - 主诉 (Subjective)</h4>
                <p>${soap.S || soap.subjective}</p>
            </div>
        `;
    }

    if (soap.O || soap.objective) {
        html += `
            <div class="soap-section">
                <h4>O - 客观检查 (Objective)</h4>
                <p>${soap.O || soap.objective}</p>
            </div>
        `;
    }

    if (soap.A || soap.assessment) {
        html += `
            <div class="soap-section">
                <h4>A - 评估诊断 (Assessment)</h4>
                <p>${soap.A || soap.assessment}</p>
            </div>
        `;
    }

    if (soap.P || soap.plan) {
        html += `
            <div class="soap-section">
                <h4>P - 治疗计划 (Plan)</h4>
                <p>${soap.P || soap.plan}</p>
            </div>
        `;
    }

    if (!html) {
        html = `<p>${JSON.stringify(data, null, 2)}</p>`;
    }

    elements.soapResult.innerHTML = html;
    elements.copySoapBtn.style.display = 'flex';
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
