/**
 * 智能医疗语音助手 - 前端交互逻辑
 */

// ========================================
// 全局变量
// ========================================
const API_BASE = window.location.origin;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let sessionId = 'web-' + Date.now();

// DOM 元素
const elements = {
    recordBtn: null,
    recordingIndicator: null,
    textInput: null,
    sendTextBtn: null,
    chatMessages: null,
    audioPlayer: null,
    asrText: null,
    emotionBadge: null,
    emotionScore: null,
    speakerText: null,
    ragStatus: null,
    ragContent: null,
    registerBtn: null,
    speakerIdInput: null,
    clearHistoryBtn: null
};

// ========================================
// 初始化
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    initElements();
    initEventListeners();
    checkServerConnection();
});

function initElements() {
    elements.recordBtn = document.getElementById('record-btn');
    elements.recordingIndicator = document.getElementById('recording-indicator');
    elements.textInput = document.getElementById('text-input');
    elements.sendTextBtn = document.getElementById('send-text-btn');
    elements.chatMessages = document.getElementById('chat-messages');
    elements.audioPlayer = document.getElementById('audio-player');
    elements.asrText = document.getElementById('asr-text');
    elements.emotionBadge = document.getElementById('emotion-badge');
    elements.emotionScore = document.getElementById('emotion-score');
    elements.speakerText = document.getElementById('speaker-text');
    elements.ragStatus = document.getElementById('rag-status');
    elements.ragContent = document.getElementById('rag-content');
    elements.registerBtn = document.getElementById('register-btn');
    elements.speakerIdInput = document.getElementById('speaker-id-input');
    elements.clearHistoryBtn = document.getElementById('clear-history-btn');
}

function initEventListeners() {
    // 录音按钮
    elements.recordBtn.addEventListener('click', toggleRecording);

    // 文字发送
    elements.sendTextBtn.addEventListener('click', sendTextMessage);
    elements.textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage();
        }
    });

    // 自动调整文本框高度
    elements.textInput.addEventListener('input', autoResizeTextarea);

    // 声纹注册
    elements.registerBtn.addEventListener('click', registerSpeaker);

    // 清除历史
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
}

// ========================================
// 服务器连接检查
// ========================================
async function checkServerConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            showToast('✅ 服务器连接成功');
            // 连接成功后，启动SSE监听客户端消息
            connectToEventStream();
        } else {
            showToast('⚠️ 服务器响应异常');
        }
    } catch (error) {
        showToast('❌ 无法连接到服务器');
        console.error('Server connection error:', error);
    }
}

// ========================================
// SSE 消息同步 - 显示客户端的对话
// ========================================
function connectToEventStream() {
    const eventSource = new EventSource(`${API_BASE}/events`);

    eventSource.onopen = () => {
        console.log('SSE 连接已建立');
    };

    eventSource.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);

            // 忽略心跳和连接消息
            if (message.type === 'heartbeat' || message.type === 'connected') {
                return;
            }

            // 用户消息 (来自客户端)
            if (message.type === 'user_message' && message.data.source === 'client') {
                addMessage('user', message.data.text, { fromClient: true });
            }

            // 助手回复 (来自客户端的对话)
            if (message.type === 'assistant_message' && message.data.text) {
                addMessage('assistant', message.data.text, { fromClient: true });
            }

            console.log('收到SSE消息:', message);
        } catch (e) {
            console.error('解析SSE消息失败:', e);
        }
    };

    eventSource.onerror = (error) => {
        console.log('SSE 连接断开，5秒后重连...');
        eventSource.close();
        setTimeout(connectToEventStream, 5000);
    };
}

// ========================================
// 录音功能
// ========================================
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000
            }
        });

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await sendAudioToServer(audioBlob);

            // 停止所有音轨
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;

        // 更新 UI
        elements.recordBtn.classList.add('recording');
        elements.recordBtn.querySelector('.record-text').textContent = '停止录音';
        elements.recordingIndicator.classList.add('active');

    } catch (error) {
        console.error('录音失败:', error);
        showToast('❌ 无法访问麦克风，请检查权限设置');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        // 更新 UI
        elements.recordBtn.classList.remove('recording');
        elements.recordBtn.querySelector('.record-text').textContent = '点击录音';
        elements.recordingIndicator.classList.remove('active');
    }
}

// ========================================
// 发送音频到服务器
// ========================================
async function sendAudioToServer(audioBlob) {
    // 添加用户消息占位
    addMessage('user', '🎤 [语音消息]', { isVoice: true });

    // 显示加载状态
    const loadingMsg = addMessage('assistant', '', { isLoading: true });

    try {
        // 将 webm 转换为 wav（服务器可能需要）
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        formData.append('session_id', sessionId);

        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status}`);
        }

        // 获取响应头中的 RAG 信息
        const ragUsed = response.headers.get('X-RAG-Used') === 'true';

        // 获取 JSON 响应数据
        const contentType = response.headers.get('Content-Type');
        let data;

        if (contentType && contentType.includes('application/json')) {
            data = await response.json();
        } else {
            // 如果是音频响应，先获取 blob
            const audioData = await response.blob();
            data = {
                audio: audioData,
                text: response.headers.get('X-Response-Text') || '',
                asr_text: response.headers.get('X-ASR-Text') || '',
                emotion: response.headers.get('X-Emotion') || '',
                speaker: response.headers.get('X-Speaker') || ''
            };
        }

        // 更新识别结果面板
        updateRecognitionResults(data);

        // 更新 RAG 状态
        updateRagStatus(ragUsed, data.rag_context);

        // 移除加载消息，添加真实回复
        removeMessage(loadingMsg);

        // 更新用户消息为识别内容
        const userMessages = elements.chatMessages.querySelectorAll('.message.user');
        if (userMessages.length > 0) {
            const lastUserMsg = userMessages[userMessages.length - 1];
            const textEl = lastUserMsg.querySelector('.message-text');
            if (textEl && data.asr_text) {
                textEl.textContent = data.asr_text;
            }
        }

        // 添加助手回复
        addMessage('assistant', data.response || data.text || '抱歉，我没有理解您的意思。');

        // 播放回复音频
        if (data.audio) {
            playAudio(data.audio);
        } else if (data.audio_base64) {
            const audioBlob = base64ToBlob(data.audio_base64, 'audio/wav');
            playAudio(audioBlob);
        }

    } catch (error) {
        console.error('发送音频失败:', error);
        removeMessage(loadingMsg);
        addMessage('assistant', '❌ 抱歉，处理您的语音时出现错误，请重试。');
        showToast('发送失败: ' + error.message);
    }
}

// ========================================
// 发送文字消息
// ========================================
async function sendTextMessage() {
    const text = elements.textInput.value.trim();
    if (!text) return;

    // 清空输入框
    elements.textInput.value = '';
    autoResizeTextarea();

    // 会诊模式：只记录，不触发AI回复
    if (window.AppMode && window.AppMode.current === 'consultation' && window.AppMode.sessionId) {
        await window.recordUtterance(text);
        return;
    }

    // 患者模式/医生模式：添加用户消息并获取AI回复
    addMessage('user', text);

    // 显示加载状态
    const loadingMsg = addMessage('assistant', '', { isLoading: true });

    try {
        // 确定当前模式
        const currentMode = (window.AppMode && window.AppMode.current) || 'patient';

        const response = await fetch(`${API_BASE}/dialogue`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: text,
                session_id: sessionId,
                mode: currentMode  // 传递当前模式
            })
        });

        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status}`);
        }

        const data = await response.json();

        // ========================================
        // 处理语音模式切换
        // ========================================
        if (data.mode_switched) {
            const newMode = data.mode;

            // 更新全局模式状态
            if (window.AppMode) {
                window.AppMode.current = newMode;
            }

            // 更新UI按钮状态
            const modeBtns = document.querySelectorAll('.mode-btn');
            modeBtns.forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.mode === newMode) {
                    btn.classList.add('active');
                }
            });

            // 如果切换到会诊模式，启动会诊
            if (newMode === 'consultation' && window.startDoctorConsultation) {
                removeMessage(loadingMsg);
                addMessage('assistant', data.text);
                synthesizeAndPlay(data.text);
                // 延迟启动会诊以确保语音播放
                setTimeout(() => {
                    window.startDoctorConsultation();
                }, 500);
                return;
            }

            // 显示模式切换确认消息
            removeMessage(loadingMsg);
            addMessage('assistant', data.text);
            synthesizeAndPlay(data.text);

            console.log(`语音切换模式: ${data.previous_mode} -> ${newMode}`);
            return;
        }

        // ========================================
        // 正常对话处理
        // ========================================

        // 更新 RAG 状态
        const ragUsed = data.rag_used || false;
        updateRagStatus(ragUsed, data.rag_context);

        // 移除加载消息，添加真实回复
        removeMessage(loadingMsg);
        const responseText = data.response || data.text || '抱歉，我没有理解您的意思。';
        addMessage('assistant', responseText);

        // 请求语音合成并播放
        synthesizeAndPlay(responseText);

    } catch (error) {
        console.error('发送消息失败:', error);
        removeMessage(loadingMsg);
        addMessage('assistant', '❌ 抱歉，处理您的消息时出现错误，请重试。');
        showToast('发送失败: ' + error.message);
    }
}

// ========================================
// 语音合成
// ========================================
async function synthesizeAndPlay(text) {
    if (!text) return;

    try {
        const response = await fetch(`${API_BASE}/tts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            const audioBlob = await response.blob();
            playAudio(audioBlob);
        }
    } catch (error) {
        console.error('语音合成失败:', error);
    }
}

function playAudio(audioBlob) {
    const audioUrl = URL.createObjectURL(audioBlob);
    elements.audioPlayer.src = audioUrl;
    elements.audioPlayer.play().catch(e => {
        console.error('播放失败:', e);
    });

    // 清理 URL
    elements.audioPlayer.onended = () => {
        URL.revokeObjectURL(audioUrl);
    };
}

// ========================================
// 消息管理
// ========================================
function addMessage(role, text, options = {}) {
    // 移除欢迎消息
    const welcomeMsg = elements.chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🏥';
    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    if (options.isLoading) {
        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">
                    <div class="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        `;
        messageEl.dataset.loading = 'true';
    } else {
        // 来自客户端的消息添加特殊标识
        const clientIndicator = options.fromClient ? '<span class="client-indicator">📱 客户端</span>' : '';

        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${escapeHtml(text)}</div>
                <div class="message-meta">
                    <span>${time}</span>
                    ${options.isVoice ? '<span>🎤 语音</span>' : ''}
                    ${clientIndicator}
                </div>
            </div>
        `;
    }

    elements.chatMessages.appendChild(messageEl);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    return messageEl;
}

function removeMessage(messageEl) {
    if (messageEl && messageEl.parentNode) {
        messageEl.remove();
    }
}

function clearHistory() {
    // 清除 UI 消息
    elements.chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">👋</div>
            <h3>欢迎使用智能医疗语音助手</h3>
            <p>您可以通过语音或文字向我描述您的症状，我会为您提供初步的导诊建议。</p>
            <div class="quick-tips">
                <span class="tip">💡 点击下方麦克风开始录音</span>
                <span class="tip">⌨️ 或在输入框中输入文字</span>
            </div>
        </div>
    `;

    // 重置识别结果
    elements.asrText.textContent = '等待输入...';
    elements.emotionBadge.textContent = '未知';
    elements.emotionBadge.className = 'emotion-badge';
    elements.emotionScore.textContent = '';
    elements.speakerText.textContent = '未识别';
    elements.ragContent.innerHTML = '';
    updateRagStatus(false);

    // 生成新的会话 ID
    sessionId = 'web-' + Date.now();

    showToast('✅ 对话历史已清除');
}

// ========================================
// 更新识别结果
// ========================================
function updateRecognitionResults(data) {
    // ASR 结果
    if (data.asr_text) {
        elements.asrText.textContent = data.asr_text;
        document.getElementById('asr-result').classList.add('active');
        setTimeout(() => {
            document.getElementById('asr-result').classList.remove('active');
        }, 2000);
    }

    // 情感结果
    if (data.emotion) {
        const emotionMap = {
            'neutral': { label: '平静', class: 'neutral' },
            'happy': { label: '开心', class: 'happy' },
            'sad': { label: '悲伤', class: 'sad' },
            'angry': { label: '愤怒', class: 'angry' },
            'fear': { label: '恐惧', class: 'fear' },
            'surprise': { label: '惊讶', class: 'surprise' }
        };

        const emotion = emotionMap[data.emotion] || { label: data.emotion, class: '' };
        elements.emotionBadge.textContent = emotion.label;
        elements.emotionBadge.className = 'emotion-badge ' + emotion.class;

        if (data.emotion_score) {
            elements.emotionScore.textContent = `${(data.emotion_score * 100).toFixed(0)}%`;
        }

        document.getElementById('emotion-result').classList.add('active');
        setTimeout(() => {
            document.getElementById('emotion-result').classList.remove('active');
        }, 2000);
    }

    // 声纹结果
    if (data.speaker_id) {
        elements.speakerText.textContent = data.speaker_id === 'unknown' ? '未注册用户' : data.speaker_id;
        if (data.speaker_score) {
            elements.speakerText.textContent += ` (${(data.speaker_score * 100).toFixed(0)}%)`;
        }

        document.getElementById('speaker-result').classList.add('active');
        setTimeout(() => {
            document.getElementById('speaker-result').classList.remove('active');
        }, 2000);
    }
}

// ========================================
// 更新 RAG 状态
// ========================================
function updateRagStatus(isActive, context) {
    const indicator = elements.ragStatus.querySelector('.rag-indicator');
    const statusText = elements.ragStatus.querySelector('span:last-child');

    if (isActive) {
        indicator.classList.add('active');
        indicator.classList.remove('inactive');
        statusText.textContent = '已检索医学知识';

        if (context) {
            // 解析并显示 RAG 上下文
            elements.ragContent.innerHTML = formatRagContext(context);
        }
    } else {
        indicator.classList.remove('active');
        indicator.classList.add('inactive');
        statusText.textContent = '等待查询...';
    }
}

function formatRagContext(context) {
    if (!context) return '';

    // 如果是字符串，尝试分割显示
    if (typeof context === 'string') {
        // 按换行分割，每个作为一个 rag-item
        const items = context.split('\n\n').filter(item => item.trim());

        return items.map(item => {
            const lines = item.split('\n');
            let question = '';
            let answer = '';

            lines.forEach(line => {
                if (line.startsWith('问题：') || line.startsWith('Q:')) {
                    question = line.replace(/^(问题：|Q:)\s*/, '');
                } else if (line.startsWith('答案：') || line.startsWith('A:')) {
                    answer = line.replace(/^(答案：|A:)\s*/, '');
                } else {
                    answer += (answer ? ' ' : '') + line;
                }
            });

            if (question || answer) {
                return `
                    <div class="rag-item">
                        ${question ? `<div class="rag-item-question">Q: ${escapeHtml(question)}</div>` : ''}
                        <div class="rag-item-answer">${escapeHtml(answer || item)}</div>
                    </div>
                `;
            }

            return `<div class="rag-item"><div class="rag-item-answer">${escapeHtml(item)}</div></div>`;
        }).join('');
    }

    // 如果是数组
    if (Array.isArray(context)) {
        return context.map(item => `
            <div class="rag-item">
                <div class="rag-item-answer">${escapeHtml(item.content || item)}</div>
                ${item.score ? `<div class="rag-item-score">相似度: ${(item.score * 100).toFixed(1)}%</div>` : ''}
            </div>
        `).join('');
    }

    return escapeHtml(String(context));
}

// ========================================
// 声纹注册
// ========================================
async function registerSpeaker() {
    const speakerId = elements.speakerIdInput.value.trim();
    if (!speakerId) {
        showToast('⚠️ 请输入姓名或ID');
        return;
    }

    showToast('🎤 请说话进行声纹注册...');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        const recorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        const chunks = [];

        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
        };

        recorder.onstop = async () => {
            stream.getTracks().forEach(track => track.stop());

            const audioBlob = new Blob(chunks, { type: 'audio/webm' });

            try {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'register.webm');
                formData.append('speaker_id', speakerId);

                const response = await fetch(`${API_BASE}/speaker/register`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    showToast(`✅ 声纹注册成功: ${speakerId}`);
                    elements.speakerIdInput.value = '';
                } else {
                    showToast(`❌ 注册失败: ${data.error || '未知错误'}`);
                }
            } catch (error) {
                showToast(`❌ 注册失败: ${error.message}`);
            }
        };

        recorder.start();

        // 3秒后停止录音
        setTimeout(() => {
            if (recorder.state === 'recording') {
                recorder.stop();
                showToast('⏹️ 录音结束，正在注册...');
            }
        }, 3000);

    } catch (error) {
        showToast('❌ 无法访问麦克风');
        console.error('Registration error:', error);
    }
}

// ========================================
// 工具函数
// ========================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function autoResizeTextarea() {
    elements.textInput.style.height = 'auto';
    elements.textInput.style.height = Math.min(elements.textInput.scrollHeight, 150) + 'px';
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function showToast(message) {
    // 移除已存在的 toast
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    // 3秒后自动移除
    setTimeout(() => {
        toast.remove();
    }, 3000);
}
