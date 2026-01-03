/**
 * 多模式系统 - 支持患者模式、医生模式、会诊模式
 */

// ========================================
// 模式定义
// ========================================
const AppMode = {
    current: 'patient',  // patient | doctor | consultation
    sessionId: null,
    updateInterval: null,
    currentRole: 'patient'  // 会诊模式中的当前角色
};

const MODE_CONFIG = {
    patient: {
        name: '患者模式',
        icon: '🧑',
        description: 'AI 帮助您了解症状并提供导诊建议',
        aiEnabled: true,
        systemPrompt: '你是一个医疗导诊助手，帮助患者了解症状并建议应该挂什么科室。'
    },
    doctor: {
        name: '医生模式',
        icon: '👨‍⚕️',
        description: 'AI 辅助诊断，提供鉴别诊断和治疗建议',
        aiEnabled: true,
        systemPrompt: '你是医生的AI助手，帮助分析病情、提供鉴别诊断和治疗方案建议。使用专业医学术语。'
    },
    consultation: {
        name: '会诊模式',
        icon: '📝',
        description: '记录医患对话，自动生成 SOAP 病历',
        aiEnabled: false,
        systemPrompt: null
    }
};

// ========================================
// 初始化
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initModeSystem, 1000);
});

async function initModeSystem() {
    // 检查 ACI 模块
    try {
        const response = await fetch(`${API_BASE}/aci/status`);
        const data = await response.json();

        if (data.available) {
            console.log('[模式系统] ACI 可用');
            addModeSelector();
        }
    } catch (error) {
        console.error('[模式系统] 初始化失败:', error);
    }
}

function addModeSelector() {
    const header = document.querySelector('.chat-header');
    if (!header || document.getElementById('mode-selector')) return;

    const selector = document.createElement('div');
    selector.id = 'mode-selector';
    selector.className = 'mode-selector';
    selector.innerHTML = `
        <button class="mode-btn active" data-mode="patient" onclick="switchMode('patient')">
            🧑 患者
        </button>
        <button class="mode-btn" data-mode="doctor" onclick="switchMode('doctor')">
            👨‍⚕️ 医生
        </button>
        <button class="mode-btn" data-mode="consultation" onclick="switchMode('consultation')">
            📝 会诊
        </button>
    `;

    header.querySelector('.header-actions').prepend(selector);
}

// ========================================
// 模式切换
// ========================================
function switchMode(mode) {
    if (AppMode.current === mode) return;

    // 如果正在会诊，先结束
    if (AppMode.current === 'consultation' && AppMode.sessionId) {
        endConsultationSession();
    }

    AppMode.current = mode;

    // 更新按钮状态
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // 更新 body class
    document.body.className = `mode-${mode}`;

    // 根据模式更新 UI
    const config = MODE_CONFIG[mode];
    showToast(`✅ 已切换到${config.name}`);

    if (mode === 'consultation') {
        showConsultationUI();
    } else {
        hideConsultationUI();
        showModeWelcome(mode);
    }
}

// ========================================
// 模式欢迎界面
// ========================================
function showModeWelcome(mode) {
    const config = MODE_CONFIG[mode];
    const chatMessages = document.getElementById('chat-messages');

    let tips = '';
    if (mode === 'patient') {
        tips = `
            <span>💡 描述您的症状，我会帮您分析</span>
            <span>💡 告诉您应该挂什么科室</span>
            <span>💡 提供初步的健康建议</span>
        `;
    } else if (mode === 'doctor') {
        tips = `
            <span>💡 输入患者症状进行鉴别诊断</span>
            <span>💡 获取检查方案建议</span>
            <span>💡 查阅药物和治疗方案</span>
        `;
    }

    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">${config.icon}</div>
            <h3>${config.name}</h3>
            <p>${config.description}</p>
            <div class="quick-tips">${tips}</div>
        </div>
    `;
}

// ========================================
// 会诊模式 UI
// ========================================
function showConsultationUI() {
    showRoleSelector();
    showSOAPPanel();
    startConsultationSession();
}

function hideConsultationUI() {
    hideRoleSelector();
    hideSOAPPanel();
}

function showRoleSelector() {
    let selector = document.getElementById('role-selector');
    if (!selector) {
        selector = document.createElement('div');
        selector.id = 'role-selector';
        selector.className = 'role-selector';
        selector.innerHTML = `
            <span class="role-label">记录说话人：</span>
            <button class="role-btn active" data-role="patient" onclick="setRole('patient')">🧑 患者</button>
            <button class="role-btn" data-role="doctor" onclick="setRole('doctor')">👨‍⚕️ 医生</button>
            <button class="role-btn" data-role="family" onclick="setRole('family')">👨‍👩‍👧 家属</button>
        `;
        const inputArea = document.querySelector('.chat-input-area');
        inputArea.insertBefore(selector, inputArea.firstChild);
    }
    selector.style.display = 'flex';
    updateInputPlaceholder();
    showConsultationWelcome();
}

function hideRoleSelector() {
    const selector = document.getElementById('role-selector');
    if (selector) selector.style.display = 'none';
    const textInput = document.getElementById('text-input');
    if (textInput) textInput.placeholder = '输入您的问题...';
}

function setRole(role) {
    AppMode.currentRole = role;
    document.querySelectorAll('#role-selector .role-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.role === role);
    });
    updateInputPlaceholder();
}

function updateInputPlaceholder() {
    const textInput = document.getElementById('text-input');
    if (!textInput) return;
    const placeholders = {
        patient: '输入患者说的话...',
        doctor: '输入医生说的话...',
        family: '输入家属说的话...'
    };
    textInput.placeholder = placeholders[AppMode.currentRole] || '输入对话内容...';
}

function showConsultationWelcome() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
        <div class="scribe-welcome">
            <div class="scribe-icon">📝</div>
            <h3>会诊记录模式</h3>
            <p>记录真实的医患对话，自动生成 SOAP 病历</p>
            <div class="scribe-tips">
                <span>💡 选择说话人角色后输入对话内容</span>
                <span>💡 系统自动提取症状、诊断、药物信息</span>
                <span>💡 右侧实时生成 SOAP 病历</span>
            </div>
        </div>
    `;
}

// ========================================
// SOAP 面板
// ========================================
function showSOAPPanel() {
    let panel = document.getElementById('soap-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'soap-panel';
        panel.className = 'soap-panel';
        panel.innerHTML = `
            <div class="soap-header">
                <h3>📋 SOAP 病历</h3>
                <div class="soap-actions">
                    <button onclick="refreshSOAP()" class="btn-icon" title="刷新">🔄</button>
                    <button onclick="exportSOAP()" class="btn-icon" title="导出">📥</button>
                    <button onclick="endConsultationSession()" class="btn-icon btn-danger" title="结束">⏹️</button>
                </div>
            </div>
            <div class="soap-content">
                <div class="soap-section">
                    <h4>S - 主诉</h4>
                    <div class="soap-section-content" id="soap-s-content"><span class="muted">等待患者描述...</span></div>
                </div>
                <div class="soap-section">
                    <h4>O - 客观检查</h4>
                    <div class="soap-section-content" id="soap-o-content"><span class="muted">等待检查记录...</span></div>
                </div>
                <div class="soap-section">
                    <h4>A - 评估</h4>
                    <div class="soap-section-content" id="soap-a-content"><span class="muted">等待诊断...</span></div>
                </div>
                <div class="soap-section">
                    <h4>P - 计划</h4>
                    <div class="soap-section-content" id="soap-p-content"><span class="muted">等待治疗方案...</span></div>
                </div>
            </div>
            <div class="soap-footer">
                <div class="soap-stats" id="soap-stats"><span>对话: 0</span></div>
            </div>
        `;
        const mainArea = document.querySelector('.chat-main');
        mainArea.parentNode.insertBefore(panel, mainArea.nextSibling);
    }
    panel.style.display = 'flex';
}

function hideSOAPPanel() {
    const panel = document.getElementById('soap-panel');
    if (panel) panel.style.display = 'none';
}

// ========================================
// 会诊会话管理
// ========================================
async function startConsultationSession() {
    try {
        const response = await fetch(`${API_BASE}/consultation/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_info: {} })
        });

        const data = await response.json();
        if (data.status === 'success') {
            AppMode.sessionId = data.session_id;
            console.log('[会诊] 开始:', data.session_id);
            showToast('✅ 开始记录会诊');
            AppMode.updateInterval = setInterval(refreshSOAP, 5000);
        }
    } catch (error) {
        console.error('[会诊] 开始失败:', error);
    }
}

async function endConsultationSession() {
    if (!AppMode.sessionId) return;

    if (AppMode.updateInterval) {
        clearInterval(AppMode.updateInterval);
        AppMode.updateInterval = null;
    }

    try {
        await fetch(`${API_BASE}/consultation/${AppMode.sessionId}/end`, { method: 'POST' });
        showToast('✅ 会诊已结束');

        // 显示最终 SOAP
        const response = await fetch(`${API_BASE}/consultation/${AppMode.sessionId}/soap`);
        const data = await response.json();
        if (data.status === 'success' && data.soap) {
            showSOAPModal(data.soap);
        }

        AppMode.sessionId = null;
    } catch (error) {
        console.error('[会诊] 结束失败:', error);
    }
}

// ========================================
// 记录对话（会诊模式专用）
// ========================================
async function recordUtterance(text) {
    if (!AppMode.sessionId || !text.trim()) return;

    addScribeMessage(AppMode.currentRole, text);

    try {
        const response = await fetch(`${API_BASE}/consultation/${AppMode.sessionId}/utterance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, speaker_role: AppMode.currentRole })
        });

        const data = await response.json();

        if (data.emergency_alert && data.emergency_alert.level === 'critical') {
            showEmergencyAlert(data.emergency_alert);
        }

        refreshSOAP();

    } catch (error) {
        console.error('[会诊] 记录失败:', error);
    }
}

function addScribeMessage(role, text) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const welcome = chatMessages.querySelector('.scribe-welcome, .welcome-message');
    if (welcome) welcome.remove();

    const config = {
        patient: { label: '患者', icon: '🧑', class: 'patient' },
        doctor: { label: '医生', icon: '👨‍⚕️', class: 'doctor' },
        family: { label: '家属', icon: '👨‍👩‍👧', class: 'family' }
    }[role] || { label: '未知', icon: '❓', class: '' };

    const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    const el = document.createElement('div');
    el.className = `message scribe-message ${config.class}`;
    el.innerHTML = `
        <div class="message-avatar">${config.icon}</div>
        <div class="message-content">
            <div class="message-role">${config.label}</div>
            <div class="message-text">${escapeHtmlMode(text)}</div>
            <div class="message-meta"><span>${time}</span></div>
        </div>
    `;

    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ========================================
// SOAP 预览更新
// ========================================
async function refreshSOAP() {
    if (!AppMode.sessionId) return;

    try {
        const response = await fetch(`${API_BASE}/consultation/${AppMode.sessionId}/preview`);
        const data = await response.json();

        if (data.status === 'success') {
            const p = data.preview;

            const sContent = document.getElementById('soap-s-content');
            if (sContent) {
                const symptoms = p.subjective?.symptoms || [];
                sContent.innerHTML = symptoms.length > 0
                    ? symptoms.map(s => `<span class="entity-tag symptom">${s}</span>`).join('')
                    : '<span class="muted">等待患者描述...</span>';
            }

            const oContent = document.getElementById('soap-o-content');
            if (oContent) {
                const procs = p.objective?.procedures || [];
                oContent.innerHTML = procs.length > 0
                    ? procs.map(s => `<span class="entity-tag procedure">${s}</span>`).join('')
                    : '<span class="muted">等待检查记录...</span>';
            }

            const aContent = document.getElementById('soap-a-content');
            if (aContent) {
                const diseases = p.assessment?.diseases || [];
                aContent.innerHTML = diseases.length > 0
                    ? diseases.map(s => `<span class="entity-tag disease">${s}</span>`).join('')
                    : '<span class="muted">等待诊断...</span>';
            }

            const pContent = document.getElementById('soap-p-content');
            if (pContent) {
                const meds = p.plan?.medications || [];
                pContent.innerHTML = meds.length > 0
                    ? meds.map(s => `<span class="entity-tag medication">${s}</span>`).join('')
                    : '<span class="muted">等待治疗方案...</span>';
            }

            const stats = document.getElementById('soap-stats');
            if (stats && p.statistics) {
                stats.innerHTML = `<span>对话: ${p.statistics.utterance_count || 0}</span>`;
            }
        }
    } catch (error) {
        console.error('[SOAP] 刷新失败:', error);
    }
}

// ========================================
// SOAP 导出和模态框
// ========================================
async function exportSOAP() {
    if (!AppMode.sessionId) {
        showToast('⚠️ 没有活动的会诊');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/consultation/${AppMode.sessionId}/soap?format=markdown`);
        const markdown = await response.text();

        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `SOAP_${AppMode.sessionId}_${new Date().toISOString().slice(0, 10)}.md`;
        a.click();
        URL.revokeObjectURL(url);

        showToast('✅ 已导出');
    } catch (error) {
        showToast('❌ 导出失败');
    }
}

function showSOAPModal(soap) {
    const modal = document.createElement('div');
    modal.className = 'soap-modal';
    modal.innerHTML = `
        <div class="soap-modal-content">
            <div class="soap-modal-header">
                <h2>📋 SOAP 病历</h2>
                <button onclick="this.closest('.soap-modal').remove()" class="btn-close">✕</button>
            </div>
            <div class="soap-modal-body">
                <section><h3>S - 主诉</h3><p>${soap.subjective?.chief_complaint || '未记录'}</p></section>
                <section><h3>O - 检查</h3><p>${soap.objective?.test_results || '未记录'}</p></section>
                <section><h3>A - 评估</h3><p>${(soap.assessment?.diagnoses || []).join('、') || '未记录'}</p></section>
                <section><h3>P - 计划</h3><p>${(soap.plan?.medications || []).join('、') || '未记录'}</p></section>
            </div>
            <div class="soap-modal-footer">
                <button onclick="exportSOAP()" class="btn-primary">📥 导出</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

// ========================================
// 急救警报
// ========================================
function showEmergencyAlert(alert) {
    const modal = document.createElement('div');
    modal.className = 'emergency-modal';
    modal.innerHTML = `
        <div class="emergency-modal-content">
            <div class="emergency-header">
                <span class="emergency-icon">🚨</span>
                <h2>危急症状</h2>
            </div>
            <div class="emergency-body">
                <p>${alert.message}</p>
            </div>
            <div class="emergency-actions">
                <button class="emergency-action-btn" onclick="window.location.href='tel:120'">🚑 拨打120</button>
                <button class="emergency-dismiss-btn" onclick="this.closest('.emergency-modal').remove()">知道了</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

// ========================================
// 工具函数
// ========================================
function escapeHtmlMode(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ========================================
// 导出
// ========================================
window.AppMode = AppMode;
window.MODE_CONFIG = MODE_CONFIG;
window.switchMode = switchMode;
window.setRole = setRole;
window.recordUtterance = recordUtterance;
window.refreshSOAP = refreshSOAP;
window.exportSOAP = exportSOAP;
window.endConsultationSession = endConsultationSession;
