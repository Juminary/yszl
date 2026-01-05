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
                    <button onclick="showPasteDialogueModal()" class="btn-icon" title="粘贴对话">📝</button>
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
// 粘贴对话生成病历
// ========================================
function showPasteDialogueModal() {
    // 移除已存在的模态框
    const existing = document.getElementById('paste-dialogue-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'paste-dialogue-modal';
    modal.className = 'paste-dialogue-modal';
    modal.innerHTML = `
        <div class="paste-modal-content">
            <div class="paste-modal-header">
                <h2>📝 粘贴对话记录</h2>
                <button onclick="closePasteDialogueModal()" class="btn-close">✕</button>
            </div>
            <div class="paste-modal-body">
                <p class="paste-hint">请粘贴医患对话记录，每行一句，格式如下：</p>
                <div class="paste-example">
                    <code>患者：我头疼了三天，还有点发烧</code><br>
                    <code>医生：有没有其他症状？比如咳嗽、流鼻涕？</code><br>
                    <code>患者：有一点咳嗽</code><br>
                    <code>家属：他昨天晚上体温到了38.5度</code>
                </div>
                <textarea id="dialogue-text-input" class="dialogue-textarea" rows="10" 
                    placeholder="在此粘贴对话记录...&#10;&#10;患者：我最近感觉头很疼&#10;医生：疼了多久了？&#10;患者：大概三天了&#10;家属：他还有点发烧"></textarea>
            </div>
            <div class="paste-modal-footer">
                <button onclick="closePasteDialogueModal()" class="btn-secondary">取消</button>
                <button onclick="generateSOAPFromText()" class="btn-primary">🏥 生成病历</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // 聚焦到文本框
    setTimeout(() => {
        document.getElementById('dialogue-text-input').focus();
    }, 100);
}

function closePasteDialogueModal() {
    const modal = document.getElementById('paste-dialogue-modal');
    if (modal) modal.remove();
}

async function generateSOAPFromText() {
    const textInput = document.getElementById('dialogue-text-input');
    const dialogueText = textInput.value.trim();

    if (!dialogueText) {
        showToast('⚠️ 请输入对话记录');
        return;
    }

    // 显示加载状态
    const generateBtn = document.querySelector('#paste-dialogue-modal .btn-primary');
    const originalText = generateBtn.textContent;
    generateBtn.textContent = '⏳ 生成中...';
    generateBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/aci/generate-soap`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dialogue_text: dialogueText })
        });

        const data = await response.json();

        if (data.status === 'success' && data.soap) {
            // 关闭粘贴模态框
            closePasteDialogueModal();

            // 显示SOAP结果
            showGeneratedSOAPModal(data.soap, dialogueText);

            showToast('✅ 病历生成成功');
        } else {
            showToast('❌ 生成失败: ' + (data.error || '未知错误'));
        }
    } catch (error) {
        console.error('生成SOAP失败:', error);
        showToast('❌ 生成失败: ' + error.message);
    } finally {
        generateBtn.textContent = originalText;
        generateBtn.disabled = false;
    }
}

function showGeneratedSOAPModal(soap, originalText) {
    const modal = document.createElement('div');
    modal.className = 'soap-modal generated-soap-modal';

    // 格式化SOAP内容
    const subjective = soap.subjective || {};
    const objective = soap.objective || {};
    const assessment = soap.assessment || {};
    const plan = soap.plan || {};
    const entities = soap.entities || [];

    // 提取实体标签
    const symptoms = entities.filter(e => e.type === 'symptom').map(e => e.text);
    const diseases = entities.filter(e => e.type === 'disease').map(e => e.text);
    const medications = entities.filter(e => e.type === 'medication').map(e => e.text);

    modal.innerHTML = `
        <div class="soap-modal-content">
            <div class="soap-modal-header">
                <h2>🏥 生成的 SOAP 病历</h2>
                <button onclick="this.closest('.soap-modal').remove()" class="btn-close">✕</button>
            </div>
            <div class="soap-modal-body">
                <section class="soap-result-section">
                    <h3>📋 S - 主诉 (Subjective)</h3>
                    <p><strong>主诉：</strong>${subjective.chief_complaint || '未记录'}</p>
                    ${subjective.history ? `<p><strong>病史：</strong>${subjective.history}</p>` : ''}
                </section>
                
                <section class="soap-result-section">
                    <h3>🔬 O - 客观检查 (Objective)</h3>
                    <p><strong>生命体征：</strong>${objective.vital_signs || '待检查'}</p>
                    <p>${objective.content || '暂无客观检查数据'}</p>
                </section>
                
                <section class="soap-result-section">
                    <h3>🩺 A - 评估 (Assessment)</h3>
                    <p><strong>诊断：</strong>${assessment.diagnosis || '待诊断'}</p>
                    <p>${assessment.content || ''}</p>
                </section>
                
                <section class="soap-result-section">
                    <h3>💊 P - 计划 (Plan)</h3>
                    <p><strong>治疗方案：</strong>${plan.treatment || '待制定'}</p>
                    <p>${plan.content || ''}</p>
                </section>

                ${entities.length > 0 ? `
                <section class="soap-result-section entities-section">
                    <h3>🏷️ 提取的医学实体</h3>
                    <div class="entity-tags">
                        ${symptoms.length > 0 ? `<div class="entity-group"><span class="entity-label">症状:</span> ${symptoms.map(s => `<span class="entity-tag symptom">${s}</span>`).join('')}</div>` : ''}
                        ${diseases.length > 0 ? `<div class="entity-group"><span class="entity-label">疾病:</span> ${diseases.map(d => `<span class="entity-tag disease">${d}</span>`).join('')}</div>` : ''}
                        ${medications.length > 0 ? `<div class="entity-group"><span class="entity-label">药物:</span> ${medications.map(m => `<span class="entity-tag medication">${m}</span>`).join('')}</div>` : ''}
                    </div>
                </section>
                ` : ''}
            </div>
            <div class="soap-modal-footer">
                <button onclick="copySOAPToClipboard(this)" class="btn-secondary" data-soap='${JSON.stringify(soap).replace(/'/g, "\\'")}'>📋 复制</button>
                <button onclick="downloadSOAPAsMarkdown(this)" class="btn-secondary" data-soap='${JSON.stringify(soap).replace(/'/g, "\\'")}'>📥 MD</button>
                <button onclick="downloadSOAPAsHTML(this)" class="btn-secondary" data-soap='${JSON.stringify(soap).replace(/'/g, "\\'")}'>🏥 电子病历</button>
                <button onclick="this.closest('.soap-modal').remove()" class="btn-primary">确定</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function copySOAPToClipboard(btn) {
    const soap = JSON.parse(btn.dataset.soap);
    const text = formatSOAPAsText(soap);

    navigator.clipboard.writeText(text).then(() => {
        showToast('✅ 已复制到剪贴板');
    }).catch(err => {
        console.error('复制失败:', err);
        showToast('❌ 复制失败');
    });
}

function downloadSOAPAsMarkdown(btn) {
    const soap = JSON.parse(btn.dataset.soap);
    const markdown = formatSOAPAsMarkdown(soap);

    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SOAP_病历_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('✅ 已导出');
}

function formatSOAPAsText(soap) {
    const s = soap.subjective || {};
    const o = soap.objective || {};
    const a = soap.assessment || {};
    const p = soap.plan || {};

    return `SOAP 病历
============

【S - 主诉】
主诉：${s.chief_complaint || '未记录'}
${s.history ? `病史：${s.history}` : ''}

【O - 客观检查】
生命体征：${o.vital_signs || '待检查'}
${o.content || '暂无客观检查数据'}

【A - 评估】
诊断：${a.diagnosis || '待诊断'}
${a.content || ''}

【P - 计划】
治疗方案：${p.treatment || '待制定'}
${p.content || ''}

生成时间：${new Date().toLocaleString('zh-CN')}
`;
}

function formatSOAPAsMarkdown(soap) {
    const s = soap.subjective || {};
    const o = soap.objective || {};
    const a = soap.assessment || {};
    const p = soap.plan || {};
    const entities = soap.entities || [];

    let md = `# SOAP 病历

## S - 主诉 (Subjective)

**主诉：** ${s.chief_complaint || '未记录'}

${s.history ? `**病史：** ${s.history}` : ''}

## O - 客观检查 (Objective)

**生命体征：** ${o.vital_signs || '待检查'}

${o.content || '暂无客观检查数据'}

## A - 评估 (Assessment)

**诊断：** ${a.diagnosis || '待诊断'}

${a.content || ''}

## P - 计划 (Plan)

**治疗方案：** ${p.treatment || '待制定'}

${p.content || ''}
`;

    if (entities.length > 0) {
        const symptoms = entities.filter(e => e.type === 'symptom').map(e => e.text);
        const diseases = entities.filter(e => e.type === 'disease').map(e => e.text);
        const medications = entities.filter(e => e.type === 'medication').map(e => e.text);

        md += `
## 提取的医学实体

${symptoms.length > 0 ? `- **症状：** ${symptoms.join('、')}` : ''}
${diseases.length > 0 ? `- **疾病：** ${diseases.join('、')}` : ''}
${medications.length > 0 ? `- **药物：** ${medications.join('、')}` : ''}
`;
    }

    md += `
---
*生成时间：${new Date().toLocaleString('zh-CN')}*
`;

    return md;
}

// ========================================
// HTML 电子病历生成
// ========================================
function downloadSOAPAsHTML(btn) {
    const soap = JSON.parse(btn.dataset.soap);
    const html = formatSOAPAsHTML(soap);

    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `电子病历_${new Date().toISOString().slice(0, 10)}.html`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('✅ 电子病历已导出');
}

function formatSOAPAsHTML(soap) {
    const s = soap.subjective || {};
    const o = soap.objective || {};
    const a = soap.assessment || {};
    const p = soap.plan || {};
    const entities = soap.entities || [];

    const now = new Date();
    const dateStr = now.toLocaleDateString('zh-CN');
    const timeStr = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    // 提取实体
    const symptoms = entities.filter(e => e.type === 'symptom').map(e => e.text);
    const diseases = entities.filter(e => e.type === 'disease').map(e => e.text);
    const medications = entities.filter(e => e.type === 'medication').map(e => e.text);

    return `<!DOCTYPE html>
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
                <span class="info-value">${timeStr}</span>
            </div>
            <div class="info-item">
                <span class="info-label">病历号：</span>
                <span class="info-value">${Math.random().toString(36).substr(2, 8).toUpperCase()}</span>
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
                    ${s.chief_complaint || s.chief_complaint_text || '未记录'}
                </div>
                <div class="content-row">
                    <span class="content-label">现病史：</span>
                    ${s.history || s.history_present_illness || '未记录'}
                </div>
                ${symptoms.length > 0 ? `
                <div class="entity-tags">
                    <span class="content-label">症状标签：</span>
                    ${symptoms.map(s => `<span class="entity-tag symptom">${s}</span>`).join('')}
                </div>
                ` : ''}
            </div>
        </div>

        <div class="section">
            <div class="section-title">二、体格检查 (Objective)</div>
            <div class="section-content">
                <div class="content-row">
                    <span class="content-label">生命体征：</span>
                    ${o.vital_signs || '待检查'}
                </div>
                <div class="content-row">
                    <span class="content-label">体格检查：</span>
                    ${o.content || o.physical_exam || '待检查'}
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">三、诊断意见 (Assessment)</div>
            <div class="section-content">
                <div class="content-row">
                    <span class="content-label">初步诊断：</span>
                    ${a.diagnosis || (a.diagnoses && a.diagnoses.join('、')) || '待诊断'}
                </div>
                <div class="content-row">
                    <span class="content-label">病情评估：</span>
                    ${a.content || a.severity || '待评估'}
                </div>
                ${diseases.length > 0 ? `
                <div class="entity-tags">
                    <span class="content-label">疾病标签：</span>
                    ${diseases.map(d => `<span class="entity-tag disease">${d}</span>`).join('')}
                </div>
                ` : ''}
            </div>
        </div>

        <div class="section">
            <div class="section-title">四、治疗方案 (Plan)</div>
            <div class="section-content">
                <div class="content-row">
                    <span class="content-label">治疗方案：</span>
                    ${p.treatment || '待制定'}
                </div>
                <div class="content-row">
                    <span class="content-label">医　　嘱：</span>
                    ${p.content || p.instructions || '遵医嘱'}
                </div>
                ${medications.length > 0 ? `
                <div class="entity-tags">
                    <span class="content-label">用药建议：</span>
                    ${medications.map(m => `<span class="entity-tag medication">${m}</span>`).join('')}
                </div>
                ` : ''}
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
            <p>如有疑问请咨询专业医生 | 生成时间：${now.toLocaleString('zh-CN')}</p>
        </div>
    </div>
</body>
</html>`;
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
window.showPasteDialogueModal = showPasteDialogueModal;
window.closePasteDialogueModal = closePasteDialogueModal;
window.generateSOAPFromText = generateSOAPFromText;
window.copySOAPToClipboard = copySOAPToClipboard;
window.downloadSOAPAsMarkdown = downloadSOAPAsMarkdown;
window.downloadSOAPAsHTML = downloadSOAPAsHTML;
