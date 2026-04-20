/* global state */
let sessionId = `session_${Date.now()}`;
let isLoading = false;

/* ===== DOM helpers ===== */
const $ = (id) => document.getElementById(id);

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/** Lightweight markdown → HTML conversion for common patterns */
function simpleMarkdown(text) {
    // Escape HTML first
    text = escapeHtml(text);

    // Code blocks (```…```)
    text = text.replace(/```[\s\S]*?```/g, (m) => `<pre><code>${m.slice(3, -3).trim()}</code></pre>`);
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Unordered list items (- item or * item)
    const lines = text.split('\n');
    const result = [];
    let inUl = false;
    let inOl = false;
    let olIdx = 0;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const ulMatch = line.match(/^[-*]\s+(.+)/);
        const olMatch = line.match(/^\d+\.\s+(.+)/);

        if (ulMatch) {
            if (!inUl) { result.push('<ul>'); inUl = true; }
            if (inOl) { result.push('</ol>'); inOl = false; }
            result.push(`<li>${ulMatch[1]}</li>`);
        } else if (olMatch) {
            if (!inOl) { result.push('<ol>'); inOl = true; }
            if (inUl) { result.push('</ul>'); inUl = false; }
            result.push(`<li>${olMatch[1]}</li>`);
        } else {
            if (inUl) { result.push('</ul>'); inUl = false; }
            if (inOl) { result.push('</ol>'); inOl = false; }
            // Headings
            const h3 = line.match(/^###\s+(.+)/);
            const h2 = line.match(/^##\s+(.+)/);
            const h1 = line.match(/^#\s+(.+)/);
            if (h3) result.push(`<h4>${h3[1]}</h4>`);
            else if (h2) result.push(`<h3>${h2[1]}</h3>`);
            else if (h1) result.push(`<h2>${h1[1]}</h2>`);
            else if (line.trim()) result.push(`<p>${line}</p>`);
        }
    }
    if (inUl) result.push('</ul>');
    if (inOl) result.push('</ol>');

    return result.join('\n');
}

/* ===== Render message ===== */
function addMessage(role, content, isError = false) {
    const container = $('chatMessages');
    const el = document.createElement('div');
    el.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🍊';
    const html = isError
        ? `<div class="error-bubble">${escapeHtml(content)}</div>`
        : `<div class="bubble">${simpleMarkdown(content)}</div>`;

    el.innerHTML = `<div class="avatar">${avatar}</div>${html}`;
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
    return el;
}

/* ===== Typing indicator ===== */
let typingEl = null;
function showTyping() {
    const container = $('chatMessages');
    typingEl = document.createElement('div');
    typingEl.className = 'message assistant';
    typingEl.innerHTML = `
        <div class="avatar">🍊</div>
        <div class="bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    container.appendChild(typingEl);
    container.scrollTop = container.scrollHeight;
}
function hideTyping() {
    if (typingEl) { typingEl.remove(); typingEl = null; }
}

/* ===== Send message ===== */
async function sendMessage() {
    if (isLoading) return;

    const input = $('userInput');
    const question = input.value.trim();
    if (!question) return;

    input.value = '';
    input.style.height = 'auto';
    isLoading = true;
    $('btnSend').disabled = true;

    addMessage('user', question);
    showTyping();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: sessionId }),
        });

        hideTyping();

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            addMessage('assistant', err.detail || `请求失败（${res.status}），请稍后重试。`, true);
        } else {
            const data = await res.json();
            addMessage('assistant', data.answer);
        }
    } catch (err) {
        hideTyping();
        addMessage('assistant', `网络错误：${err.message}，请检查网络连接后重试。`, true);
    } finally {
        isLoading = false;
        $('btnSend').disabled = false;
        input.focus();
    }
}

/* ===== Fill question from example ===== */
function fillQuestion(text) {
    const input = $('userInput');
    input.value = text;
    input.focus();
}

/* ===== Load topics ===== */
async function loadTopics() {
    try {
        const res = await fetch('/api/topics');
        if (!res.ok) return;
        const data = await res.json();
        const list = $('topicList');
        list.innerHTML = '';
        data.topics.forEach((topic) => {
            const li = document.createElement('li');
            li.innerHTML = `<span class="topic-icon">${topic.icon}</span>${topic.name}`;
            li.title = `点击搜索"${topic.name}"相关内容`;
            li.addEventListener('click', () => {
                document.querySelectorAll('.topic-list li').forEach((el) => el.classList.remove('active'));
                li.classList.add('active');
                fillQuestion(`请介绍脐橙的${topic.name}`);
            });
            list.appendChild(li);
        });
    } catch (_) {
        // Topics are optional
    }
}

/* ===== Clear session ===== */
async function clearSession() {
    if (!confirm('确定要清空当前对话记录吗？')) return;
    try {
        await fetch(`/api/session/${sessionId}`, { method: 'DELETE' });
    } catch (_) { /* ignore */ }

    sessionId = `session_${Date.now()}`;
    const container = $('chatMessages');
    container.innerHTML = '';
    addMessage('assistant', '对话已清空，请重新提问！😊');
}

/* ===== Event listeners ===== */
document.addEventListener('DOMContentLoaded', () => {
    loadTopics();

    $('btnSend').addEventListener('click', sendMessage);
    $('btnClear').addEventListener('click', clearSession);

    $('userInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    $('userInput').addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
});
