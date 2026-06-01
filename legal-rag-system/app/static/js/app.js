/* ========== 法律RAG多智能体系统 - 前端应用 ========== */

// ---- 配置 ----
const API_BASE = '/api/v1';

// ---- 状态管理 ----
const state = {
  conversationId: null,
  currentMode: 'react',
  messages: [],
  uploadFiles: [],
  isProcessing: false,
  abortController: null,
  _lastQuery: '',
};

// ---- DOM 引用 ----
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function getDom() {
  return {
    messages: $('#chat-messages'),
    input: $('#chat-input'),
    sendBtn: $('#btn-send'),
    modeBtns: $$('.mode-btn'),
    modeLabel: $('#current-mode-label'),
    convId: $('#conversation-id'),
    convList: $('#conversation-list'),
    newConvBtn: $('#new-conversation'),
    clearChatBtn: $('#btn-clear-chat'),
    uploadBtn: $('#btn-upload'),
    uploadModal: $('#upload-modal'),
    uploadDropzone: $('#upload-dropzone'),
    uploadFileInput: $('#upload-file-input'),
    uploadDocType: $('#upload-doc-type'),
    uploadPreview: $('#upload-preview'),
    uploadProgress: $('#upload-progress'),
    uploadConfirm: $('#btn-upload-confirm'),
    citationPanel: $('#citation-panel'),
    citationContent: $('#citation-content'),
    fileInput: $('#file-input'),
    kbDocList: $('#kb-doc-list'),
    kbRefreshBtn: $('#kb-refresh-btn'),
  };
}

// ---- 初始化 ----
function init() {
  loadGreeting();
  setupModeButtons();
  setupInput();
  setupUploadModal();
  setupSuggestions();
  setupClearChat();
  setupNewConversation();
  setupCitationDelegation();
  setupCitationPanelClose();
  setupKBManagement();
  refreshKBStats();
  loadConversations();
  setupTools();
  updateConvIdDisplay();
}

// ---- 时间天气问候 ----
let _greetingTimer = null;
let _greetingData = null;  // 缓存最近一次 API 返回数据

function _pad(n) { return n < 10 ? '0' + n : '' + n; }

function _renderGreeting(data, city) {
  const emojiMap = {
    '夜深了': '🌙', '早上好': '🌅', '上午好': '☀️',
    '中午好': '🌤️', '下午好': '🌤️', '晚上好': '🌙',
  };
  const emoji = $('#greeting-emoji');
  const word = $('#greeting-word');
  const date = $('#greeting-date');
  const weather = $('#greeting-weather');

  if (emoji) emoji.textContent = emojiMap[data.greeting] || '👋';
  if (word) word.textContent = `${data.greeting}，今天${data.weekday}`;
  if (date) {
    const now = new Date(Date.now() + (data._serverOffset || 0));
    const hh = _pad(now.getHours()), mm = _pad(now.getMinutes());
    date.textContent = `${data.date} ${hh}:${mm}`;
  }
  if (weather && !weather.dataset.citySet) {
    const parts = data.weather.split(': ');
    const loc = parts.length > 1 ? parts[0] : city;
    const cond = parts.length > 1 ? parts[1] : data.weather;
    weather.innerHTML = `📍 ${loc} <span class="greeting-sep">|</span> 🌡️ ${cond}`;
    weather.title = '点击切换城市';
    weather.style.cursor = 'pointer';
    weather.onclick = async () => {
      const newCity = prompt('输入城市名称（如：北京、上海、深圳）：', localStorage.getItem('weather_city') || '天津');
      if (newCity && newCity.trim()) {
        localStorage.setItem('weather_city', newCity.trim());
        loadGreeting();
      }
    };
    weather.dataset.citySet = '1';
  }
}

function _tickTime() {
  if (!_greetingData) return;
  _renderGreeting(_greetingData);
}

async function loadGreeting() {
  try {
    const city = localStorage.getItem('weather_city') || '天津';
    const res = await fetch(`${API_BASE}/now?city=${encodeURIComponent(city)}`);
    if (!res.ok) return;
    const data = await res.json();

    // 计算服务器时间与本地时间的偏移
    const serverTime = new Date(`${data.date} ${data.time}`);
    data._serverOffset = serverTime.getTime() - Date.now();

    _greetingData = data;
    _renderGreeting(data, city);

    // 每 10 秒刷新时间显示
    if (_greetingTimer) clearInterval(_greetingTimer);
    _greetingTimer = setInterval(_tickTime, 10000);

    // 每 30 分钟全量刷新（问候语+天气可能变化）
    setTimeout(() => { loadGreeting(); }, 30 * 60 * 1000);
  } catch (err) {
    console.error('加载天气信息失败:', err);
  }
}

function updateConvIdDisplay() {
  const id = $('#conversation-id');
  if (id) id.textContent = state.conversationId || '';
}

// ---- 引用事件委托 ----
function setupCitationDelegation() {
  getDom().messages.addEventListener('click', (e) => {
    const btn = e.target.closest('.citation-btn');
    if (!btn) return;
    try {
      const citations = JSON.parse(btn.dataset.citations);
      showCitations(citations);
    } catch (err) {
      console.error('解析引用数据失败:', err);
    }
  });
}

// ---- 推理模式切换 ----
function setupModeButtons() {
  getDom().modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (state.isProcessing) return;
      $$('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentMode = btn.dataset.mode;
      const labels = {
        react: 'ReAct 推理模式',
        plan_solve: 'Plan-Solve 推理模式',
        reflection: 'Reflection 推理模式',
      };
      const label = $('#current-mode-label');
      if (label) label.textContent = labels[state.currentMode] || labels.react;
    });
  });
}

// ---- 输入处理 ----
function setupInput() {
  const d = getDom();
  d.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!state.isProcessing) sendMessage();
    }
  });

  d.input.addEventListener('input', () => {
    d.input.style.height = 'auto';
    d.input.style.height = Math.min(d.input.scrollHeight, 150) + 'px';
  });

  d.sendBtn.addEventListener('click', () => {
    if (!state.isProcessing) sendMessage();
  });

  d.fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 7) {
      alert('一次最多上传 7 个文件');
      e.target.value = '';
      return;
    }
    if (files.length > 0) {
      state.uploadFiles = files;
      showUploadPreviewForChat(files);
    }
  });
}

// ---- 处理状态控制 ----
function setProcessing(processing) {
  state.isProcessing = processing;
  const d = getDom();

  d.input.disabled = processing;
  d.sendBtn.disabled = processing;

  $$('.mode-btn').forEach(btn => { btn.disabled = processing; });
  document.querySelectorAll('.suggestion-btn').forEach(btn => { btn.disabled = processing; });

  if (d.clearChatBtn) d.clearChatBtn.disabled = processing;
  if (d.uploadBtn) d.uploadBtn.disabled = processing;
  if (d.newConvBtn) d.newConvBtn.disabled = processing;

  if (processing) {
    d.sendBtn.textContent = '思考中...';
    d.input.placeholder = '正在生成回答，请稍候...';
  } else {
    d.sendBtn.textContent = '发送';
    d.input.placeholder = '请输入您的法律问题...';
    d.sendBtn.disabled = false;
    d.input.disabled = false;
    $$('.mode-btn').forEach(b => b.disabled = false);
    if (d.clearChatBtn) d.clearChatBtn.disabled = false;
    if (d.uploadBtn) d.uploadBtn.disabled = false;
    if (d.newConvBtn) d.newConvBtn.disabled = false;
  }
}

// ---- 打字机效果队列 ----
let _typewriterTimer = null;
let _typewriterActiveStep = null;
let _scrollPending = false;

function scheduleScroll() {
  if (_scrollPending) return;
  _scrollPending = true;
  requestAnimationFrame(() => {
    _scrollPending = false;
    const d = getDom();
    d.messages.scrollTop = d.messages.scrollHeight;
  });
}

function cancelTypewriter() {
  if (_typewriterTimer) {
    clearInterval(_typewriterTimer);
    _typewriterTimer = null;
  }
  if (_typewriterActiveStep) {
    const body = _typewriterActiveStep.querySelector('.thinking-step-body');
    const full = _typewriterActiveStep._fullText || '';
    if (body && full) {
      body.textContent = full;
    }
    _typewriterActiveStep = null;
  }
}

function typewriteText(stepDiv, text, speed) {
  if (_typewriterActiveStep === stepDiv && _typewriterTimer) {
    stepDiv._fullText = text;
    return;
  }

  cancelTypewriter();
  _typewriterActiveStep = stepDiv;
  stepDiv._fullText = text;

  const body = stepDiv.querySelector('.thinking-step-body');
  if (!body) return;

  let i = body.textContent.length;
  let tick = 0;

  _typewriterTimer = setInterval(() => {
    const currentFull = stepDiv._fullText || text;
    if (i >= currentFull.length) {
      clearInterval(_typewriterTimer);
      _typewriterTimer = null;
      _typewriterActiveStep = null;
      body.textContent = currentFull;
      scheduleScroll();
      return;
    }
    i++;
    tick++;
    body.textContent = currentFull.slice(0, i);
    if (tick % 3 === 0) scheduleScroll();
  }, speed);
}

// ---- SSE 流式事件处理 ----
function handleStreamEvent(eventType, data, assistantMsg, thinkingContainer) {
  const bubble = assistantMsg.querySelector('.message-bubble');

  switch (eventType) {
    case 'thinking':
      addThinkingStep(thinkingContainer, data.step, data.content);
      break;

    case 'token':
      // Collapse thinking when answer starts, but let typewriter finish naturally
      if (!thinkingContainer.classList.contains('thinking-collapsed')) {
        collapseThinking(thinkingContainer);
      }
      appendToken(bubble, data.content);
      break;

    case 'done':
      cancelTypewriter();
      // Let answer typewriter finish naturally, then add metadata
      finalizeMarkdown(bubble, () => {
        addMessageMeta(assistantMsg, data);
        if (data.conversation_id) {
          if (!state.conversationId || state.conversationId !== data.conversation_id) {
            state.conversationId = data.conversation_id;
            updateConvIdDisplay();
            loadConversations();
          }
        }
      });
      break;

    case 'error':
      cancelTypewriter();
      stopAnswerTimer();
      bubble.innerHTML = renderMarkdown('**错误**: ' + escapeHtml(data.message || '未知错误'));
      break;
  }
}

// ---- 发送消息 ----
async function sendMessage() {
  const d = getDom();
  const query = d.input.value.trim();
  if (!query || state.isProcessing) return;

  setProcessing(true);
  state._lastQuery = query;

  const welcome = d.messages.querySelector('.welcome-message');
  if (welcome) welcome.remove();

  addMessage('user', query);
  d.input.value = '';
  d.input.style.height = 'auto';

  const assistantMsg = document.createElement('div');
  assistantMsg.className = 'message assistant';
  assistantMsg.innerHTML = `
    <div class="message-avatar">⚖️</div>
    <div class="message-content">
      <div class="message-bubble"><span class="streaming-cursor"></span></div>
    </div>`;
  d.messages.appendChild(assistantMsg);

  const thinkingContainer = createThinkingContainer();
  const msgContent = assistantMsg.querySelector('.message-content');
  msgContent.insertBefore(thinkingContainer, msgContent.firstChild);

  const persona = getPersonaConfig();
  state.abortController = new AbortController();

  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        persona,
        reasoning_mode: state.currentMode,
        conversation_id: state.conversationId,
      }),
      signal: state.abortController.signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent = null;
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('event: ')) {
          currentEvent = trimmed.slice(7);
        } else if (trimmed.startsWith('data: ') && currentEvent) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            handleStreamEvent(currentEvent, data, assistantMsg, thinkingContainer);
          } catch (e) {
            console.error('SSE parse error:', e);
          }
          currentEvent = null;
        }
      }
    }
  } catch (error) {
    cancelTypewriter();
    stopAnswerTimer();
    if (error.name === 'AbortError') {
      assistantMsg.remove();
    } else {
      const bubble = assistantMsg.querySelector('.message-bubble');
      bubble.innerHTML = renderMarkdown('抱歉，处理您的请求时出现了错误：' + escapeHtml(error.message));
      removeStreamingCursor(bubble);
    }
  } finally {
    cancelTypewriter();
    setProcessing(false);
    state.abortController = null;
  }
}

// ---- 思考过程容器 ----
function createThinkingContainer() {
  const container = document.createElement('div');
  container.className = 'thinking-container thinking-expanded';
  container.style.display = 'none';
  container.innerHTML = `
    <div class="thinking-header">
      <span class="thinking-header-text">🧠 思考过程</span>
      <button class="thinking-toggle">收起</button>
    </div>
    <div class="thinking-steps"></div>`;

  container.querySelector('.thinking-toggle').addEventListener('click', () => {
    toggleThinking(container);
  });

  container.querySelector('.thinking-header').addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') return;
    toggleThinking(container);
  });

  return container;
}

function addThinkingStep(container, stepLabel, content) {
  container.style.display = 'block';
  const steps = container.querySelector('.thinking-steps');

  // Check if last step has same label → append to it
  const lastStep = steps.lastElementChild;
  if (lastStep && lastStep._stepLabel === stepLabel) {
    lastStep._fullText = (lastStep._fullText || '') + content;
    // Typewriter will pick up the new content
    if (_typewriterActiveStep === lastStep) {
      // Already typing this step, new chars will be picked up
    } else {
      // Restart typewriter for this step
      const isResult = lastStep._stepLabel && (lastStep._stepLabel.includes('检索结果') || lastStep._stepLabel.includes('工具结果'));
      typewriteText(lastStep, lastStep._fullText, isResult ? 5 : 30);
    }
    return;
  }

  const stepDiv = document.createElement('div');
  stepDiv.className = `thinking-step ${getThinkingStepClass(stepLabel)}`;
  stepDiv._stepLabel = stepLabel;
  stepDiv.innerHTML = `
    <div class="thinking-step-header">${escapeHtml(stepLabel)}</div>
    <div class="thinking-step-body"></div>`;
  steps.appendChild(stepDiv);

  // Smooth typing: reasoning 30ms/char (continuous flow), results 5ms/char (fast)
  const isResult = stepLabel.includes('检索结果') || stepLabel.includes('工具结果');
  typewriteText(stepDiv, content, isResult ? 5 : 30);

  const d = getDom();
  d.messages.scrollTop = d.messages.scrollHeight;
}

function getThinkingStepClass(label) {
  if (label.includes('开始分析') || label.includes('启动')) return 'thinking-init';
  if (label.includes('推理')) return 'thinking-reasoning';
  if (label.includes('工具结果') || label.includes('检索结果')) return 'thinking-observation';
  if (label.includes('Thought') || label.includes('思考')) return 'thinking-thought';
  if (label.includes('Action') || label.includes('工具')) return 'thinking-action';
  if (label.includes('Observation') || label.includes('观察')) return 'thinking-observation';
  if (label.includes('计划') || label.includes('Plan')) return 'thinking-plan';
  if (label.includes('反思') || label.includes('反馈')) return 'thinking-reflect';
  if (label.includes('执行')) return 'thinking-execute';
  if (label.includes('改进')) return 'thinking-refine';
  if (label.includes('回答')) return 'thinking-answer';
  if (label.includes('审查')) return 'thinking-review';
  return '';
}

function toggleThinking(container) {
  if (container.classList.contains('thinking-collapsed')) {
    container.classList.remove('thinking-collapsed');
    container.classList.add('thinking-expanded');
    container.querySelector('.thinking-toggle').textContent = '收起';
    container.querySelector('.thinking-steps').style.display = '';
  } else {
    collapseThinking(container);
  }
}

function collapseThinking(container) {
  container.classList.remove('thinking-expanded');
  container.classList.add('thinking-collapsed');
  container.querySelector('.thinking-toggle').textContent = '展开';
  container.querySelector('.thinking-steps').style.display = 'none';
}

// ---- Token 追加和渲染 (逐字打字机) ----
let _answerTimer = null;
let _answerBubble = null;
let _answerPos = 0;
let _answerDone = false;
let _onAnswerComplete = null;

function _finishAnswerTypewriter() {
  clearInterval(_answerTimer);
  _answerTimer = null;
  const full = _answerBubble.dataset.rawText || '';
  _answerBubble.innerHTML = renderMarkdown(full);
  delete _answerBubble.dataset.rawText;
  const cursor = _answerBubble.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
  _answerBubble = null;
  _answerPos = 0;
  _answerDone = false;
  const cb = _onAnswerComplete;
  _onAnswerComplete = null;
  if (cb) cb();
}

function appendToken(bubble, token) {
  let rawText = bubble.dataset.rawText || '';
  rawText += token;
  bubble.dataset.rawText = rawText;

  _answerBubble = bubble;

  if (!_answerTimer) {
    _answerPos = 0;
    _answerTimer = setInterval(() => {
      const full = _answerBubble.dataset.rawText || '';
      if (_answerPos >= full.length) {
        if (_answerDone) _finishAnswerTypewriter();
        return;
      }
      // Speed through remaining chars once streaming is done (5 chars/tick)
      const step = _answerDone ? 5 : 1;
      _answerPos = Math.min(_answerPos + step, full.length);
      _answerBubble.innerHTML = renderMarkdown(full.slice(0, _answerPos)) + '<span class="streaming-cursor"></span>';
      scheduleScroll();
    }, 10);
  }
}

function stopAnswerTimer() {
  if (_answerTimer) {
    clearInterval(_answerTimer);
    _answerTimer = null;
  }
  _answerBubble = null;
  _answerPos = 0;
  _answerDone = false;
  _onAnswerComplete = null;
}

function removeStreamingCursor(bubble) {
  const cursor = bubble.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
}

function finalizeMarkdown(bubble, onComplete) {
  // Let typewriter finish naturally then call onComplete
  _answerDone = true;
  if (onComplete) _onAnswerComplete = onComplete;
  // If timer isn't running (no tokens arrived), immediately finalize
  if (!_answerTimer) {
    _answerBubble = bubble;
    _finishAnswerTypewriter();
  }
}

// ---- 消息元数据 ----
function addMessageMeta(msgDiv, data) {
  const confidence = data.confidence ?? 0.9;
  const confClass = confidence >= 0.85 ? 'confidence-high' :
                    confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';
  const modeLabels = { react: 'ReAct', plan_solve: 'Plan-Solve', reflection: 'Reflection' };

  let metaHtml = `<div class="message-meta">
    <span class="mode-badge">${modeLabels[data.reasoning_mode] || data.reasoning_mode}</span>
    <span class="confidence-badge ${confClass}">置信度 ${(confidence * 100).toFixed(0)}%</span>`;

  if (data.citations && data.citations.length > 0) {
    metaHtml += `
      <button class="citation-link citation-btn" data-citations='${JSON.stringify(data.citations).replace(/'/g, "&#39;")}'>
        📚 ${data.citations.length}条引用
      </button>`;
  }
  metaHtml += '</div>';

  const content = msgDiv.querySelector('.message-content');
  content.insertAdjacentHTML('beforeend', metaHtml);
}

// ---- 消息渲染 ----
function addMessage(role, content, metadata) {
  const d = getDom();
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;
  const avatar = role === 'user' ? '👤' : '⚖️';
  const bubbleContent = role === 'assistant' ? renderMarkdown(content) : escapeHtml(content);

  msgDiv.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-bubble">${bubbleContent}</div>
    </div>`;

  d.messages.appendChild(msgDiv);

  // Render metadata (citations, confidence, reasoning mode) for assistant messages
  if (role === 'assistant' && metadata) {
    addMessageMeta(msgDiv, metadata);
  }

  d.messages.scrollTop = d.messages.scrollHeight;
}

// ---- Markdown 渲染 ----
function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked === 'undefined') return escapeHtml(text);

  marked.setOptions({ breaks: true, gfm: true });
  let html = marked.parse(text);

  html = html.replace(/([^<]*)(《[^》]+》)/g, '$1<strong style="color:var(--accent);">$2</strong>');
  html = html.replace(/第[一二三四五六七八九十百千\d]+条/g, '<strong>$&</strong>');

  return html;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML.replace(/\n/g, '<br>');
}

// ---- 建议问题 ----
function setupSuggestions() {
  document.querySelectorAll('.suggestion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (state.isProcessing) return;
      const d = getDom();
      d.input.value = btn.dataset.query;
      sendMessage();
    });
  });
}

// ---- 性格配置 ----
function getPersonaConfig() {
  return {
    role: $('#persona-role')?.value || '法律顾问',
    formality: $('#persona-formality')?.value || '正式严谨',
    verbosity: $('#persona-verbosity')?.value || '标准篇幅',
    tone: $('#persona-tone')?.value || '冷静中立',
    language: '中文',
    address_user: $('#persona-address')?.value || '您',
    conclusion_first: $('#persona-conclusion')?.checked ?? true,
  };
}

// ---- 清空对话 ----
function setupClearChat() {
  const d = getDom();
  d.clearChatBtn.addEventListener('click', () => {
    if (state.isProcessing) {
      if (state.abortController) state.abortController.abort();
      setProcessing(false);
    }
    state.conversationId = null;
    state.messages = [];
    updateConvIdDisplay();
    resetChatUI();
  });
}

function resetChatUI() {
  const d = getDom();
  d.input.value = '';
  d.input.style.height = 'auto';
  d.messages.innerHTML = `
    <div class="welcome-message">
      <div class="welcome-icon">⚖️</div>
      <h2>欢迎使用法律RAG多智能体系统</h2>
      <p>基于DeepSeek的法律智能助手，支持三种推理范式</p>
      <div class="welcome-suggestions">
        <button class="suggestion-btn" data-query="劳动仲裁需要什么材料？">劳动仲裁需要什么材料？</button>
        <button class="suggestion-btn" data-query="民法典关于合同违约的规定是什么？">民法典关于合同违约的规定是什么？</button>
        <button class="suggestion-btn" data-query="公司拖欠工资，如何申请劳动仲裁？">公司拖欠工资，如何申请劳动仲裁？</button>
        <button class="suggestion-btn" data-query="诉讼时效是多长时间？">诉讼时效是多长时间？</button>
      </div>
    </div>`;
  setupSuggestions();
}

// ---- 文件上传弹窗 ----
function setupUploadModal() {
  const d = getDom();
  d.uploadBtn.addEventListener('click', () => {
    d.uploadModal.style.display = 'flex';
  });

  const modalClose = d.uploadModal.querySelector('.modal-close');
  if (modalClose) {
    modalClose.addEventListener('click', () => { d.uploadModal.style.display = 'none'; });
  }

  d.uploadModal.addEventListener('click', (e) => {
    if (e.target === d.uploadModal) d.uploadModal.style.display = 'none';
  });

  d.uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    d.uploadDropzone.classList.add('dragover');
  });

  d.uploadDropzone.addEventListener('dragleave', () => {
    d.uploadDropzone.classList.remove('dragover');
  });

  d.uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    d.uploadDropzone.classList.remove('dragover');
    handleUploadFiles(Array.from(e.dataTransfer.files));
  });

  d.uploadDropzone.addEventListener('click', () => { d.uploadFileInput.click(); });

  d.uploadFileInput.addEventListener('change', (e) => {
    handleUploadFiles(Array.from(e.target.files));
  });

  d.uploadConfirm.addEventListener('click', uploadFilesToKB);

  // Library target selector tabs
  document.querySelectorAll('.kb-target-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.kb-target-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      d.uploadDocType.value = btn.dataset.type;
    });
  });
}

function handleUploadFiles(files) {
  const d = getDom();
  if (files.length > 7) {
    alert('一次最多上传 7 个文件');
    return;
  }
  state.uploadFiles = files;
  d.uploadPreview.innerHTML = files.map((f, i) => `
    <span class="file-preview-item">
      📄 ${f.name} (${formatFileSize(f.size)})
      <span class="remove-file" data-index="${i}">&times;</span>
    </span>
  `).join('');

  d.uploadPreview.querySelectorAll('.remove-file').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(btn.dataset.index);
      state.uploadFiles.splice(idx, 1);
      handleUploadFiles(state.uploadFiles);
    });
  });
}

function uploadFileWithProgress(file, docType, onProgress) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_type', docType);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/knowledge/upload`);

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        onProgress({ loaded: e.loaded, total: e.total, percent: Math.round((e.loaded / e.total) * 100) });
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          resolve({ status: 'ok' });
        }
      } else {
        let msg = `HTTP ${xhr.status}`;
        try {
          const err = JSON.parse(xhr.responseText);
          msg = err.detail || msg;
        } catch {}
        reject(new Error(msg));
      }
    });

    xhr.addEventListener('error', () => reject(new Error('网络错误')));
    xhr.addEventListener('abort', () => reject(new Error('上传取消')));

    xhr.send(formData);
  });
}

// Smooth progress: animates display value towards real value using RAF
function createSmoothProgress(progressBar, onDisplay) {
  let realValue = 0;
  let displayValue = 0;
  let rafId = null;

  function animate() {
    // Ease towards real value (catch-up speed proportional to gap)
    const gap = realValue - displayValue;
    if (Math.abs(gap) < 0.3) {
      displayValue = realValue;
      progressBar.value = displayValue;
      if (onDisplay) onDisplay(displayValue);
      rafId = null;
      return;
    }
    // Move 15% of the gap per frame (~smooth exponential catch-up)
    displayValue += gap * 0.15;
    progressBar.value = Math.round(displayValue);
    if (onDisplay) onDisplay(displayValue);
    rafId = requestAnimationFrame(animate);
  }

  return {
    setTarget(value) {
      realValue = value;
      if (!rafId) {
        rafId = requestAnimationFrame(animate);
      }
    },
    finish(value) {
      realValue = value;
      displayValue = value;
      progressBar.value = value;
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      if (onDisplay) onDisplay(value);
    },
    stop() {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    },
  };
}

async function uploadFilesToKB() {
  const d = getDom();
  if (state.uploadFiles.length === 0) {
    alert('请先选择文件');
    return;
  }

  const docType = d.uploadDocType.value;
  const progressBar = d.uploadProgress;
  const progressLabel = document.getElementById('upload-progress-label');
  progressBar.style.display = 'block';
  progressBar.value = 0;

  const smooth = createSmoothProgress(progressBar, (displayVal) => {
    if (progressLabel) {
      progressLabel.textContent = `上传中... ${Math.min(100, Math.round(displayVal))}%`;
    }
  });

  if (progressLabel) progressLabel.textContent = '准备上传...';

  let success = 0;
  let globalLoaded = 0;
  const totalSize = state.uploadFiles.reduce((s, f) => s + f.size, 0);

  for (let i = 0; i < state.uploadFiles.length; i++) {
    const file = state.uploadFiles[i];
    try {
      if (progressLabel) progressLabel.textContent = `正在上传: ${file.name}`;

      await uploadFileWithProgress(file, docType, (p) => {
        const rawPercent = ((globalLoaded + p.loaded) / totalSize) * 100;
        const realPercent = Math.min(99, Math.round(rawPercent));
        smooth.setTarget(realPercent);
      });

      globalLoaded += file.size;
      success++;
    } catch (err) {
      console.error(`上传 ${file.name} 失败:`, err);
      alert(`上传 ${file.name} 失败: ${err.message}`);
    }
  }

  smooth.finish(100);
  if (progressLabel) progressLabel.textContent = '上传完成！';
  setTimeout(() => {
    progressBar.style.display = 'none';
    progressBar.value = 0;
    if (progressLabel) progressLabel.textContent = '';
  }, 1500);

  alert(`上传完成: ${success}/${state.uploadFiles.length} 个文件成功`);
  state.uploadFiles = [];
  d.uploadPreview.innerHTML = '';
  d.uploadModal.style.display = 'none';
  refreshKBStats();
}

function showUploadPreviewForChat(files) {
  const d = getDom();
  const names = files.map(f => f.name).join(', ');
  d.input.value = `[已附加文件: ${names}]\n${d.input.value}`;
}

// ---- 知识库管理 ----
function setupKBManagement() {
  const refreshBtn = $('#kb-refresh-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => { refreshKBStats(); });
  }
}

async function refreshKBStats() {
  try {
    const [statsRes, listRes] = await Promise.all([
      fetch(`${API_BASE}/knowledge/stats`),
      fetch(`${API_BASE}/knowledge/list`),
    ]);
    const stats = await statsRes.json();
    const list = await listRes.json();

    const el1 = $('#stat-statutes');
    const el2 = $('#stat-cases');
    const el3 = $('#stat-practical');
    if (el1) el1.textContent = stats.statutes ?? 0;
    if (el2) el2.textContent = stats.cases ?? 0;
    if (el3) el3.textContent = stats.practical ?? 0;

    renderKBDocList(list.documents || []);
  } catch (err) {
    console.error('刷新知识库失败:', err);
  }
}

function renderKBDocList(documents) {
  const listEl = $('#kb-doc-list');
  if (!listEl) return;

  if (!documents || documents.length === 0) {
    listEl.innerHTML = '<div class="kb-empty">暂无文档，请上传文件</div>';
    return;
  }

  // Sort: by doc_type order, then by created_at desc (newest first)
  const typeOrder = { statutes: 0, cases: 1, practical: 2 };
  const sorted = [...documents].sort((a, b) => {
    const typeDiff = (typeOrder[a.doc_type] ?? 9) - (typeOrder[b.doc_type] ?? 9);
    if (typeDiff !== 0) return typeDiff;
    const ta = a.metadata?.created_at || 0;
    const tb = b.metadata?.created_at || 0;
    return tb - ta; // newest first
  });

  // Group by doc_type for section headers
  const typeLabels = { statutes: '法规库', cases: '案例库', practical: '实务库' };
  const typeIcons = { statutes: '📜', cases: '⚖️', practical: '📋' };

  let html = '';
  let currentType = null;
  for (const doc of sorted) {
    if (doc.doc_type !== currentType) {
      currentType = doc.doc_type;
      html += `<div class="kb-section-header">${typeIcons[currentType] || '📄'} ${typeLabels[currentType] || currentType}</div>`;
    }
    html += `
      <div class="kb-doc-item">
        <span class="kb-doc-icon">${typeIcons[doc.doc_type] || '📄'}</span>
        <div class="kb-doc-info">
          <span class="kb-doc-title" title="${escapeHtml(doc.title)}">${escapeHtml(doc.title)}</span>
        </div>
        <button class="kb-doc-delete" data-id="${doc.id}" data-type="${doc.doc_type}" title="删除">✕</button>
      </div>`;
  }
  listEl.innerHTML = html;

  // Bind delete buttons
  listEl.querySelectorAll('.kb-doc-delete').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const docId = btn.dataset.id;
      const docType = btn.dataset.type;
      try {
        await fetch(`${API_BASE}/knowledge/${docType}/${docId}`, { method: 'DELETE' });
        refreshKBStats();
      } catch (err) {
        console.error('删除失败:', err);
      }
    });
  });
}

// ---- 引用展示 ----
function setupCitationPanelClose() {
  const panel = $('#citation-panel');
  if (!panel) return;
  const closeBtn = panel.querySelector('.citation-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => { panel.style.display = 'none'; });
  }
}

function showCitations(citations) {
  const panel = $('#citation-panel');
  const content = $('#citation-content');
  if (!panel || !content) return;
  panel.style.display = 'flex';
  content.innerHTML = citations.map((c, i) => `
    <div class="citation-item" style="margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid var(--border);">
      <strong>${escapeHtml(c.source_type)} #${i + 1}</strong>
      <h4 style="margin:4px 0;">${escapeHtml(c.title)}</h4>
      <p style="color:var(--text-light);font-size:12px;">相关度: ${((c.relevance_score || 1) * 100).toFixed(0)}%</p>
      ${c.content ? `<p style="margin-top:8px;">${escapeHtml(c.content)}</p>` : ''}
    </div>
  `).join('');
}

// ---- 会话管理 ----
async function loadConversations() {
  try {
    const res = await fetch(`${API_BASE}/conversations`);
    const data = await res.json();
    renderConversationList(data);
  } catch (err) {
    console.error('加载会话列表失败:', err);
  }
}

function renderConversationList(conversations) {
  const listEl = $('#conversation-list');
  if (!listEl) return;

  listEl.innerHTML = '';
  if (conversations.length === 0) {
    listEl.innerHTML = '<div class="conv-empty">暂无会话记录</div>';
    return;
  }

  conversations.forEach(conv => {
    const item = document.createElement('div');
    item.className = 'conv-item';
    item.dataset.convId = conv.id;
    item.title = conv.title;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'conv-item-title';
    titleSpan.textContent = conv.title.length > 18 ? conv.title.slice(0, 18) + '...' : conv.title;

    const delBtn = document.createElement('button');
    delBtn.className = 'conv-item-delete';
    delBtn.title = '删除会话';
    delBtn.textContent = '✕';
    delBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('确定删除此会话？')) return;
      try {
        await fetch(`${API_BASE}/conversations/${conv.id}`, { method: 'DELETE' });
        if (state.conversationId === conv.id) {
          state.conversationId = null;
          updateConvIdDisplay();
          resetChatUI();
        }
        loadConversations();
      } catch (err) {
        console.error('删除会话失败:', err);
      }
    });

    item.appendChild(titleSpan);
    item.appendChild(delBtn);

    if (conv.id === state.conversationId) {
      item.classList.add('active');
    }

    item.addEventListener('click', (e) => {
      if (e.target === delBtn) return;
      switchConversation(conv.id);
    });

    listEl.appendChild(item);
  });
}

async function switchConversation(convId) {
  if (state.isProcessing) {
    if (state.abortController) state.abortController.abort();
    setProcessing(false);
  }

  state.conversationId = convId;
  updateConvIdDisplay();

  // Highlight active
  const listEl = $('#conversation-list');
  if (listEl) {
    listEl.querySelectorAll('.conv-item').forEach(el => {
      el.classList.toggle('active', el.dataset.convId === convId);
    });
  }

  // Show loading state
  const d = getDom();
  d.messages.innerHTML = `
    <div class="welcome-message">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
      <p style="margin-top:12px;">加载会话...</p>
    </div>`;

  // Load and display messages
  try {
    const res = await fetch(`${API_BASE}/conversations/${convId}`);
    if (!res.ok) throw new Error('加载失败');
    const conv = await res.json();

    d.messages.innerHTML = '';

    if (!conv.messages || conv.messages.length === 0) {
      d.messages.innerHTML = `
        <div class="welcome-message">
          <div class="welcome-icon">⚖️</div>
          <h2>会话 "${escapeHtml(conv.title)}"</h2>
          <p>此会话暂无消息</p>
          <div class="welcome-suggestions">
            <button class="suggestion-btn" data-query="劳动仲裁需要什么材料？">劳动仲裁需要什么材料？</button>
            <button class="suggestion-btn" data-query="民法典关于合同违约的规定是什么？">民法典关于合同违约的规定是什么？</button>
          </div>
        </div>`;
      setupSuggestions();
    } else {
      conv.messages.forEach(msg => {
        addMessage(msg.role, msg.content, msg.metadata);
      });
    }
    d.messages.scrollTop = d.messages.scrollHeight;
  } catch (err) {
    console.error('加载会话失败:', err);
    d.messages.innerHTML = `
      <div class="welcome-message">
        <div class="welcome-icon">⚠️</div>
        <h2>加载失败</h2>
        <p>无法加载会话记录，请重试</p>
      </div>`;
  }
}

function setupNewConversation() {
  const d = getDom();
  d.newConvBtn.addEventListener('click', () => {
    if (state.isProcessing) {
      if (state.abortController) state.abortController.abort();
      setProcessing(false);
    }

    state.conversationId = null;
    state.messages = [];
    state.uploadFiles = [];
    updateConvIdDisplay();

    // Clear active
    const listEl = $('#conversation-list');
    if (listEl) {
      listEl.querySelectorAll('.conv-item').forEach(el => el.classList.remove('active'));
    }

    resetChatUI();
    d.messages.scrollTop = 0;
  });
}

// ---- 工具函数 ----
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ---- 工具开关管理 ----
async function setupTools() {
  try {
    const res = await fetch(`${API_BASE}/tools`);
    if (!res.ok) return;
    const data = await res.json();
    renderToolToggles(data.tools || []);
  } catch (err) {
    console.error('加载工具列表失败:', err);
  }
}

function renderToolToggles(tools) {
  const listEl = $('#tools-toggle-list');
  const countEl = $('#tools-count');
  if (!listEl) return;

  if (!tools || tools.length === 0) {
    listEl.innerHTML = '<div class="tools-empty">暂无可用工具</div>';
    if (countEl) countEl.textContent = '';
    return;
  }

  const locals = tools.filter(t => t.source !== 'mcp');
  const mcps = tools.filter(t => t.source === 'mcp');
  const enabledMcp = mcps.filter(t => t.enabled).length;
  // Count only MCP tools since locals are always on
  if (countEl) countEl.textContent = mcps.length > 0 ? `${enabledMcp}/${mcps.length}` : '';

  listEl.innerHTML = [
    // Local tools — always on, no toggle
    ...locals.map(tool => {
      const sourceLabel = '内置';
      const sourceClass = 'local';
      return `
        <div class="tool-toggle-item tool-local">
          <span class="toggle-locked" title="内置工具，始终启用">🔒</span>
          <div class="tool-toggle-info">
            <span class="tool-toggle-name" title="${escapeHtml(tool.name)}">${escapeHtml(tool.name)}</span>
            <span class="tool-toggle-desc" title="${escapeHtml(tool.description)}">${escapeHtml(tool.description)}</span>
          </div>
          <span class="tool-toggle-source ${sourceClass}">${sourceLabel}</span>
        </div>`;
    }),
    // Separator if both groups present
    ...(locals.length > 0 && mcps.length > 0
      ? ['<div class="tools-divider">MCP 远程工具</div>']
      : []),
    // MCP tools — toggleable
    ...mcps.map(tool => {
      const sourceLabel = 'MCP';
      const sourceClass = 'mcp';
      return `
        <div class="tool-toggle-item${tool.enabled ? '' : ' disabled'}">
          <label class="toggle-switch" title="${tool.enabled ? '禁用' : '启用'} ${tool.name}">
            <input type="checkbox" ${tool.enabled ? 'checked' : ''}
                   data-tool="${escapeHtml(tool.name)}"
                   onchange="toggleTool(this)">
            <span class="toggle-slider"></span>
          </label>
          <div class="tool-toggle-info">
            <span class="tool-toggle-name" title="${escapeHtml(tool.name)}">${escapeHtml(tool.name)}</span>
            <span class="tool-toggle-desc" title="${escapeHtml(tool.description)}">${escapeHtml(tool.description)}</span>
          </div>
          <span class="tool-toggle-source ${sourceClass}">${sourceLabel}</span>
        </div>`;
    }),
  ].join('');
}

async function toggleTool(checkbox) {
  const name = checkbox.dataset.tool;
  const enabled = checkbox.checked;

  // Optimistic UI: update item appearance immediately
  const item = checkbox.closest('.tool-toggle-item');
  if (item) {
    if (enabled) { item.classList.remove('disabled'); }
    else { item.classList.add('disabled'); }
  }

  try {
    const res = await fetch(`${API_BASE}/tools/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, enabled }),
    });
    if (!res.ok) throw new Error('Toggle failed');
    // Update count (MCP tools only)
    const countEl = $('#tools-count');
    if (countEl) {
      const mcps = document.querySelectorAll('#tools-toggle-list .toggle-switch input[type=checkbox]');
      const checked = document.querySelectorAll('#tools-toggle-list .toggle-switch input[type=checkbox]:checked');
      if (mcps.length > 0) countEl.textContent = `${checked.length}/${mcps.length}`;
    }
  } catch (err) {
    // Revert on failure
    checkbox.checked = !enabled;
    if (item) {
      if (enabled) { item.classList.add('disabled'); }
      else { item.classList.remove('disabled'); }
    }
    console.error('工具切换失败:', err);
  }
}

// ---- 启动 ----
document.addEventListener('DOMContentLoaded', init);
