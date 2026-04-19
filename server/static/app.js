/**
 * JimiAgent WebChat — 前端交互逻辑
 */
(function () {
  "use strict";

  // === DOM ===
  const $ = (sel) => document.querySelector(sel);
  const chatMessages = $("#chat-messages");
  const messageInput = $("#message-input");
  const btnSend = $("#btn-send");
  const btnNewSession = $("#btn-new-session");
  const btnMenu = $("#btn-menu");
  const sidebar = $("#sidebar");
  const sessionList = $("#session-list");
  const statusDot = $(".status-dot");
  const statusText = $(".status-text");
  const welcomeMessage = $("#welcome-message");
  const modelInfo = $("#model-info");
  const chatTitle = $("#chat-title");
  // 多模态
  const btnAttach = $("#btn-attach");
  const fileInput = $("#file-input");
  const attachmentBar = $("#attachment-bar");
  const inputWrapper = $("#input-wrapper");
  // 待发送图片：[{url, name, pending}]
  let pendingImages = [];

  // === 状态 ===
  let ws = null;
  let currentSessionId = null;
  let isStreaming = false;
  let currentStreamDiv = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT = 5;

  // === WebSocket ===
  function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}/ws/chat`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      reconnectAttempts = 0;
      setStatus("connected", "已连接");
      loadSessions();
      loadStatus();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWSMessage(data);
      } catch (e) {
        console.error("WS parse error:", e);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected", "已断开");
      if (reconnectAttempts < MAX_RECONNECT) {
        reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
        setTimeout(connectWebSocket, delay);
      }
    };

    ws.onerror = () => {
      setStatus("disconnected", "连接错误");
    };
  }

  function handleWSMessage(data) {
    switch (data.type) {
      case "session":
        // /new 或初次连接都会发
        if (currentSessionId !== data.session_id) {
          currentSessionId = data.session_id;
          loadSessions();
        }
        break;

      case "stream":
        if (!currentStreamDiv) {
          currentStreamDiv = appendMessage("assistant", "");
          removeTypingIndicator();
          hideWelcome();
        }
        const contentEl = currentStreamDiv.querySelector(".message-content");
        contentEl.textContent += data.content;
        scrollToBottom();
        break;

      case "tool":
        // 显示工具调用提示
        if (!currentStreamDiv) {
          currentStreamDiv = appendMessage("assistant", "");
          removeTypingIndicator();
          hideWelcome();
        }
        const toolEl = currentStreamDiv.querySelector(".message-content");
        toolEl.textContent += `\n[tool] ${data.name}...\n`;
        scrollToBottom();
        break;

      case "end":
        if (currentStreamDiv) {
          // 渲染 markdown
          const contentEl = currentStreamDiv.querySelector(".message-content");
          contentEl.innerHTML = renderMarkdown(contentEl.textContent);
        }
        currentStreamDiv = null;
        isStreaming = false;
        btnSend.disabled = false;
        removeTypingIndicator();
        loadSessions(); // 刷新会话列表
        break;

      case "error":
        removeTypingIndicator();
        if (currentStreamDiv) {
          const contentEl = currentStreamDiv.querySelector(".message-content");
          contentEl.innerHTML = `<span style="color: var(--error)">${escapeHtml(data.content)}</span>`;
        } else {
          appendMessage("assistant", data.content);
        }
        currentStreamDiv = null;
        isStreaming = false;
        btnSend.disabled = false;
        break;
    }
  }

  // === 上传辅助 ===
  async function uploadFile(file) {
    const fd = new FormData();
    fd.append("file", file, file.name || "image.png");
    const resp = await fetch("/api/upload", { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${err}`);
    }
    return await resp.json();
  }

  function renderAttachments() {
    attachmentBar.innerHTML = "";
    if (pendingImages.length === 0) {
      attachmentBar.hidden = true;
      return;
    }
    attachmentBar.hidden = false;
    pendingImages.forEach((img, idx) => {
      const wrap = document.createElement("div");
      wrap.className = "attachment-item" + (img.pending ? " pending" : "");
      wrap.innerHTML = `
        <img src="${img.url}" alt="${img.name || "image"}">
        ${img.pending ? '<div class="spinner"></div>' : ""}
        <button class="attachment-remove" title="移除">×</button>
      `;
      wrap.querySelector(".attachment-remove").addEventListener("click", () => {
        pendingImages.splice(idx, 1);
        renderAttachments();
      });
      attachmentBar.appendChild(wrap);
    });
  }

  async function addFiles(files) {
    for (const f of files) {
      if (!f.type || !f.type.startsWith("image/")) continue;
      // 先用本地 URL 预览
      const localUrl = URL.createObjectURL(f);
      const item = { url: localUrl, name: f.name, pending: true };
      pendingImages.push(item);
      renderAttachments();
      try {
        const res = await uploadFile(f);
        item.url = res.url;
        item.pending = false;
        renderAttachments();
      } catch (e) {
        console.error("upload failed:", e);
        const idx = pendingImages.indexOf(item);
        if (idx !== -1) pendingImages.splice(idx, 1);
        renderAttachments();
        appendMessage("assistant", `图片上传失败：${e.message}`);
      }
    }
  }

  // === 发送消息 ===
  function sendMessage() {
    const message = messageInput.value.trim();
    // 有图片时允许空文本
    if ((!message && pendingImages.length === 0) || isStreaming) return;
    // 上传中不发送
    if (pendingImages.some((i) => i.pending)) {
      appendMessage("assistant", "图片仍在上传中，请稍候再发送。");
      return;
    }

    // 先显示用户消息
    const imageUrls = pendingImages.map((i) => i.url);
    appendMessage("user", message, imageUrls);
    hideWelcome();

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        message: message,
        session_id: currentSessionId || "",
        images: imageUrls,
      }));

      isStreaming = true;
      btnSend.disabled = true;
      showTypingIndicator();
    } else {
      appendMessage("assistant", "未连接到服务器，请刷新页面重试。");
    }

    messageInput.value = "";
    pendingImages = [];
    renderAttachments();
    autoResizeInput();
    scrollToBottom();
  }

  // === UI 辅助 ===
  function appendMessage(role, content, images) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    const avatar = role === "user" ? "U" : "A";
    const rendered = content ? renderMarkdown(content) : "";
    let imgsHtml = "";
    if (images && images.length) {
      imgsHtml =
        '<div class="message-images">' +
        images
          .map(
            (url) =>
              `<a href="${url}" target="_blank" rel="noopener">` +
              `<img src="${url}" alt="image">` +
              "</a>"
          )
          .join("") +
        "</div>";
    }

    div.innerHTML = `
      <div class="message-avatar">${avatar}</div>
      <div class="message-content">${imgsHtml}${rendered}</div>
    `;

    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
  }

  function showTypingIndicator() {
    const div = document.createElement("div");
    div.className = "message assistant typing-msg";
    div.innerHTML = `
      <div class="message-avatar">A</div>
      <div class="message-content">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const el = chatMessages.querySelector(".typing-msg");
    if (el) el.remove();
  }

  function hideWelcome() {
    if (welcomeMessage) {
      welcomeMessage.style.display = "none";
    }
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  function setStatus(state, text) {
    statusDot.className = `status-dot ${state}`;
    statusText.textContent = text;
  }

  // === 简易 Markdown ===
  function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    // 代码块
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // 行内代码
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // 粗体
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // 工具提示
    html = html.replace(
      /\[tool\] (.+?)\.{3}/g,
      '<div class="tool-indicator">[tool] $1</div>'
    );

    // 换行
    html = html.replace(/\n/g, "<br>");

    return html;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // === Sessions API ===
  async function loadSessions() {
    try {
      const res = await fetch("/api/sessions");
      const sessions = await res.json();
      renderSessionList(sessions);

      if (!currentSessionId && sessions.length > 0) {
        currentSessionId = sessions[0].id;
      }
    } catch (e) {
      console.error("Load sessions error:", e);
    }
  }

  function renderSessionList(sessions) {
    sessionList.innerHTML = "";
    sessions.forEach((s) => {
      const div = document.createElement("div");
      div.className = `session-item${s.id === currentSessionId ? " active" : ""}`;
      div.innerHTML = `
        <span class="session-title">${escapeHtml(s.title)}</span>
        <span class="session-count">${s.message_count}</span>
        <button class="session-delete" title="删除">×</button>
      `;

      div.addEventListener("click", (e) => {
        if (e.target.classList.contains("session-delete")) {
          deleteSession(s.id);
          return;
        }
        switchSession(s.id, s.title);
      });

      sessionList.appendChild(div);
    });
  }

  function switchSession(id, title) {
    currentSessionId = id;
    chatTitle.textContent = title || "JimiAgent";

    // 清空消息区
    chatMessages.innerHTML = "";
    if (welcomeMessage) {
      chatMessages.appendChild(welcomeMessage);
      welcomeMessage.style.display = "flex";
    }

    // 更新活跃态
    document.querySelectorAll(".session-item").forEach((el) => {
      el.classList.remove("active");
    });
    event.currentTarget?.classList.add("active");

    loadSessions();
  }

  async function createSession() {
    try {
      const res = await fetch("/api/sessions/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "新对话" }),
      });
      const session = await res.json();
      currentSessionId = session.id;
      chatTitle.textContent = session.title;

      // 清空消息
      chatMessages.innerHTML = "";
      if (welcomeMessage) {
        chatMessages.appendChild(welcomeMessage);
        welcomeMessage.style.display = "flex";
      }

      loadSessions();
    } catch (e) {
      console.error("Create session error:", e);
    }
  }

  async function deleteSession(id) {
    try {
      await fetch(`/api/sessions/${id}`, { method: "DELETE" });
      if (currentSessionId === id) {
        currentSessionId = null;
      }
      loadSessions();
    } catch (e) {
      console.error("Delete session error:", e);
    }
  }

  async function loadStatus() {
    try {
      const res = await fetch("/api/status");
      const status = await res.json();
      modelInfo.textContent = status.model || "-";
    } catch (e) {
      console.error("Load status error:", e);
    }
  }

  // === 输入框自适应 ===
  function autoResizeInput() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
  }

  // === 事件绑定 ===
  btnSend.addEventListener("click", sendMessage);

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  messageInput.addEventListener("input", autoResizeInput);

  btnNewSession.addEventListener("click", createSession);

  btnMenu.addEventListener("click", () => {
    sidebar.classList.toggle("open");
  });

  // 快捷操作
  document.querySelectorAll(".quick-action").forEach((btn) => {
    btn.addEventListener("click", () => {
      messageInput.value = btn.dataset.message;
      sendMessage();
    });
  });

  // === 图片上传交互 ===
  if (btnAttach && fileInput) {
    btnAttach.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length) {
        addFiles(Array.from(fileInput.files));
        fileInput.value = "";
      }
    });
  }

  // 粘贴图片自动附加
  messageInput.addEventListener("paste", (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    const files = [];
    for (const it of items) {
      if (it.kind === "file") {
        const f = it.getAsFile();
        if (f && f.type.startsWith("image/")) files.push(f);
      }
    }
    if (files.length) {
      e.preventDefault();
      addFiles(files);
    }
  });

  // 拖拽到输入区
  if (inputWrapper) {
    const highlightOn = () => inputWrapper.classList.add("drag-over");
    const highlightOff = () => inputWrapper.classList.remove("drag-over");
    ["dragenter", "dragover"].forEach((ev) => {
      inputWrapper.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        highlightOn();
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      inputWrapper.addEventListener(ev, (e) => {
        e.preventDefault();
        e.stopPropagation();
        highlightOff();
      });
    });
    inputWrapper.addEventListener("drop", (e) => {
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) addFiles(Array.from(files));
    });
  }

  // 阻止浏览器默认打开拖拽文件
  ["dragover", "drop"].forEach((ev) => {
    document.addEventListener(ev, (e) => e.preventDefault());
  });

  // 移动端点击消息区时收起侧栏
  chatMessages.addEventListener("click", () => {
    sidebar.classList.remove("open");
  });

  // === 初始化 ===
  connectWebSocket();
})();
