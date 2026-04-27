document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chat-messages");
    const composer = document.getElementById("composer");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const clearBtn = document.getElementById("clear-btn");
    const refreshBtn = document.getElementById("refresh-btn");
    const promptChips = Array.from(document.querySelectorAll(".prompt-chip"));

    (function initParticles() {
        const canvas = document.getElementById("canvas-bg");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener("resize", resize);
        const pts = Array.from({ length: 50 }, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 2 + 1,
            a: Math.random() * 0.35 + 0.05,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
        }));
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            pts.forEach((p) => {
                p.x = (p.x + p.vx + canvas.width) % canvas.width;
                p.y = (p.y + p.vy + canvas.height) % canvas.height;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(99,102,241,${p.a})`;
                ctx.fill();
            });
            requestAnimationFrame(draw);
        }
        draw();
    })();

    const esc = (t) =>
        String(t || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    const fmt = (t) =>
        esc(t)
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/`([^`]+)`/g, "<code>$1</code>");

    const el = (tag, cls, html) => {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (html !== undefined) e.innerHTML = html;
        return e;
    };

    const scrollBot = () =>
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });

    function makeRow(sender) {
        const row = el("div", `message-row ${sender}`);
        const card = el("article", `message-card ${sender}`);
        row.appendChild(card);
        return { row, card };
    }

    function renderBot(card, data) {
        const type = data.type || "generic";
        const title = data.title || "Response";
        const summary = data.summary || "";
        const urgency = data.urgency || "normal";
        const sections = Array.isArray(data.sections) ? data.sections : [];
        const alternatives = Array.isArray(data.alternatives) ? data.alternatives : [];
        const symptoms = Array.isArray(data.matched_symptoms) ? data.matched_symptoms : [];
        const confidence =
            typeof data.confidence === "number"
                ? Math.min(100, Math.round(data.confidence * 1000) / 10)
                : null;
        const matchedQ = data.matched_question || "";

        card.classList.add(`type-${type}`);

        const kickerMap = {
            greeting: "👋 Welcome",
            disease: "🔬 Symptom Analysis",
            answer: "📚 Medical Information",
            general: "💬 MedBot",
            unknown: "🩺 MedBot",
            error: "⚠️ Notice",
        };
        card.appendChild(el("p", "msg-kicker", kickerMap[type] || "🩺 Medical Assistant"));

        const hRow = el("div", "card-header-row");
        hRow.appendChild(el("h3", "card-title", esc(title)));
        const showBadge = !["greeting", "general", "unknown"].includes(type);
        if (showBadge) {
            const badgeMap = { high: "🚨 Urgent", medium: "⚠️ Watch Closely", low: "Low", normal: "Routine" };
            hRow.appendChild(el("span", `status-badge ${urgency}`, badgeMap[urgency] || "Routine"));
        }
        card.appendChild(hRow);

        if (summary) card.appendChild(el("p", "card-summary", fmt(summary)));

        if (confidence !== null && confidence > 0) {
            const wrap = el("div", "confidence-wrap");
            wrap.appendChild(el("p", "confidence-label", `ML Confidence: ${confidence}%`));
            const track = el("div", "confidence-track");
            const fill = el("span", "confidence-fill");
            fill.style.width = `${Math.max(6, confidence)}%`;
            track.appendChild(fill);
            wrap.appendChild(track);
            card.appendChild(wrap);
        }

        if (matchedQ) {
            card.appendChild(el("p", "matched-q", `<span class="mq-label">Best match:</span> ${esc(matchedQ)}`));
        }

        sections.forEach((sec) => {
            if (!sec?.points?.length) return;
            const secEl = el("div", "resp-section");
            if (sec.heading) secEl.appendChild(el("h4", "sec-heading", esc(sec.heading)));
            const ul = el("ul", "sec-points");
            sec.points.forEach((pt) => {
                if (pt) ul.appendChild(el("li", "", fmt(String(pt))));
            });
            secEl.appendChild(ul);
            card.appendChild(secEl);
        });

        if (symptoms.length && type === "disease") {
            const row = el("div", "tag-row");
            symptoms.forEach((s) => row.appendChild(el("span", "sym-tag", esc(s))));
            card.appendChild(row);
        }

        if (alternatives.length) {
            const alt = el("div", "alt-block");
            alt.appendChild(el("p", "alt-label", "Other possible conditions"));
            alternatives.forEach((a) => {
                const row = el("div", "alt-row");
                row.appendChild(el("span", "", esc(a.disease || "Unknown")));
                const pct = Math.min(100, Math.round((a.confidence || 0) * 1000) / 10);
                row.appendChild(el("span", "alt-pct", `${pct}%`));
                alt.appendChild(row);
            });
            card.appendChild(alt);
        }
    }

    function appendMessage(content, sender) {
        const { row, card } = makeRow(sender);
        if (sender === "user") {
            card.textContent = typeof content === "string" ? content : content.summary || "";
        } else if (content && typeof content === "object") {
            renderBot(card, content);
        } else {
            card.appendChild(el("p", "", fmt(String(content))));
        }
        chatMessages.appendChild(row);
        scrollBot();
    }

    function showTyping() {
        const { row, card } = makeRow("bot");
        const ind = el("div", "typing-indicator");
        ind.innerHTML = "<span></span><span></span><span></span>";
        card.appendChild(ind);
        chatMessages.appendChild(row);
        scrollBot();
        return row;
    }

    function autoResize() {
        userInput.style.height = "auto";
        userInput.style.height = `${Math.min(userInput.scrollHeight, 160)}px`;
    }

    function setDisabled(d) {
        userInput.disabled = d;
        sendBtn.disabled = d;
        if (clearBtn) clearBtn.disabled = d;
        if (refreshBtn) refreshBtn.disabled = d;
        promptChips.forEach((c) => {
            c.disabled = d;
        });
    }

    async function sendMessage(override) {
        const msg = (override || userInput.value).trim();
        if (!msg) {
            userInput.classList.add("shake");
            setTimeout(() => userInput.classList.remove("shake"), 400);
            return;
        }

        chatMessages.querySelector(".welcome-screen")?.remove();
        appendMessage(msg, "user");
        userInput.value = "";
        autoResize();

        const typingRow = showTyping();
        setDisabled(true);

        try {
            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: msg }),
            });
            const data = await res.json();
            typingRow.remove();
            appendMessage(data, "bot");
        } catch {
            typingRow.remove();
            appendMessage(
                {
                    type: "error",
                    title: "Connection Error",
                    summary: "Could not reach the server. Please check your connection and try again.",
                },
                "bot"
            );
        } finally {
            setDisabled(false);
            userInput.focus();
        }
    }

    function resetChat() {
        chatMessages.innerHTML = `
            <div class="welcome-screen">
                <div class="welcome-icon"><i class="fas fa-stethoscope"></i></div>
                <h3>Welcome to MediMind AI</h3>
                <p>Ask a medical question or describe symptoms.</p>
                <div class="welcome-tips">
                    <div class="tip"><i class="fas fa-question-circle"></i> "What causes Sudden Cardiac Arrest ?"</div>
                    <div class="tip"><i class="fas fa-question-circle"></i> "What causes Urinary Tract Infections ?"</div>
                    <div class="tip"><i class="fas fa-question-circle"></i> "How to prevent Glaucoma ?"</div>
                    <div class="tip"><i class="fas fa-heartbeat"></i> "I have fever, headache and cough for 2 days"</div>
                </div>
            </div>`;
        userInput.value = "";
        autoResize();
        userInput.focus();
    }

    composer.addEventListener("submit", (e) => {
        e.preventDefault();
        sendMessage();
    });
    userInput.addEventListener("input", autoResize);
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    if (clearBtn) clearBtn.addEventListener("click", resetChat);
    if (refreshBtn) refreshBtn.addEventListener("click", resetChat);
    promptChips.forEach((chip) => {
        chip.addEventListener("click", () => sendMessage(chip.dataset.prompt || chip.textContent.trim()));
    });

    autoResize();
    userInput.focus();
});
