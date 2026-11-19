/* ==========================================================================
   ConveyorVision — Dashboard client
   Luồng chính: Tổng quan (kiểm tra sẵn sàng) → Vận hành (thiết lập → chạy → tổng kết) → Lịch sử
   ========================================================================== */
"use strict";

const SCREENS = {
  overview: ["Tổng quan", "Tình trạng trạm và kết quả các phiên gần nhất"],
  live: ["Vận hành", "Thiết lập, giám sát và kết thúc phiên kiểm định"],
  history: ["Lịch sử phiên", "Kết quả, chi tiết từng sản phẩm và báo cáo CSV"],
  models: ["Mô hình & Dữ liệu", "Chỉ số đánh giá, checkpoint và tập dữ liệu"],
  settings: ["Cài đặt", "Giá trị mặc định cho các phiên kiểm định mới"],
  docs: ["Tài liệu", "Hướng dẫn vận hành và tài liệu PDF"],
};

const STATE_LABELS = {
  RUNNING: "Đang kiểm định",
  PAUSED: "Tạm dừng",
  ERROR: "Có lỗi",
  STOPPED: "Sẵn sàng",
};

const SESSION_STATUS = {
  COMPLETED: ["pass", "Hoàn tất"],
  RUNNING: ["inspecting", "Đang chạy"],
  FAILED: ["defect", "Lỗi"],
};

const TRACKER_NAMES = { "bytetrack.yaml": "ByteTrack", "botsort.yaml": "BoT-SORT" };
const VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv"];
const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;
const MAX_LOG_ROWS = 60;

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatDuration(totalSec) {
  const sec = Math.max(0, Math.floor(Number(totalSec) || 0));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

/** "/reports/x.jpg" → "reports/x.jpg" để đi qua cùng tiền tố proxy với trang. */
function relativeUrl(url) {
  return String(url || "").replace(/^\//, "");
}

function baseName(path) {
  return String(path || "").split(/[\\/]/).pop();
}

function defectRate(pass, defect) {
  const total = (pass || 0) + (defect || 0);
  return total > 0 ? ((defect / total) * 100).toFixed(1) : "0.0";
}

/** fetch + JSON, ném lỗi kèm thông điệp `detail` của FastAPI khi HTTP lỗi. */
async function api(url, options = {}) {
  const res = await fetch(url, options);
  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    data = null;
  }
  if (!res.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}`);
  }
  return data;
}

function postJson(url, body) {
  return api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

class ConveyorVisionApp {
  constructor() {
    this.screen = "overview";
    this.status = { state: "STOPPED" };

    // Dữ liệu cho bước thiết lập phiên
    this.systemCheck = null;
    this.sources = [];
    this.models = [];
    this.sourceType = "video";
    this.selectedVideo = null;
    this.defaults = { tracker: "bytetrack.yaml", default_conf: 0.45, sound_alarm: true };

    // Phiên đang chạy
    this.eventRows = new Map();
    this.elapsed = { sec: 0, at: 0 };
    this.soundEnabled = true;
    this.audioContext = null;

    // Lịch sử & dữ liệu đã nạp
    this.sessions = [];
    this.modelsTabLoaded = false;

    // Luồng video MJPEG
    this.lastFps = 0;
    this.streamActive = false;
    this.streamLost = false;
    this.streamRetryTimer = null;
    this.streamRetryDelay = 1000;

    // WebSocket
    this.ws = null;
    this.wsEverConnected = false;

    this.init();
  }

  init() {
    this.bindNavigation();
    this.bindSetup();
    this.bindRunControls();
    this.bindHistory();
    this.bindModelsScreen();
    this.bindSettings();
    this.bindModals();
    this.bindShortcuts();
    this.initVideoStream();
    this.initClock();
    this.initWebSocket();

    this.loadSettings();
    setInterval(() => this.renderElapsed(), 1000);

    const fromHash = location.hash.replace("#", "");
    this.switchScreen(SCREENS[fromHash] ? fromHash : "overview");
  }

  // ==========================================================================
  // ĐIỀU HƯỚNG
  // ==========================================================================
  bindNavigation() {
    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => this.switchScreen(btn.dataset.screen));
    });
    document.querySelectorAll("[data-goto]").forEach((btn) => {
      btn.addEventListener("click", () => this.switchScreen(btn.dataset.goto));
    });
    $("btn-goto-live").addEventListener("click", () => this.switchScreen("live"));
    $("btn-recheck").addEventListener("click", () => this.refreshReadiness());
    $("btn-menu").addEventListener("click", () => $("sidebar").classList.toggle("open"));
  }

  switchScreen(name) {
    if (!SCREENS[name]) return;
    this.screen = name;

    document.querySelectorAll(".nav-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.screen === name);
    });
    document.querySelectorAll(".screen").forEach((section) => {
      section.classList.toggle("active", section.id === `screen-${name}`);
    });
    $("page-title").textContent = SCREENS[name][0];
    $("page-subtitle").textContent = SCREENS[name][1];
    $("sidebar").classList.remove("open");
    history.replaceState(null, "", `#${name}`);

    if (name === "overview") {
      this.loadOverview();
      this.refreshReadiness();
    }
    if (name === "live" && this.phase() === "setup") this.refreshReadiness();
    if (name === "history") this.loadHistory();
    if (name === "models" && !this.modelsTabLoaded) this.loadModelsScreen();
    if (name === "settings") this.loadSettings();
    this.syncVideoStream();
  }

  // ==========================================================================
  // KIỂM TRA SẴN SÀNG (Tổng quan + bước thiết lập)
  // ==========================================================================
  async refreshReadiness() {
    try {
      const [check, sources, models] = await Promise.all([
        api("api/system_check"),
        api("api/sources"),
        api("api/models"),
      ]);
      this.systemCheck = check;
      this.sources = sources;
      this.models = models;
    } catch (err) {
      this.systemCheck = null;
      this.toast(`Không kiểm tra được hệ thống: ${err.message}`, "error");
    }
    this.renderReadiness();
    this.renderSourceList();
    this.renderModeModels();
    this.updateStartAvailability();
  }

  renderReadiness() {
    const list = $("readiness-list");
    const c = this.systemCheck;
    if (!c) {
      list.innerHTML = `<li class="check-item" data-status="fail"><span class="check-icon"></span><div><strong>Không kết nối được máy chủ</strong><p>Kiểm tra server đang chạy rồi bấm "Kiểm tra lại".</p></div></li>`;
      return;
    }

    const items = [
      c.ultralytics
        ? ["ok", "Thư viện YOLO (Ultralytics)", "Đã cài đặt"]
        : ["fail", "Chưa cài Ultralytics", "Chạy: pip install -r requirements.txt"],
      c.gpu
        ? ["ok", "Tăng tốc GPU", c.gpu]
        : ["warn", "Không có GPU", "Chạy bằng CPU — nên dùng độ phân giải 480px"],
      c.coco_models.length
        ? ["ok", "Trọng số COCO", `${baseName(c.coco_models[0])} — cho chế độ Chai & Dị vật`]
        : ["warn", "Chưa có trọng số COCO", "Đặt yolo26n.pt vào models/ để dùng chế độ Chai & Dị vật"],
      c.custom_models.length
        ? ["ok", "Checkpoint QC Đạt/Lỗi", `${c.custom_models.length} checkpoint`]
        : ["warn", "Chưa có checkpoint QC", "Huấn luyện bằng train_pilot.py rồi đặt vào models/"],
      c.videos > 0
        ? ["ok", "Video nguồn", `${c.videos} video trong raw_videos/`]
        : ["warn", "Chưa có video", "Tải video lên ở màn hình Vận hành, hoặc dùng webcam"],
    ];

    list.innerHTML = items.map(([status, title, desc]) => `
      <li class="check-item" data-status="${status}">
        <span class="check-icon"></span>
        <div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(desc)}</p></div>
      </li>
    `).join("");
  }

  // ==========================================================================
  // VẬN HÀNH — BƯỚC THIẾT LẬP
  // ==========================================================================
  bindSetup() {
    document.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.sourceType = btn.dataset.sourceType;
        document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
        $("source-pane-video").hidden = this.sourceType !== "video";
        $("source-pane-webcam").hidden = this.sourceType !== "webcam";
        this.updateStartAvailability();
      });
    });

    $("video-list").addEventListener("change", (e) => {
      if (e.target.name === "video-source") {
        this.selectedVideo = e.target.value;
        this.updateStartAvailability();
      }
    });

    const upload = $("video-file-upload");
    upload.addEventListener("change", () => {
      if (upload.files[0]) this.uploadVideo(upload.files[0]);
      upload.value = "";
    });

    const dropzone = $("dropzone");
    ["dragenter", "dragover"].forEach((evt) => dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((evt) => dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    }));
    dropzone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) this.uploadVideo(file);
    });

    document.querySelectorAll('input[name="mode"]').forEach((radio) => {
      radio.addEventListener("change", () => {
        $("custom-model-field").hidden = this.selectedMode() !== "bottle_inspection";
        this.updateStartAvailability();
      });
    });

    $("conf-slider").addEventListener("input", () => this.renderAdvancedSummary());
    $("imgsz-select").addEventListener("change", () => this.renderAdvancedSummary());
    $("tracker-select").addEventListener("change", () => this.renderAdvancedSummary());

    $("btn-start-inspection").addEventListener("click", () => this.startInspection());
  }

  selectedMode() {
    return document.querySelector('input[name="mode"]:checked').value;
  }

  renderSourceList() {
    const list = $("video-list");
    if (!this.sources.length) {
      this.selectedVideo = null;
      list.innerHTML = `<div class="empty-inline">Chưa có video nào. Tải lên bên dưới.</div>`;
      return;
    }
    if (!this.sources.some((s) => s.path === this.selectedVideo)) {
      this.selectedVideo = this.sources[0].path;
    }
    list.innerHTML = this.sources.map((s) => `
      <label class="video-option">
        <input type="radio" name="video-source" value="${escapeHtml(s.path)}" ${s.path === this.selectedVideo ? "checked" : ""} />
        <span class="vo-name" title="${escapeHtml(s.name)}">${escapeHtml(s.name)}</span>
        <span class="vo-meta">${escapeHtml(s.size_mb)} MB</span>
      </label>
    `).join("");
  }

  renderModeModels() {
    const coco = this.models.filter((m) => m.kind === "coco");
    const custom = this.models.filter((m) => m.kind === "custom");

    const cocoLabel = $("mode-model-coco");
    cocoLabel.textContent = coco.length ? coco[0].name : "Chưa có trọng số COCO";
    cocoLabel.classList.toggle("missing", !coco.length);

    const customLabel = $("mode-model-custom");
    customLabel.textContent = custom.length ? `${custom.length} checkpoint` : "Chưa có checkpoint";
    customLabel.classList.toggle("missing", !custom.length);

    const select = $("model-select");
    const previous = select.value;
    select.innerHTML = custom.map((m) => `
      <option value="${escapeHtml(m.path)}">${escapeHtml(m.name)} (${escapeHtml(m.size_mb)} MB)</option>
    `).join("");
    const preferred = custom.find((m) => m.path === previous) || custom.find((m) => m.is_active) || custom[0];
    if (preferred) select.value = preferred.path;
  }

  renderAdvancedSummary() {
    const conf = Number($("conf-slider").value).toFixed(2);
    $("conf-display-val").textContent = conf;
    const tracker = TRACKER_NAMES[$("tracker-select").value] || $("tracker-select").value;
    $("advanced-summary").textContent = `${$("imgsz-select").value}px • ${tracker} • conf ${conf}`;
  }

  /** Trả về lý do không thể bắt đầu, hoặc "" nếu sẵn sàng. */
  startBlocker() {
    const c = this.systemCheck;
    if (!c) return "Đang kiểm tra hệ thống…";
    if (!c.ultralytics) return "Máy chủ chưa cài Ultralytics (xem Tổng quan).";
    if (this.sourceType === "video" && !this.selectedVideo) return "Chọn hoặc tải lên một video.";
    if (this.selectedMode() === "bottle_and_other" && !c.coco_models.length) {
      return "Chưa có trọng số COCO (yolo26n.pt) trong models/.";
    }
    if (this.selectedMode() === "bottle_inspection" && !c.custom_models.length) {
      return "Chưa có checkpoint QC Đạt/Lỗi trong models/.";
    }
    return "";
  }

  updateStartAvailability() {
    const blocker = this.startBlocker();
    $("btn-start-inspection").disabled = !!blocker;
    $("start-hint").textContent = blocker;
  }

  uploadVideo(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!VIDEO_EXTENSIONS.includes(ext)) {
      this.toast("Chỉ hỗ trợ file .mp4, .avi, .mov, .mkv", "error");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      this.toast("File vượt quá giới hạn 200MB", "error");
      return;
    }

    const progress = $("upload-progress");
    const bar = $("upload-progress-bar");
    progress.hidden = false;
    bar.style.width = "0%";

    // XHR thay vì fetch để có tiến trình tải lên
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "api/upload_video");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) bar.style.width = `${(e.loaded / e.total) * 100}%`;
    });
    xhr.addEventListener("load", async () => {
      progress.hidden = true;
      let data = {};
      try {
        data = JSON.parse(xhr.responseText);
      } catch (_) {
        data = {};
      }
      if (xhr.status >= 200 && xhr.status < 300 && data.success) {
        this.selectedVideo = data.source;
        this.toast(`Đã tải lên ${data.filename}`, "success");
        await this.refreshReadiness();
      } else {
        this.toast(`Tải lên thất bại: ${data.detail || `HTTP ${xhr.status}`}`, "error");
      }
    });
    xhr.addEventListener("error", () => {
      progress.hidden = true;
      this.toast("Mất kết nối khi tải video lên", "error");
    });

    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  }

  async startInspection() {
    if (this.startBlocker()) return;

    const mode = this.selectedMode();
    const model = mode === "bottle_and_other"
      ? this.systemCheck.coco_models[0]
      : $("model-select").value;
    const source = this.sourceType === "webcam"
      ? String(parseInt($("webcam-index").value, 10) || 0)
      : this.selectedVideo;

    const btn = $("btn-start-inspection");
    btn.disabled = true;
    btn.textContent = "Đang khởi động…";

    this.clearEventLog();
    $("session-summary").hidden = true;

    try {
      await postJson("api/start_inspection", {
        source,
        model,
        mode,
        imgsz: parseInt($("imgsz-select").value, 10),
        conf: parseFloat($("conf-slider").value),
        tracker: $("tracker-select").value,
        loop: true,
      });
      this.applyStatus(await api("api/status"));
    } catch (err) {
      this.toast(`Không thể bắt đầu: ${err.message}`, "error");
    } finally {
      btn.textContent = "▶ Bắt đầu kiểm định";
      this.updateStartAvailability();
    }
  }

  // ==========================================================================
  // VẬN HÀNH — ĐANG CHẠY
  // ==========================================================================
  bindRunControls() {
    $("btn-pause-inspection").addEventListener("click", () => this.togglePause());
    $("btn-stop-inspection").addEventListener("click", () => this.stopInspection());
    $("btn-snapshot").addEventListener("click", () => this.takeSnapshot());
    $("btn-fullscreen").addEventListener("click", () => this.toggleFullscreen());

    $("sound-alarm-toggle").addEventListener("change", (e) => {
      this.soundEnabled = e.target.checked;
      $("setting-sound-alarm").checked = this.soundEnabled;
      postJson("api/settings", { sound_alarm: this.soundEnabled }).catch(() => {});
    });
  }

  phase() {
    return ["RUNNING", "PAUSED"].includes(this.status.state) ? "run" : "setup";
  }

  applyStatus(status) {
    this.status = Object.assign({}, this.status, status);
    const state = this.status.state || "STOPPED";

    const pill = $("state-pill");
    pill.dataset.state = state;
    $("state-text").textContent = STATE_LABELS[state] || state;

    const running = state === "RUNNING" || state === "PAUSED";
    $("nav-live-dot").hidden = !running;
    $("screen-live").dataset.phase = this.phase();

    $("btn-pause-inspection").textContent = state === "PAUSED" ? "▶ Tiếp tục" : "⏸ Tạm dừng";
    $("sb-session").textContent = this.status.session_id || "--";
    $("sb-source").textContent = this.status.source ? baseName(this.status.source) : "--";
    $("sb-source").title = this.status.source || "";
    $("sb-model").textContent = this.status.model || "--";
    $("lk-other").hidden = this.status.mode === "bottle_inspection";

    if (status.elapsed_sec !== undefined) {
      this.elapsed = { sec: status.elapsed_sec, at: performance.now() };
    }
    this.renderElapsed();

    const errorCard = $("setup-error");
    errorCard.hidden = state !== "ERROR";
    $("setup-error-text").textContent = this.status.error_message || "";

    if (status.fps !== undefined) this.lastFps = status.fps;
    if (running && status.total_pass !== undefined) this.updateLiveStats(status);
    this.updateLiveBadge();
  }

  renderElapsed() {
    if (this.phase() !== "run") return;
    const extra = (performance.now() - this.elapsed.at) / 1000;
    $("sb-elapsed").textContent = formatDuration(this.elapsed.sec + extra);
  }

  updateLiveStats(stats) {
    const setKpi = (id, value) => {
      const el = $(id);
      const text = String(value);
      if (el.textContent !== text) {
        el.textContent = text;
        const card = el.closest(".lk");
        card.classList.remove("bump");
        void card.offsetWidth; // khởi động lại animation
        card.classList.add("bump");
      }
    };
    setKpi("live-pass-val", stats.total_pass ?? 0);
    setKpi("live-defect-val", stats.total_defect ?? 0);
    setKpi("live-other-val", stats.total_other ?? 0);
    $("live-inspecting-val").textContent = stats.inspecting_count ?? 0;
    $("live-rate-val").textContent = `${Number(stats.defect_rate ?? 0).toFixed(1)}%`;
    $("live-fps-val").textContent = Number(stats.fps ?? 0).toFixed(1);
    $("live-ms-val").textContent = stats.inference_ms !== undefined ? `${Number(stats.inference_ms).toFixed(1)} ms/frame` : "-- ms";

    this.lastFps = stats.fps ?? 0;
    if (stats.frame_idx !== undefined) this.status.frame_idx = stats.frame_idx;
    this.updateLiveBadge();

    const banner = $("video-rejection-banner");
    banner.classList.toggle("active", !!stats.alert_active);
    if (stats.alert_active && stats.alert_message) {
      banner.classList.toggle("other", !stats.alert_message.includes("DEFECT"));
      banner.textContent = `⚠ ${stats.alert_message} (mô phỏng)`;
    }
  }

  clearEventLog() {
    $("live-events-tbody").innerHTML = "";
    $("live-events-empty").hidden = false;
    this.eventRows.clear();
  }

  handleTrackEvent(item) {
    const tbody = $("live-events-tbody");
    const trackId = item.track_id;
    const isOther = item.status === "OTHER" || item.category === "OTHER";
    const isCorrection = item.event === "DEFECT_CORRECTION";

    let badgeClass = "inspecting";
    let badgeText = "Đang quét";
    let typeClass = "text-warn";
    if (item.status === "DEFECT") {
      badgeClass = "defect";
      badgeText = "DEFECT";
      typeClass = "text-defect";
    } else if (item.status === "PASS") {
      badgeClass = "pass";
      badgeText = "PASS";
      typeClass = "text-pass";
    } else if (isOther) {
      badgeClass = "other";
      badgeText = "DỊ VẬT";
      typeClass = "text-other";
    }

    let row = this.eventRows.get(trackId);
    if (!row) {
      row = document.createElement("tr");
      row.innerHTML = `
        <td class="mono">#${escapeHtml(trackId)}</td>
        <td class="mono muted cell-time"></td>
        <td class="cell-type"></td>
        <td class="num mono cell-conf"></td>
        <td class="cell-status"></td>
      `;
      tbody.insertBefore(row, tbody.firstChild);
      this.eventRows.set(trackId, row);
      $("live-events-empty").hidden = true;

      while (tbody.children.length > MAX_LOG_ROWS) {
        const last = tbody.lastElementChild;
        for (const [id, el] of this.eventRows) {
          if (el === last) this.eventRows.delete(id);
        }
        last.remove();
      }
    } else if (item.status !== "INSPECTING") {
      row.classList.remove("row-flash");
      void row.offsetWidth;
      row.classList.add("row-flash");
    }

    row.querySelector(".cell-time").textContent = item.timestamp || "--";
    const typeCell = row.querySelector(".cell-type");
    typeCell.textContent = item.class_name || "--";
    typeCell.className = `cell-type ${typeClass}`;
    row.querySelector(".cell-conf").textContent = item.conf ? Number(item.conf).toFixed(2) : "--";
    row.querySelector(".cell-status").innerHTML =
      `<span class="badge ${badgeClass}">${badgeText}</span>${isCorrection ? '<span class="badge-fix">hiệu chỉnh</span>' : ""}`;

    if (item.status === "DEFECT" && this.soundEnabled) this.playAlertSound();
  }

  handleSessionFinished(summary) {
    const card = $("session-summary");
    const sid = escapeHtml(summary.session_id);
    card.innerHTML = `
      <h3>✓ Phiên đã kết thúc</h3>
      <div class="summary-sub mono">${sid} • ${escapeHtml(formatDuration(summary.duration_sec))}</div>
      <div class="summary-stats">
        <div><span>Tổng</span><strong>${escapeHtml(summary.total_items)}</strong></div>
        <div><span>PASS</span><strong class="text-pass">${escapeHtml(summary.total_pass)}</strong></div>
        <div><span>DEFECT</span><strong class="text-defect">${escapeHtml(summary.total_defect)}</strong></div>
        <div><span>Tỷ lệ lỗi</span><strong>${escapeHtml(summary.defect_rate)}%</strong></div>
      </div>
      <div class="summary-actions">
        <button type="button" class="btn btn-secondary btn-sm" data-action="detail">Xem chi tiết</button>
        <a class="btn btn-ghost btn-sm" href="api/download_report/${encodeURIComponent(summary.session_id)}" download>Tải CSV</a>
        <button type="button" class="btn btn-ghost btn-sm" data-action="dismiss">Đóng</button>
      </div>
    `;
    card.querySelector('[data-action="detail"]').addEventListener("click", () => this.viewSessionDetail(summary.session_id));
    card.querySelector('[data-action="dismiss"]').addEventListener("click", () => { card.hidden = true; });
    card.hidden = false;

    this.toast(`Phiên ${summary.session_id} đã lưu: ${summary.total_items} sản phẩm, ${summary.total_defect} lỗi`, "success");
    this.loadOverview();
    if (this.screen === "history") this.loadHistory();
  }

  async togglePause() {
    const endpoint = this.status.state === "PAUSED" ? "api/resume_inspection" : "api/pause_inspection";
    try {
      await postJson(endpoint);
    } catch (err) {
      this.toast(`Thao tác thất bại: ${err.message}`, "error");
    }
  }

  async stopInspection() {
    const ok = await this.confirm("Kết thúc phiên?", "Phiên sẽ được chốt và lưu vào lịch sử. Không thể tiếp tục phiên này sau khi kết thúc.", "Kết thúc phiên");
    if (!ok) return;
    try {
      await postJson("api/stop_inspection");
    } catch (err) {
      this.toast(`Không thể dừng: ${err.message}`, "error");
    }
  }

  async takeSnapshot() {
    try {
      const data = await postJson("api/snapshot");
      this.toast(`📸 Đã lưu ảnh — <a href="${escapeHtml(relativeUrl(data.url))}" target="_blank" rel="noopener">mở ảnh</a>`, "success", { html: true });
    } catch (err) {
      this.toast(`Không thể chụp: ${err.message}`, "error");
    }
  }

  toggleFullscreen() {
    const box = $("video-viewport-container");
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      box.requestFullscreen().catch((err) => this.toast(`Không thể bật toàn màn hình: ${err.message}`, "error"));
    }
  }

  playAlertSound() {
    try {
      if (!this.audioContext) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioCtx();
      }
      if (this.audioContext.state === "suspended") this.audioContext.resume();

      const ctx = this.audioContext;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.18);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.22);
    } catch (e) {
      console.warn("Không thể phát âm thanh cảnh báo:", e);
    }
  }

  // ==========================================================================
  // LUỒNG VIDEO MJPEG
  // ==========================================================================
  initVideoStream() {
    const img = $("video-stream-feed");

    img.addEventListener("load", () => {
      this.streamRetryDelay = 1000;
      if (this.streamLost) {
        this.streamLost = false;
        this.updateLiveBadge();
      }
    });

    // Mất kết nối → thử lại với backoff tăng dần (1s → 8s)
    img.addEventListener("error", () => {
      if (!this.streamActive) return;
      this.streamLost = true;
      this.updateLiveBadge();
      clearTimeout(this.streamRetryTimer);
      this.streamRetryTimer = setTimeout(() => this.connectVideoStream(true), this.streamRetryDelay);
      this.streamRetryDelay = Math.min(this.streamRetryDelay * 2, 8000);
    });

    document.addEventListener("visibilitychange", () => this.syncVideoStream());
  }

  // Chỉ giữ kết nối stream khi đang ở màn hình Vận hành và tab trình duyệt đang hiển thị
  syncVideoStream() {
    const shouldStream = this.screen === "live" && document.visibilityState === "visible";
    if (shouldStream) {
      this.connectVideoStream();
    } else {
      this.disconnectVideoStream();
    }
  }

  connectVideoStream(force = false) {
    if (this.streamActive && !force) return;
    clearTimeout(this.streamRetryTimer);
    this.streamActive = true;
    $("video-stream-feed").src = `api/video_feed?t=${Date.now()}`;
  }

  disconnectVideoStream() {
    clearTimeout(this.streamRetryTimer);
    if (!this.streamActive) return;
    this.streamActive = false;
    $("video-stream-feed").removeAttribute("src"); // Đóng kết nối HTTP, giải phóng băng thông
  }

  updateLiveBadge() {
    const badge = $("video-live-badge");
    const text = $("video-live-text");
    const state = this.status.state;

    badge.classList.remove("idle", "paused", "lost");
    if (this.streamLost) {
      badge.classList.add("lost");
      text.textContent = "MẤT TÍN HIỆU • ĐANG KẾT NỐI LẠI";
    } else if (state === "RUNNING" && !this.status.frame_idx) {
      badge.classList.add("paused");
      text.textContent = "ĐANG NẠP MÔ HÌNH…"; // lần đầu nạp model lên GPU có thể mất vài giây
    } else if (state === "RUNNING") {
      text.textContent = `TRỰC TIẾP • ${Number(this.lastFps).toFixed(1)} FPS`;
    } else if (state === "PAUSED") {
      badge.classList.add("paused");
      text.textContent = "TẠM DỪNG";
    } else if (state === "ERROR") {
      badge.classList.add("lost");
      text.textContent = "LỖI NGUỒN VIDEO";
    } else {
      badge.classList.add("idle");
      text.textContent = "CHỜ KHỞI CHẠY";
    }
  }

  // ==========================================================================
  // WEBSOCKET
  // ==========================================================================
  initWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const basePath = location.pathname.replace(/[^/]*$/, "");
    const url = `${protocol}//${location.host}${basePath}ws/events`;

    const setConn = (state, text) => {
      $("conn-indicator").dataset.state = state;
      $("conn-text").textContent = text;
    };

    const connect = () => {
      this.ws = new WebSocket(url);

      this.ws.addEventListener("open", () => {
        // Máy chủ vừa khởi động lại → luồng MJPEG cũ đã chết, kết nối lại
        if (this.wsEverConnected) {
          if (this.streamActive) this.connectVideoStream(true);
          this.refreshReadiness();
        }
        this.wsEverConnected = true;
        setConn("online", "Đã kết nối máy chủ");
      });

      this.ws.addEventListener("message", (event) => {
        if (event.data === "pong") return;
        try {
          this.handleServerEvent(JSON.parse(event.data));
        } catch (e) {
          console.warn("Lỗi phân tích sự kiện WebSocket:", e);
        }
      });

      this.ws.addEventListener("close", () => {
        setConn("offline", "Mất kết nối — đang thử lại…");
        setTimeout(connect, 2500);
      });
    };

    connect();
    setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send("ping");
    }, 25000);
  }

  handleServerEvent({ type, payload }) {
    if (type === "init_status" || type === "status_changed") this.applyStatus(payload);
    else if (type === "stats_update") this.updateLiveStats(payload);
    else if (type === "track_event") this.handleTrackEvent(payload);
    else if (type === "session_finished") this.handleSessionFinished(payload);
  }

  // ==========================================================================
  // TỔNG QUAN
  // ==========================================================================
  async loadOverview() {
    try {
      const sessions = await api("api/history");
      let items = 0;
      let pass = 0;
      let defect = 0;
      sessions.forEach((s) => {
        items += s.total_items || 0;
        pass += s.total_pass || 0;
        defect += s.total_defect || 0;
      });

      $("ov-total-sessions").textContent = sessions.length.toLocaleString("vi-VN");
      $("ov-total-inspected").textContent = items.toLocaleString("vi-VN");
      $("ov-total-pass").textContent = pass.toLocaleString("vi-VN");
      $("ov-total-defect").textContent = defect.toLocaleString("vi-VN");
      $("ov-defect-rate").textContent = `${defectRate(pass, defect)}%`;

      const tbody = $("overview-recent-sessions-tbody");
      if (!sessions.length) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty">Chưa có phiên nào. Bắt đầu phiên đầu tiên ở màn hình Vận hành.</td></tr>`;
        return;
      }
      tbody.innerHTML = sessions.slice(0, 5).map((s) => `
        <tr>
          <td class="mono">${escapeHtml(s.session_id)}</td>
          <td class="muted">${escapeHtml(s.start_time)}</td>
          <td title="${escapeHtml(s.source)}">${escapeHtml(baseName(s.source))}</td>
          <td class="num"><strong>${escapeHtml(s.total_items)}</strong></td>
          <td class="num text-pass">${escapeHtml(s.total_pass)}</td>
          <td class="num text-defect">${escapeHtml(s.total_defect)}</td>
          <td class="num">${escapeHtml(s.defect_rate)}%</td>
        </tr>
      `).join("");
    } catch (err) {
      console.warn("Lỗi nạp dữ liệu tổng quan:", err);
    }
  }

  // ==========================================================================
  // LỊCH SỬ PHIÊN
  // ==========================================================================
  bindHistory() {
    $("btn-history-refresh").addEventListener("click", () => this.loadHistory());
    $("history-search").addEventListener("input", () => this.renderHistory());

    $("history-sessions-tbody").addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-action]");
      if (!btn) return;
      const sid = btn.dataset.sid;
      if (btn.dataset.action === "detail") this.viewSessionDetail(sid);
      if (btn.dataset.action === "delete") this.deleteSession(sid);
    });
  }

  async loadHistory() {
    try {
      this.sessions = await api("api/history");
    } catch (err) {
      this.toast(`Không tải được lịch sử: ${err.message}`, "error");
      this.sessions = [];
    }
    this.renderHistory();
  }

  renderHistory() {
    const tbody = $("history-sessions-tbody");
    const q = $("history-search").value.trim().toLowerCase();
    const rows = this.sessions.filter((s) =>
      !q || `${s.session_id} ${s.source} ${s.model_name}`.toLowerCase().includes(q));

    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="10" class="empty">${this.sessions.length ? "Không có phiên khớp từ khoá." : "Chưa có phiên nào được lưu."}</td></tr>`;
      return;
    }

    tbody.innerHTML = rows.map((s) => {
      const [cls, label] = SESSION_STATUS[s.status] || ["", s.status || "--"];
      const sid = escapeHtml(s.session_id);
      return `
        <tr>
          <td class="mono">${sid}</td>
          <td class="muted">${escapeHtml(s.start_time)}</td>
          <td class="num mono">${s.duration_sec ? escapeHtml(formatDuration(s.duration_sec)) : "--"}</td>
          <td title="${escapeHtml(s.source)}">${escapeHtml(baseName(s.source))}</td>
          <td class="num"><strong>${escapeHtml(s.total_items)}</strong></td>
          <td class="num text-pass">${escapeHtml(s.total_pass)}</td>
          <td class="num text-defect">${escapeHtml(s.total_defect)}</td>
          <td class="num ${s.defect_rate > 15 ? "text-defect" : ""}">${escapeHtml(s.defect_rate)}%</td>
          <td><span class="badge ${cls}" title="${escapeHtml(s.error_message || "")}">${escapeHtml(label)}</span></td>
          <td>
            <div class="cell-actions">
              <button type="button" class="btn btn-secondary btn-sm" data-action="detail" data-sid="${sid}">Chi tiết</button>
              <a class="btn btn-ghost btn-sm" href="api/download_report/${encodeURIComponent(s.session_id)}" download>CSV</a>
              <button type="button" class="btn btn-danger btn-sm" data-action="delete" data-sid="${sid}" aria-label="Xoá phiên">✕</button>
            </div>
          </td>
        </tr>
      `;
    }).join("");
  }

  async viewSessionDetail(sessionId) {
    let session;
    try {
      session = await api(`api/history/${encodeURIComponent(sessionId)}`);
    } catch (err) {
      this.toast(`Không tìm thấy phiên: ${err.message}`, "error");
      return;
    }

    $("session-detail-title").textContent = `Phiên ${session.session_id}`;
    const tracks = session.tracks || [];
    const trackTable = tracks.length
      ? `
        <div class="table-wrap">
          <table class="table table-compact">
            <thead><tr><th>Track</th><th>Phân loại</th><th class="num">Conf</th><th class="num">Số frame</th><th>Xuất hiện</th><th>Kết quả</th></tr></thead>
            <tbody>
              ${tracks.map((t) => {
                const state = String(t.state || "");
                const cls = state === "DEFECT" ? "defect" : state === "OTHER" ? "other" : "pass";
                return `
                  <tr>
                    <td class="mono">#${escapeHtml(t.track_id)}</td>
                    <td>${escapeHtml(t.class_name)}</td>
                    <td class="num mono">${escapeHtml(t.max_conf)}</td>
                    <td class="num">${escapeHtml(t.obs_votes)}</td>
                    <td class="mono muted">${escapeHtml(t.first_ts)}s → ${escapeHtml(t.last_ts)}s</td>
                    <td><span class="badge ${cls}">${escapeHtml(state)}</span>${t.was_corrected ? '<span class="badge-fix">hiệu chỉnh</span>' : ""}</td>
                  </tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>`
      : `<div class="empty-inline">Phiên này không có sản phẩm nào được chốt.</div>`;

    $("session-detail-body").innerHTML = `
      <div class="summary-stats">
        <div><span>Tổng</span><strong>${escapeHtml(session.total_items)}</strong></div>
        <div><span>PASS</span><strong class="text-pass">${escapeHtml(session.total_pass)}</strong></div>
        <div><span>DEFECT</span><strong class="text-defect">${escapeHtml(session.total_defect)}</strong></div>
        <div><span>Tỷ lệ lỗi</span><strong>${escapeHtml(session.defect_rate)}%</strong></div>
      </div>
      <div class="detail-meta">
        <div>Nguồn: <strong>${escapeHtml(session.source)}</strong></div>
        <div>Mô hình: <strong>${escapeHtml(session.model_name)}</strong></div>
        <div>Bắt đầu: <strong>${escapeHtml(session.start_time)}</strong></div>
        <div>Kết thúc: <strong>${escapeHtml(session.end_time || "--")}</strong></div>
        <div>Thời lượng: <strong>${escapeHtml(formatDuration(session.duration_sec))}</strong></div>
        <div>Tracker / conf: <strong>${escapeHtml(TRACKER_NAMES[session.tracker] || session.tracker)} / ${escapeHtml(session.conf_thresh)}</strong></div>
      </div>
      ${session.error_message ? `<div class="error-card"><strong>Lỗi</strong><p>${escapeHtml(session.error_message)}</p></div>` : ""}
      <div style="margin: 12px 0 6px; display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: var(--text);">Sản phẩm đã chốt (${tracks.length})</strong>
        <a class="btn btn-ghost btn-sm" href="api/download_report/${encodeURIComponent(session.session_id)}" download>Tải CSV</a>
      </div>
      ${trackTable}
    `;
    this.openModal("session-detail-modal");
  }

  async deleteSession(sessionId) {
    const ok = await this.confirm("Xoá phiên?", `Phiên ${sessionId} và file CSV của nó sẽ bị xoá vĩnh viễn.`, "Xoá");
    if (!ok) return;
    try {
      await api(`api/history/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      this.toast(`Đã xoá phiên ${sessionId}`, "success");
      this.loadHistory();
      this.loadOverview();
    } catch (err) {
      this.toast(`Không thể xoá: ${err.message}`, "error");
    }
  }

  // ==========================================================================
  // MÔ HÌNH & DỮ LIỆU
  // ==========================================================================
  bindModelsScreen() {
    document.querySelectorAll("#screen-models .tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll("#screen-models .tab").forEach((t) => t.classList.toggle("active", t === tab));
        document.querySelectorAll("#screen-models .tab-pane").forEach((pane) => {
          pane.classList.toggle("active", pane.id === `tab-${tab.dataset.tab}`);
        });
      });
    });
  }

  loadModelsScreen() {
    this.modelsTabLoaded = true;
    // Mỗi tab hiển thị ngay khi dữ liệu của nó về (dataset_info lần đầu phải đọc hàng nghìn file nhãn)
    const load = (url, render) => api(url).then(render).catch((err) => {
      this.modelsTabLoaded = false; // thử lại lần sau
      this.toast(`Không tải được ${url}: ${err.message}`, "warn");
    });
    load("api/evaluation", (d) => this.renderEvaluation(d));
    load("api/models", (d) => this.renderCheckpoints(d));
    load("api/dataset_info", (d) => this.renderDataset(d));
  }

  renderEvaluation(data) {
    $("eval-empty").hidden = !!data.available;
    $("eval-content").hidden = !data.available;
    if (!data.available) return;

    const pct = (v) => `${escapeHtml(v)}%`;
    const CLASS_LABELS = { pass: "pass — đạt", defect: "defect — lỗi" };

    $("eval-test-size").textContent = `${data.num_images} ảnh • ${data.num_boxes} box • imgsz ${data.imgsz}`;
    const ov = data.overall;
    $("eval-metrics-tbody").innerHTML = `
      <tr class="row-total">
        <td>Tất cả</td><td class="num">${escapeHtml(data.num_images)}</td><td class="num">${escapeHtml(data.num_boxes)}</td>
        <td class="num">${pct(ov.precision)}</td><td class="num">${pct(ov.recall)}</td>
        <td class="num">${pct(ov.map50)}</td><td class="num">${pct(ov.map50_95)}</td>
      </tr>
      ${data.classes.map((c) => `
        <tr>
          <td class="${c.name === "defect" ? "text-defect" : "text-pass"}">${escapeHtml(CLASS_LABELS[c.name] || c.name)}</td>
          <td class="num">${escapeHtml(c.images)}</td><td class="num">${escapeHtml(c.boxes)}</td>
          <td class="num">${pct(c.precision)}</td><td class="num">${pct(c.recall)}</td>
          <td class="num"><strong>${pct(c.map50)}</strong></td><td class="num">${pct(c.map50_95)}</td>
        </tr>`).join("")}
    `;

    const conf = data.confusion || {};
    $("eval-confusion").innerHTML = Object.keys(conf).length ? `
      <p class="muted small">Tại ngưỡng vận hành conf=${escapeHtml(data.confusion_conf)}, theo nhãn thật:</p>
      <div class="confusion-grid">
        ${Object.entries(conf).map(([name, c]) => `
          <div class="confusion-item">
            <strong class="${name === "defect" ? "text-defect" : "text-pass"}">${escapeHtml(name)}</strong>
            <span>Đúng <b>${pct(c.correct_pct)}</b></span>
            <span>Nhầm lớp <b>${pct(c.as_other_class_pct)}</b></span>
            <span>Bỏ sót <b>${pct(c.missed_pct)}</b></span>
          </div>`).join("")}
      </div>` : "";

    const m = data.model;
    const t = data.training;
    const rows = [
      ["Checkpoint", `<span class="mono">${escapeHtml(m.file)}</span>`],
      ["Kiến trúc", `${escapeHtml(m.architecture)} • ${(m.parameters / 1e6).toFixed(2)}M tham số • ${escapeHtml(m.size_mb)} MB`],
    ];
    if (t) {
      rows.push(["Huấn luyện", `${escapeHtml(t.epochs_run)} epoch (tốt nhất: ${escapeHtml(t.best_epoch)}) • ${escapeHtml(formatDuration(t.time_sec))}`]);
    }
    if (data.speed_gpu) {
      const g = data.speed_gpu;
      const total = g.preprocess_ms + g.inference_ms + g.postprocess_ms;
      rows.push([`GPU — ${escapeHtml(g.device)}`, `<span class="speed-big">${(1000 / total).toFixed(0)} FPS</span> <span class="muted mono small">${total.toFixed(1)} ms/ảnh</span>`]);
    }
    if (data.speed_cpu) {
      const c = data.speed_cpu;
      rows.push([`CPU — ${escapeHtml(c.device)}`, `<span class="speed-big">${escapeHtml(c.fps)} FPS</span> <span class="muted mono small">${(c.preprocess + c.inference + c.postprocess).toFixed(1)} ms/ảnh @${escapeHtml(c.imgsz)}px</span>`]);
    }
    rows.push(["Đánh giá lúc", `<span class="mono">${escapeHtml(data.generated_at.replace("T", " "))}</span>`]);
    $("eval-speed-details").innerHTML = rows.map(([k, v]) => `<div class="speed-row"><span>${k}</span><span>${v}</span></div>`).join("");

    const split = data.dataset_split;
    const notice = $("eval-split-notice");
    notice.hidden = !split;
    if (split) {
      notice.innerHTML = `
        <strong>Chia dữ liệu chống rò rỉ</strong>
        <p>${escapeHtml(split.method)}. Số ảnh val/test có bản gần trùng trong train:
        split ngẫu nhiên gốc của Roboflow <b>${escapeHtml(split.leak_roboflow_split)}</b> →
        split theo nhóm <b>${escapeHtml(split.leak_group_split)}</b>. Vì vậy số liệu dưới đây phản ánh khả năng tổng quát hoá thật hơn so với chia ngẫu nhiên.</p>
      `;
    }

    this.renderCharts(data.charts || []);
  }

  renderCharts(charts) {
    const nav = $("chart-tabs-nav");
    const view = $("chart-display-container");
    if (!charts.length) {
      view.innerHTML = `<div class="empty-inline">Không có biểu đồ.</div>`;
      return;
    }

    const show = (chart) => {
      view.innerHTML = `
        <p>${escapeHtml(chart.description)}</p>
        <img src="${escapeHtml(relativeUrl(chart.image_url))}" alt="${escapeHtml(chart.title)}" loading="lazy" />
      `;
      view.querySelector("img").addEventListener("click", () => this.openImageZoom(relativeUrl(chart.image_url), chart.title));
    };

    nav.innerHTML = charts.map((ch, i) => `
      <button type="button" class="chart-tab ${i === 0 ? "active" : ""}" data-idx="${i}">${escapeHtml(ch.title.split("(")[0].trim())}</button>
    `).join("");
    nav.querySelectorAll(".chart-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        nav.querySelectorAll(".chart-tab").forEach((b) => b.classList.toggle("active", b === btn));
        show(charts[Number(btn.dataset.idx)]);
      });
    });
    show(charts[0]);
  }

  renderDataset(ds) {
    if (!ds.splits) return;
    if (!ds.total_images) {
      $("dataset-splits-container").innerHTML = "";
      $("dataset-notes").innerHTML = `<p>Chưa có dữ liệu trong <code>data/</code>. Tải dataset rồi chạy <code>python scripts/prepare_dataset.py</code>.</p>`;
      return;
    }
    const split = (label, s) => `
      <div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${escapeHtml(s.count)}</div><div class="kpi-sub">${escapeHtml(s.pct)}%</div></div>`;
    $("dataset-splits-container").innerHTML =
      split("Train", ds.splits.train) +
      split("Val", ds.splits.val) +
      split("Test", ds.splits.test) +
      `<div class="kpi"><div class="kpi-label">Tổng số ảnh</div><div class="kpi-value">${escapeHtml(ds.total_images)}</div><div class="kpi-sub">${escapeHtml(ds.total_boxes)} box</div></div>`;

    const [pass, defect] = ds.classes;
    const report = ds.split_report;
    const sources = report ? Object.entries(report.sources).map(([k, v]) => `<code>${escapeHtml(k)}</code> ${escapeHtml(v)} ảnh`).join(", ") : "";
    $("dataset-notes").innerHTML = `
      <p><strong>Nhãn:</strong> lớp 0 <code>pass</code> — ${escapeHtml(pass.description)} (${escapeHtml(pass.boxes)} box, ${escapeHtml(pass.ratio_pct)}%);
      lớp 1 <code>defect</code> — ${escapeHtml(defect.description)} (${escapeHtml(defect.boxes)} box, ${escapeHtml(defect.ratio_pct)}%).</p>
      ${report ? `<p><strong>Nguồn ảnh thật (Roboflow):</strong> ${sources}.</p>
      <p><strong>Cách chia:</strong> ${escapeHtml(report.method)}.</p>` : ""}
    `;
  }

  renderCheckpoints(models) {
    const tbody = $("models-list-tbody");
    if (!models.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty">Chưa có checkpoint nào trong models/.</td></tr>`;
      return;
    }
    tbody.innerHTML = models.map((m) => `
      <tr>
        <td class="mono">${escapeHtml(m.name)}</td>
        <td>${m.kind === "coco"
          ? '<span class="badge info">COCO — Chai &amp; Dị vật</span>'
          : '<span class="badge pass">QC Đạt/Lỗi</span>'}</td>
        <td class="num">${escapeHtml(m.size_mb)} MB</td>
        <td class="muted">${escapeHtml(m.modified)}</td>
        <td class="muted">${m.kind === "coco" ? "Trọng số gốc YOLO huấn luyện trên COCO (80 lớp)" : escapeHtml(m.description)}</td>
      </tr>
    `).join("");
  }

  // ==========================================================================
  // CÀI ĐẶT (giá trị mặc định cho phiên mới)
  // ==========================================================================
  bindSettings() {
    $("setting-conf").addEventListener("input", (e) => {
      $("setting-conf-val").textContent = Number(e.target.value).toFixed(2);
    });
    $("btn-save-settings").addEventListener("click", () => this.saveSettings());
  }

  async loadSettings() {
    try {
      const s = await api("api/settings");
      this.defaults = Object.assign(this.defaults, s);
    } catch (err) {
      console.warn("Lỗi nạp cài đặt:", err);
    }
    const d = this.defaults;

    $("setting-tracker").value = d.tracker;
    $("setting-conf").value = d.default_conf;
    $("setting-conf-val").textContent = Number(d.default_conf).toFixed(2);
    if (d.lost_track_buffer !== undefined) $("setting-lost-buffer").value = d.lost_track_buffer;
    if (d.min_observation_votes !== undefined) $("setting-min-votes").value = d.min_observation_votes;
    $("setting-auto-csv").checked = d.auto_save_csv !== false;
    $("setting-sound-alarm").checked = d.sound_alarm !== false;

    this.soundEnabled = d.sound_alarm !== false;
    $("sound-alarm-toggle").checked = this.soundEnabled;

    // Điền giá trị mặc định vào bước "Tham số nâng cao" (chỉ khi chưa chạy)
    if (this.phase() === "setup") {
      $("tracker-select").value = d.tracker;
      $("conf-slider").value = d.default_conf;
    }
    this.renderAdvancedSummary();
  }

  async saveSettings() {
    const payload = {
      tracker: $("setting-tracker").value,
      default_conf: parseFloat($("setting-conf").value),
      lost_track_buffer: parseInt($("setting-lost-buffer").value, 10) || 30,
      min_observation_votes: parseInt($("setting-min-votes").value, 10) || 3,
      auto_save_csv: $("setting-auto-csv").checked,
      sound_alarm: $("setting-sound-alarm").checked,
    };
    try {
      await postJson("api/settings", payload);
      this.toast("Đã lưu cài đặt", "success");
      this.loadSettings();
    } catch (err) {
      this.toast(`Không thể lưu cài đặt: ${err.message}`, "error");
    }
  }

  // ==========================================================================
  // MODAL, XÁC NHẬN, TOAST, PHÍM TẮT, ĐỒNG HỒ
  // ==========================================================================
  bindModals() {
    document.querySelectorAll(".modal").forEach((modal) => {
      modal.addEventListener("click", (e) => {
        if (e.target === modal || e.target.closest(".modal-close")) this.closeModal(modal);
      });
    });
  }

  openModal(id) {
    $(id).classList.add("open");
  }

  closeModal(modal) {
    modal.classList.remove("open");
    if (modal.id === "confirm-modal" && this.confirmResolve) {
      this.confirmResolve(false);
      this.confirmResolve = null;
    }
  }

  openImageZoom(url, title) {
    $("image-zoom-img").src = url;
    $("image-zoom-img").alt = title || "";
    $("image-zoom-title").textContent = title || "Biểu đồ";
    this.openModal("image-zoom-modal");
  }

  /** Hộp thoại xác nhận thay cho window.confirm. Trả về Promise<boolean>. */
  confirm(title, message, okLabel = "Đồng ý") {
    $("confirm-title").textContent = title;
    $("confirm-message").textContent = message;
    $("confirm-ok").textContent = okLabel;
    this.openModal("confirm-modal");
    $("confirm-ok").focus();

    return new Promise((resolve) => {
      this.confirmResolve = resolve;
      const finish = (value) => {
        $("confirm-ok").onclick = null;
        $("confirm-cancel").onclick = null;
        this.confirmResolve = null;
        $("confirm-modal").classList.remove("open");
        resolve(value);
      };
      $("confirm-ok").onclick = () => finish(true);
      $("confirm-cancel").onclick = () => finish(false);
    });
  }

  toast(message, type = "info", { html = false, timeout = 4500 } = {}) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    if (html) el.innerHTML = message;
    else el.textContent = message;
    $("toast-region").appendChild(el);
    setTimeout(() => {
      el.classList.add("leaving");
      setTimeout(() => el.remove(), 250);
    }, timeout);
  }

  bindShortcuts() {
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        document.querySelectorAll(".modal.open").forEach((m) => this.closeModal(m));
        return;
      }
      if (this.screen !== "live" || document.querySelector(".modal.open")) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.target.closest("input, select, textarea, [contenteditable]")) return;
      if (e.key === " " && e.target.closest("button, a, summary")) return; // để trình duyệt tự kích hoạt nút đang focus

      const key = e.key.toLowerCase();
      if (key === "f") {
        e.preventDefault();
        this.toggleFullscreen();
      } else if (this.phase() === "run" && key === " ") {
        e.preventDefault();
        this.togglePause();
      } else if (this.phase() === "run" && key === "s") {
        e.preventDefault();
        this.takeSnapshot();
      }
    });
  }

  initClock() {
    const pad = (n) => String(n).padStart(2, "0");
    const tick = () => {
      const now = new Date();
      $("live-clock").textContent =
        `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())} · ${pad(now.getDate())}/${pad(now.getMonth() + 1)}/${now.getFullYear()}`;
    };
    tick();
    setInterval(tick, 1000);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new ConveyorVisionApp();
});
