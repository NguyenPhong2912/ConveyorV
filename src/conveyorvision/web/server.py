"""FastAPI Web Server cho ConveyorVision.

Cung cấp:
- Single Page Application (SPA) dashboard chuyên nghiệp.
- Endpoint Video MJPEG Stream `/api/video_feed`.
- WebSocket `/ws/events` phát tán sự kiện thời gian thực.
- REST API điều khiển kiểm định, cấu hình hệ thống, quản lý phiên, truy vấn đánh giá mô hình và tập dữ liệu.
- Cơ chế bảo mật: Chống Path Traversal, giới hạn dung lượng tải video, cấu hình CORS an toàn.
"""

import asyncio
import csv
import json
import logging
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from conveyorvision.web.session_manager import session_manager
from conveyorvision.web.worker import worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("conveyorvision.server")

# Đường dẫn thư mục chuẩn
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
EVAL_DIR = REPO_ROOT / "doc" / "evaluation"
REPORTS_DIR = REPO_ROOT / "reports"
SNAPSHOTS_DIR = REPORTS_DIR / "snapshots"
RAW_VIDEOS_DIR = REPO_ROOT / "raw_videos"
MODELS_DIR = REPO_ROOT / "models"

# Đảm bảo các thư mục luôn sẵn sàng trước khi mount tĩnh
RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
# Trọng số COCO gốc (yolo26n.pt, yolov8s.pt...) — dùng cho chế độ "Chai & Dị vật khác"
COCO_WEIGHTS_RE = re.compile(r"^yolo(v\d+|\d+)[nsmlx]\.pt$", re.IGNORECASE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ws_manager.set_loop(asyncio.get_running_loop())
    logger.info("ConveyorVision Web Server đã khởi động thành công.")
    yield


app = FastAPI(
    title="ConveyorVision Dashboard",
    description="Hệ thống kiểm định & phân loại sản phẩm thời gian thực trên băng chuyền công nghiệp",
    version="2.0.0",
    lifespan=lifespan
)

# Cấu hình CORS an toàn cho trạm cục bộ
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount thư mục tĩnh (chắc chắn snapshots tồn tại để bản clone mới không bị 404)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if EVAL_DIR.exists():
    app.mount("/doc/evaluation", StaticFiles(directory=str(EVAL_DIR)), name="eval_assets")

app.mount("/reports/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Quản lý kết nối WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client WebSocket kết nối. Tổng số: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client WebSocket ngắt kết nối. Còn lại: {len(self.active_connections)}")

    async def broadcast_json(self, data: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)

    def dispatch_from_thread(self, event_type: str, payload: Dict[str, Any]):
        """Hàm callback được gọi từ worker thread để phát tán qua WebSocket."""
        try:
            loop = self._loop
            if loop and loop.is_running():
                msg = {"type": event_type, "payload": payload, "timestamp": time.time()}
                asyncio.run_coroutine_threadsafe(self.broadcast_json(msg), loop)
        except Exception as e:
            logger.warning(f"Lỗi gửi tin WebSocket: {e}")


ws_manager = ConnectionManager()
worker.subscribe_events(ws_manager.dispatch_from_thread)


# ==============================================================================
# 1. GIAO DIỆN CHÍNH (HTML SPA)
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ==============================================================================
# 2. VIDEO STREAMING (MJPEG)
# ==============================================================================
MJPEG_BOUNDARY = b"--frame\r\n"
MJPEG_POLL_SEC = 0.01        # Chu kỳ kiểm tra khung hình mới (~100Hz, không chiếm thread)
MJPEG_KEEPALIVE_SEC = 1.0    # Gửi lại khung hình hiện tại khi hệ thống đứng yên (giữ kết nối sống)


def _mjpeg_part(jpeg: bytes) -> bytes:
    # Boundary đặt SAU mỗi khung hình: trình duyệt hiển thị ngay, không phải chờ khung kế tiếp
    return (
        b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
        + jpeg + b"\r\n" + MJPEG_BOUNDARY
    )


async def mjpeg_frame_generator():
    """Async generator stream khung hình JPEG mới nhất.

    - Không giữ thread của threadpool cho mỗi client (nhiều tab không làm cạn threadpool).
    - Client chậm tự động bỏ qua khung cũ, luôn nhận khung mới nhất (không tích lũy độ trễ).
    - Khi hệ thống dừng chỉ gửi keepalive 1 lần/giây thay vì lặp liên tục.
    """
    yield MJPEG_BOUNDARY
    last_seq = -1
    last_sent = 0.0
    while True:
        seq = worker._frame_seq
        now = time.monotonic()
        if seq != last_seq or now - last_sent >= MJPEG_KEEPALIVE_SEC:
            frame_bytes = worker.latest_jpeg
            if frame_bytes:
                last_seq, last_sent = seq, now
                yield _mjpeg_part(frame_bytes)
        await asyncio.sleep(MJPEG_POLL_SEC)


@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ==============================================================================
# 3. WEBSOCKET REAL-TIME EVENTS
# ==============================================================================
@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Gửi ngay trạng thái hiện tại khi client kết nối
        await websocket.send_json({
            "type": "init_status",
            "payload": worker.get_status()
        })
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"Lỗi WebSocket: {e}")
        ws_manager.disconnect(websocket)


# ==============================================================================
# 4. REST API - ĐIỀU KHIỂN KIỂM ĐỊNH & CÀI ĐẶT
# ==============================================================================
class StartInspectionRequest(BaseModel):
    source: str = "raw_videos/conveyor_demo.mp4"
    model: str = "models/conveyor_qc_yolo26n.pt"
    mode: str = "bottle_and_other"  # "bottle_and_other" hoặc "bottle_inspection"
    conf: float = 0.45
    tracker: str = "bytetrack.yaml"
    imgsz: int = 480
    loop: bool = True
    lost_track_buffer: Optional[int] = None
    min_observation_votes: Optional[int] = None
    auto_save_csv: Optional[bool] = None


class SettingsUpdateRequest(BaseModel):
    tracker: Optional[str] = None
    lost_track_buffer: Optional[int] = None
    min_observation_votes: Optional[int] = None
    auto_save_csv: Optional[bool] = None
    sound_alarm: Optional[bool] = None
    default_conf: Optional[float] = None


@app.get("/api/status")
async def get_system_status():
    return worker.get_status()


@app.get("/api/settings")
async def get_settings():
    return session_manager.get_settings()


@app.post("/api/settings")
async def update_settings(req: SettingsUpdateRequest):
    dump_fn = getattr(req, "model_dump", None) or getattr(req, "dict")
    payload = {k: v for k, v in dump_fn().items() if v is not None}
    saved = session_manager.save_settings(payload)
    return {"success": True, "settings": saved}


@app.post("/api/start_inspection")
def start_inspection(req: StartInspectionRequest):
    res = worker.start(
        source=req.source,
        model_path=req.model,
        inspection_mode=req.mode,
        conf_thresh=req.conf,
        tracker_config=req.tracker,
        imgsz=req.imgsz,
        loop_video=req.loop,
        lost_track_buffer=req.lost_track_buffer,
        min_observation_votes=req.min_observation_votes,
        auto_save_csv=req.auto_save_csv
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message", "Không thể khởi động kiểm định"))
    return res


@app.post("/api/pause_inspection")
def pause_inspection():
    res = worker.pause()
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/resume_inspection")
def resume_inspection():
    res = worker.resume()
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/stop_inspection")
def stop_inspection():
    res = worker.stop()
    return res


@app.post("/api/snapshot")
async def capture_snapshot():
    path = worker.capture_snapshot()
    if not path:
        raise HTTPException(status_code=400, detail="Không có khung hình để chụp")
    filename = Path(path).name
    return {"success": True, "url": f"/reports/snapshots/{filename}", "path": path}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Cho phép tải file video lên trạm với xác thực bảo mật và giới hạn dung lượng."""
    original_filename = Path(file.filename or "").name
    ext = Path(original_filename).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file video (.mp4, .avi, .mov, .mkv)")

    # Chuẩn hóa tên file an toàn (chống chèn ký tự lạ và Path Traversal)
    base_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(original_filename).stem)
    if not base_stem:
        base_stem = f"upload_{int(time.time())}"
    safe_filename = f"{base_stem}{ext}"

    dest_path = (RAW_VIDEOS_DIR / safe_filename).resolve()
    # Kiểm tra nghiêm ngặt đường dẫn đích nằm trong RAW_VIDEOS_DIR
    if not dest_path.is_relative_to(RAW_VIDEOS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ hoặc chứa đường dẫn trái phép.")

    size_read = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    try:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                size_read += len(chunk)
                if size_read > MAX_UPLOAD_SIZE:
                    buffer.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="Dung lượng file vượt quá giới hạn 200MB.")
                buffer.write(chunk)
        rel_path = f"raw_videos/{safe_filename}"
        return {"success": True, "source": rel_path, "filename": safe_filename}
    except HTTPException:
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {e}")


# ==============================================================================
# 5. REST API - LỊCH SỬ KIỂM ĐỊNH & BÁO CÁO CSV
# ==============================================================================
@app.get("/api/history")
async def get_history():
    return session_manager.get_all_sessions()


@app.get("/api/history/{session_id}")
async def get_session_detail(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên kiểm định")
    return session


@app.delete("/api/history/{session_id}")
async def delete_session(session_id: str):
    ok = session_manager.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên cần xóa")
    return {"success": True, "session_id": session_id}


@app.get("/api/download_report/{session_id}")
async def download_report(session_id: str):
    csv_path = session_manager.get_session_csv_path(session_id)
    if not csv_path or not csv_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file log CSV của phiên này")
    return FileResponse(
        path=str(csv_path),
        filename=f"Bao_cao_kiem_dinh_{session_id}.csv",
        media_type="text/csv"
    )


@app.get("/api/download_pdf_report")
async def download_pdf_report():
    """Tải về bản báo cáo nghiệm thu kỹ thuật định dạng PDF hoàn chỉnh của dự án."""
    pdf_path = REPO_ROOT / "doc" / "BAO_CAO_TONG_KET_CONVEYORVISION.pdf"
    if not pdf_path.exists():
        try:
            import importlib.util
            gen_script = REPO_ROOT / "scripts" / "generate_pdf_report.py"
            if gen_script.exists():
                spec = importlib.util.spec_from_file_location("generate_pdf_report", str(gen_script))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.build_pdf(str(pdf_path))
        except Exception as e:
            logger.error(f"Lỗi sinh báo cáo PDF: {e}")

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file báo cáo PDF tổng kết")

    return FileResponse(
        path=str(pdf_path),
        filename="BaoCao_TongKet_ConveyorVision.pdf",
        media_type="application/pdf"
    )


@app.get("/api/download_pdf_uiux")
async def download_pdf_uiux():
    """Tải về tài liệu hướng dẫn sử dụng giao diện và vận hành UI/UX định dạng PDF."""
    pdf_path = REPO_ROOT / "doc" / "HUONG_DAN_SU_DUNG_UI_UX.pdf"
    if not pdf_path.exists():
        try:
            import importlib.util
            gen_script = REPO_ROOT / "scripts" / "generate_pdf_uiux.py"
            if gen_script.exists():
                spec = importlib.util.spec_from_file_location("generate_pdf_uiux", str(gen_script))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.build_pdf(str(pdf_path))
        except Exception as e:
            logger.error(f"Lỗi sinh cẩm nang UI/UX PDF: {e}")

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file cẩm nang UI/UX PDF")

    return FileResponse(
        path=str(pdf_path),
        filename="HuongDan_SuDung_UIUX_ConveyorVision.pdf",
        media_type="application/pdf"
    )


@app.get("/api/download_pdf_handbook")
async def download_pdf_handbook():
    """Tải về bộ cẩm nang toàn diện toàn tập A đến Z về dự án định dạng PDF."""
    pdf_path = REPO_ROOT / "doc" / "CONVEYORVISION_TOAN_TAP_A_DEN_Z.pdf"
    if not pdf_path.exists():
        try:
            import importlib.util
            gen_script = REPO_ROOT / "scripts" / "generate_pdf_handbook.py"
            if gen_script.exists():
                spec = importlib.util.spec_from_file_location("generate_pdf_handbook", str(gen_script))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.build_pdf(str(pdf_path))
        except Exception as e:
            logger.error(f"Lỗi sinh cẩm nang toàn tập PDF: {e}")

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file cẩm nang toàn tập PDF")

    return FileResponse(
        path=str(pdf_path),
        filename="ConveyorVision_Toan_Tap_A_Den_Z.pdf",
        media_type="application/pdf"
    )


@app.get("/api/download_pdf_master")
async def download_pdf_master():
    """Tải về tài liệu kỹ thuật toàn diện và kiến trúc hệ thống ConveyorVision định dạng PDF."""
    pdf_path = REPO_ROOT / "doc" / "TAI_LIEU_TOAN_DIEN_DO_AN_CONVEYORVISION.pdf"
    if not pdf_path.exists():
        try:
            import importlib.util
            gen_script = REPO_ROOT / "scripts" / "generate_master_doc_pdf.py"
            if gen_script.exists():
                spec = importlib.util.spec_from_file_location("generate_master_doc_pdf", str(gen_script))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.build_pdf(str(pdf_path))
        except Exception as e:
            logger.error(f"Lỗi sinh tài liệu kỹ thuật toàn diện PDF: {e}")

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file tài liệu kỹ thuật toàn diện PDF")

    return FileResponse(
        path=str(pdf_path),
        filename="TaiLieu_KyThuat_ToanDien_ConveyorVision.pdf",
        media_type="application/pdf"
    )


# ==============================================================================
# 6. REST API - ĐÁNH GIÁ MÔ HÌNH (MODEL EVALUATION)
# ==============================================================================
EVAL_CHARTS = [
    ("training_progress", "results.png", "Tiến trình huấn luyện",
     "Train/Val loss và mAP qua từng epoch của lần huấn luyện đang dùng."),
    ("confusion_matrix", "test_confusion_matrix_normalized.png", "Ma trận nhầm lẫn (Test)",
     "Chuẩn hoá theo nhãn thật, tại ngưỡng conf=0.25. Hàng background = bỏ sót / báo nhầm nền."),
    ("pr_curve", "test_BoxPR_curve.png", "Đường cong Precision – Recall (Test)",
     "Đánh đổi giữa độ chính xác và độ nhạy cho từng lớp."),
    ("test_predictions", "test_batch0_pred.jpg", "Dự đoán mẫu trên tập Test",
     "Bounding box và nhãn mô hình dự đoán trên một batch ảnh thật của tập Test."),
]


@app.get("/api/evaluation")
async def get_evaluation_data():
    """Số liệu đánh giá THẬT do `evaluate.py --export doc/evaluation` sinh ra (không hardcode)."""
    metrics_file = EVAL_DIR / "metrics.json"
    if not metrics_file.exists():
        return {"available": False, "charts": []}
    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"Lỗi đọc {metrics_file}: {e}")
        return {"available": False, "charts": []}

    data["available"] = True
    data["charts"] = [
        {"id": cid, "title": title, "description": desc, "image_url": f"doc/evaluation/{filename}"}
        for cid, filename, title, desc in EVAL_CHARTS
        if (EVAL_DIR / filename).exists()
    ]
    return data


# ==============================================================================
# 7. REST API - DỮ LIỆU & MÔ HÌNH (DATASET & MODELS)
# ==============================================================================
_dataset_cache: Dict[str, Any] = {}


@app.get("/api/dataset_info")
def get_dataset_info():  # sync → chạy trong threadpool, không chặn event loop khi đọc nhiều file
    """Đếm động số lượng ảnh và bounding boxes thực tế từng lớp từ data/labels/."""
    now = time.time()
    if _dataset_cache and (now - _dataset_cache.get("timestamp", 0)) < 60:
        return _dataset_cache["data"]

    train_dir = REPO_ROOT / "data" / "images" / "train"
    val_dir = REPO_ROOT / "data" / "images" / "val"
    test_dir = REPO_ROOT / "data" / "images" / "test"

    train_c = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_c = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
    test_c = len(list(test_dir.glob("*.jpg"))) if test_dir.exists() else 0
    total_imgs = train_c + val_c + test_c
    pct = (lambda n: round(n / total_imgs * 100, 1)) if total_imgs else (lambda n: 0.0)

    # Đếm số lượng bounding boxes thực tế từng lớp từ thư mục data/labels
    box_counts = {0: 0, 1: 0}
    labels_dir = REPO_ROOT / "data" / "labels"
    if labels_dir.exists():
        for split in ["train", "val", "test"]:
            sp = labels_dir / split
            if sp.exists():
                for txt in sp.glob("*.txt"):
                    try:
                        with open(txt, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    cid = int(parts[0])
                                    box_counts[cid] = box_counts.get(cid, 0) + 1
                    except Exception:
                        pass

    total_boxes = sum(box_counts.values())
    pass_boxes = box_counts.get(0, 0)
    defect_boxes = box_counts.get(1, 0)

    pass_pct = round(pass_boxes / total_boxes * 100, 1) if total_boxes > 0 else 0.0
    defect_pct = round(defect_boxes / total_boxes * 100, 1) if total_boxes > 0 else 0.0

    split_report_file = REPO_ROOT / "data" / "split_report.json"
    split_report = None
    if split_report_file.exists():
        try:
            split_report = json.loads(split_report_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            split_report = None

    data = {
        "dataset_name": "ConveyorVision Industrial Bottles",
        "total_images": total_imgs,
        "total_boxes": total_boxes,
        "splits": {
            "train": {"count": train_c, "pct": pct(train_c)},
            "val": {"count": val_c, "pct": pct(val_c)},
            "test": {"count": test_c, "pct": pct(test_c)}
        },
        "split_report": split_report,
        "classes": [
            {
                "id": 0,
                "name": "pass (Đạt chuẩn)",
                "boxes": pass_boxes,
                "ratio_pct": pass_pct,
                "description": "Chai nguyên vẹn (gồm chai trên băng chuyền phân loại rác thải)"
            },
            {
                "id": 1,
                "name": "defect (Khuyết tật/Lỗi)",
                "boxes": defect_boxes,
                "ratio_pct": defect_pct,
                "description": "Chai bị bóp méo, móp, biến dạng"
            }
        ],
        "image_resolution": "Huấn luyện ở 640x640 (RGB)",
        "source_format": "YOLOv8/YOLO26 PyTorch Format"
    }
    _dataset_cache["timestamp"] = now
    _dataset_cache["data"] = data
    return data


def _scan_models() -> List[Dict[str, Any]]:
    """Quét và liệt kê các checkpoint trọng số sẵn có trong repository."""
    models_list = []
    seen_paths = set()

    # Chỉ models/ là nơi triển khai; runs/ là sản phẩm phụ của huấn luyện (best/last trùng lặp)
    candidate_dirs = [MODELS_DIR]
    for cdir in candidate_dirs:
        if cdir.exists():
            for pt in cdir.rglob("*.pt"):
                abs_str = str(pt.resolve())
                if abs_str not in seen_paths:
                    seen_paths.add(abs_str)
                    rel_path = pt.relative_to(REPO_ROOT).as_posix()
                    size_mb = round(pt.stat().st_size / (1024 * 1024), 2)
                    mtime = datetime.fromtimestamp(pt.stat().st_mtime).strftime("%H:%M %d/%m/%Y")
                    is_active = (pt.name == "conveyor_qc_yolo26n.pt")

                    models_list.append({
                        "name": pt.name,
                        "kind": "coco" if COCO_WEIGHTS_RE.match(pt.name) else "custom",
                        "path": rel_path,
                        "size_mb": size_mb,
                        "modified": mtime,
                        "is_active": is_active,
                        "description": "Mô hình QC Đạt/Lỗi đang dùng (YOLO26n, chia dữ liệu theo nhóm)" if is_active else "Checkpoint thử nghiệm / pilot"
                    })

    models_list.sort(key=lambda m: (not m["is_active"], m["name"]))
    return models_list


@app.get("/api/models")
async def get_available_models():
    return _scan_models()


# ==============================================================================
# 9. NGUỒN VIDEO & KIỂM TRA SẴN SÀNG HỆ THỐNG
# ==============================================================================
def _list_videos() -> List[Path]:
    if not RAW_VIDEOS_DIR.exists():
        return []
    videos = [p for p in RAW_VIDEOS_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=lambda p: p.stat().st_mtime, reverse=True)


@app.get("/api/sources")
async def get_sources():
    """Liệt kê các file video có sẵn trong raw_videos/ (mới nhất trước)."""
    return [
        {
            "name": p.name,
            "path": f"raw_videos/{p.name}",
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M %d/%m/%Y"),
        }
        for p in _list_videos()
    ]


@app.get("/api/system_check")
def system_check():
    """Kiểm tra điều kiện cần để chạy kiểm định: thư viện, GPU, trọng số, video.

    Hàm sync để FastAPI chạy trong threadpool — lần đầu import torch mất vài giây.
    """
    import importlib.util

    has_ultralytics = importlib.util.find_spec("ultralytics") is not None
    gpu_name = None
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning(f"Không thể kiểm tra GPU: {e}")

    models = _scan_models()
    return {
        "ultralytics": has_ultralytics,
        "gpu": gpu_name,
        "coco_models": [m["path"] for m in models if m["kind"] == "coco"],
        "custom_models": [m["path"] for m in models if m["kind"] == "custom"],
        "videos": len(_list_videos()),
    }
