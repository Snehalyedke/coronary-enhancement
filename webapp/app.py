"""Web UI: upload an image or a video, see the enhanced result in the browser.

Run:   python webapp/app.py            then open http://127.0.0.1:5000

Endpoints
  GET  /                    the UI
  POST /api/image           multipart: file, vessels_dark   -> JSON with result image URLs (synchronous)
  POST /api/video           multipart: file, mode, vessels_dark -> {"job_id": ...}   (background job)
  GET  /api/jobs/<job_id>   -> {"status": queued|processing|done|error, "done": n, "total": n, ...}
  GET  /files/<run>/<name>  result files (image PNGs / MP4, supports seeking)
"""
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coronary_enhance import PipelineConfig                                     # noqa: E402
from coronary_enhance.media import (IMAGE_EXTS, VIDEO_EXTS, VIDEO_MODES,        # noqa: E402
                                    process_image, process_video)

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
RUNS.mkdir(exist_ok=True)
RETENTION_S = 60 * 60                                    # delete uploads/results after 1 hour

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 1024)) * 1024 * 1024

_jobs, _lock = {}, threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)            # videos are processed one at a time


# ------------------------------------------------------------------ helpers
def _cleanup():
    now = time.time()
    for d in RUNS.iterdir():
        try:
            with _lock:
                busy = _jobs.get(d.name, {}).get("status") in ("queued", "processing")
            if d.is_dir() and not busy and now - d.stat().st_mtime > RETENTION_S:
                shutil.rmtree(d, ignore_errors=True)
                with _lock:
                    _jobs.pop(d.name, None)
        except OSError:
            pass


def _receive(allowed):
    """Validate + save the uploaded file. Returns (run_id, run_dir, src_path) or raises ValueError."""
    f = request.files.get("file")
    if f is None or not f.filename:
        raise ValueError("No file received.")
    ext = Path(f.filename).suffix.lower()
    if ext not in allowed:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}")
    _cleanup()
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS / run_id
    run_dir.mkdir()
    src = run_dir / f"input{ext}"
    f.save(src)
    return run_id, run_dir, src


def _config():
    return PipelineConfig(vessels_dark=request.form.get("vessels_dark", "1") == "1")


def _url(run_id, name):
    return f"/files/{run_id}/{name}"


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return render_template("index.html", image_exts=sorted(IMAGE_EXTS), video_exts=sorted(VIDEO_EXTS))


@app.post("/api/image")
def api_image():
    try:
        run_id, run_dir, src = _receive(IMAGE_EXTS)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    try:
        r = process_image(src, run_dir, _config())
    except Exception as e:                                # unreadable / corrupt image
        return jsonify(error=f"Could not process this image: {e}"), 422
    return jsonify(original=_url(run_id, r["original"]), enhanced=_url(run_id, r["enhanced"]),
                   enhanced16=_url(run_id, r["enhanced16"]), width=r["width"], height=r["height"],
                   ms=r["ms"])


@app.post("/api/video")
def api_video():
    try:
        run_id, run_dir, src = _receive(VIDEO_EXTS)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    mode = request.form.get("mode", "enhanced")
    if mode not in VIDEO_MODES:
        mode = "enhanced"
    cfg = _config()
    with _lock:
        _jobs[run_id] = dict(status="queued", done=0, total=0)

    def progress(done, total):
        with _lock:
            _jobs[run_id].update(status="processing", done=done, total=total)

    def work():
        try:
            info = process_video(src, run_dir / "enhanced.mp4", cfg, mode, progress)
            with _lock:
                _jobs[run_id].update(status="done", video=_url(run_id, "enhanced.mp4"), **info)
        except Exception as e:
            with _lock:
                _jobs[run_id].update(status="error", error=str(e))

    _executor.submit(work)
    return jsonify(job_id=run_id), 202


@app.get("/api/jobs/<job_id>")
def api_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify(error="Unknown or expired job."), 404
        return jsonify(job)


@app.get("/files/<run_id>/<path:name>")
def files(run_id, name):
    if not run_id.isalnum():
        abort(404)
    return send_from_directory(RUNS / run_id, name, conditional=True)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            debug=False, threaded=True)
