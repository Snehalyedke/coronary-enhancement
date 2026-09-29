"""Image + video front-ends for the enhancement pipeline.

    process_image(src, out_dir)        image  -> original.png, enhanced.png (8-bit, for viewing)
                                                 + enhanced_16bit.png (true 16-bit)
    process_video(src, dst, mode=...)  video / DICOM cine -> H.264 MP4 (video output only)

The pipeline itself (pipeline.py) is untouched: video is simply "call pipe.process() on every
frame in order" - the recursive temporal filter inside the pipeline is what makes it cine-aware.
Video codecs are 8-bit, so the MP4 is an 8-bit rendering; use process_image / the 16-bit PNG
for full 16-bit output.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .io import load_frame, to_uint16
from .pipeline import CoronaryPipeline, PipelineConfig

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".dcm", ".dicom"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".dcm", ".dicom"}
VIDEO_MODES = ("enhanced", "processed", "compare")     # compare = Original | Enhanced


# --------------------------------------------------------------------------- display helpers
class DisplayWindow:
    """16-bit -> 8-bit percentile window/level, smoothed over time so video does not flicker."""

    def __init__(self, smooth: float = 0.15):
        self.smooth, self.lo, self.hi = smooth, None, None

    def __call__(self, img16: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(img16[::4, ::4], (0.5, 99.5))
        if self.lo is None:
            self.lo, self.hi = float(lo), float(hi)
        else:
            self.lo += self.smooth * (float(lo) - self.lo)
            self.hi += self.smooth * (float(hi) - self.hi)
        x = (img16.astype(np.float32) - self.lo) * (255.0 / max(self.hi - self.lo, 1.0))
        return np.clip(x, 0, 255).astype(np.uint8)


def _label(img8: np.ndarray, text: str) -> np.ndarray:
    img8 = img8.copy()
    cv2.rectangle(img8, (0, 0), (img8.shape[1], 26), 0, -1)
    cv2.putText(img8, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)
    return img8


def _compose(out: dict, mode: str, wins: dict) -> np.ndarray:
    if mode == "processed":
        return wins["processed"](out["processed"])
    if mode == "compare":
        raw = _label(wins["raw"](out["raw"]), "Original")
        enh = _label(wins["enhanced"](out["enhanced"]), "Enhanced coronary arteries")
        return np.hstack([raw, enh])
    return wins["enhanced"](out["enhanced"])


# --------------------------------------------------------------------------- images
def process_image(src, out_dir, cfg: PipelineConfig | None = None) -> dict:
    """Enhance one image (PNG/JPG/TIFF/NPY/DICOM). Returns file names (inside out_dir) + timing."""
    frame = load_frame(src)                                  # uint16 HxW
    pipe = CoronaryPipeline(cfg or PipelineConfig())
    t0 = time.perf_counter()
    out = pipe.process(frame)
    ms = (time.perf_counter() - t0) * 1e3

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "original.png"), DisplayWindow()(out["raw"]))
    cv2.imwrite(str(out_dir / "enhanced.png"), DisplayWindow()(out["enhanced"]))
    cv2.imwrite(str(out_dir / "enhanced_16bit.png"), out["enhanced"])      # true 16-bit PNG
    h, w = frame.shape
    return dict(original="original.png", enhanced="enhanced.png", enhanced16="enhanced_16bit.png",
                width=w, height=h, ms=round(ms, 1))


# --------------------------------------------------------------------------- video input
def open_video(path):
    """Return (frame_iterator, fps, total_frames_or_0). Frames are uint16 HxW.

    Accepts anything OpenCV can decode (mp4/avi/mov/mkv/webm) and multi-frame DICOM cine.
    """
    p = Path(path)
    if p.suffix.lower() in (".dcm", ".dicom"):
        import pydicom
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array
        n = int(getattr(ds, "NumberOfFrames", 1) or 1)
        frames = arr if n > 1 else arr[None]
        mono1 = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1"
        fps = 15.0
        if getattr(ds, "FrameTime", None):
            fps = 1000.0 / float(ds.FrameTime)
        elif getattr(ds, "CineRate", None):
            fps = float(ds.CineRate)
        elif getattr(ds, "RecommendedDisplayFrameRate", None):
            fps = float(ds.RecommendedDisplayFrameRate)

        def gen_dicom():
            for fr in frames:
                yield to_uint16(fr.max() - fr if mono1 else fr)
        return gen_dicom(), min(max(fps, 1.0), 120.0), len(frames)

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError("Could not open this video file (unsupported codec or corrupt file).")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:
        fps = 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def gen_cv():
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield to_uint16(f)
        finally:
            cap.release()
    return gen_cv(), float(fps), max(total, 0)


# --------------------------------------------------------------------------- video output
def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class Mp4Writer:
    """Writes 8-bit grayscale frames to an H.264 MP4 that plays in every browser.

    Uses ffmpeg (system binary or the one bundled by `imageio-ffmpeg`). If no ffmpeg is
    available it falls back to OpenCV's writer (then `playable_in_browser` may be False).
    """

    def __init__(self, path, fps: float, size: tuple[int, int], crf: int = 18):
        self.path, (self.w, self.h) = str(path), size
        self.proc = self.cv = self._err = None
        exe = _ffmpeg_exe()
        if exe:
            self._err = tempfile.TemporaryFile()
            cmd = [exe, "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{self.w}x{self.h}",
                   "-r", f"{fps:.3f}", "-i", "-",
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",          # H.264 needs even sizes
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=self._err)
            self.playable_in_browser = True
        else:
            for tag in ("avc1", "mp4v"):
                vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*tag), fps, (self.w, self.h), False)
                if vw.isOpened():
                    self.cv, self.playable_in_browser = vw, tag == "avc1"
                    break
            if self.cv is None:
                raise RuntimeError("No video encoder available. Install ffmpeg or `pip install imageio-ffmpeg`.")

    def _ffmpeg_error(self) -> str:
        self._err.seek(0)
        return self._err.read().decode(errors="replace").strip()[-500:]

    def write(self, frame8: np.ndarray):
        if frame8.shape != (self.h, self.w):
            raise ValueError("frame size changed during the video")
        if self.proc:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame8).tobytes())
            except (BrokenPipeError, OSError):
                raise RuntimeError("ffmpeg stopped unexpectedly: " + self._ffmpeg_error())
        else:
            self.cv.write(frame8)

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            rc = self.proc.wait()
            if rc != 0:
                raise RuntimeError("ffmpeg failed: " + self._ffmpeg_error())
            self._err.close()
        elif self.cv:
            self.cv.release()


def process_video(src, dst, cfg: PipelineConfig | None = None, mode: str = "enhanced",
                  progress=None, max_frames: int | None = None) -> dict:
    """Enhance every frame of a video / DICOM cine and write an MP4 to `dst`.

    mode: "enhanced" (default) | "processed" | "compare" (Original | Enhanced side by side)
    progress(done, total) is called after every frame (total is 0 if unknown).
    """
    if mode not in VIDEO_MODES:
        raise ValueError(f"mode must be one of {VIDEO_MODES}")
    frames, fps, total = open_video(src)
    pipe = CoronaryPipeline(cfg or PipelineConfig())         # fresh temporal state per video
    wins = {k: DisplayWindow() for k in ("raw", "processed", "enhanced")}
    writer, n, spent = None, 0, 0.0
    try:
        for f in frames:
            if max_frames and n >= max_frames:
                break
            t0 = time.perf_counter()
            out = pipe.process(f)
            spent += time.perf_counter() - t0
            img = _compose(out, mode, wins)
            if writer is None:
                writer = Mp4Writer(dst, fps, (img.shape[1], img.shape[0]))
            writer.write(img)
            n += 1
            if progress:
                progress(n, total)
    finally:
        frames.close()
        if writer:
            writer.close()
    if n == 0:
        raise ValueError("No frames could be read from this file.")
    return dict(frames=n, fps=round(fps, 2), mean_ms=round(spent / n * 1e3, 2),
                playable_in_browser=writer.playable_in_browser)
