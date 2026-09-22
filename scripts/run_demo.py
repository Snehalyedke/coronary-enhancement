"""Process one image / cine and save the Raw | Processed | Enhanced comparison.

Usage:
  python scripts/run_demo.py --synthetic                     # synthetic demo
  python scripts/run_demo.py --input frame.png               # 16-bit PNG/TIFF/NPY/DICOM
  python scripts/run_demo.py --input cine_folder/ --video    # whole cine + comparison MP4
  python scripts/run_demo.py --input x.dcm --show            # live window (press q to quit)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coronary_enhance import CoronaryPipeline, PipelineConfig          # noqa: E402
from coronary_enhance.synth import SyntheticCine                        # noqa: E402
from coronary_enhance.io import load_cine                               # noqa: E402


def window8(img16, invert_ref=False):
    """16-bit -> 8-bit for display only (percentile window/level). Processing stays 16-bit."""
    lo, hi = np.percentile(img16[::4, ::4], (0.5, 99.5))
    return np.clip((img16.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def panel(out):
    tiles = []
    for key, title in (("raw", "Original / Raw"), ("processed", "Processed (background suppressed)"),
                       ("enhanced", "Enhanced coronary arteries")):
        t = cv2.cvtColor(window8(out[key]), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(t, (0, 0), (t.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(t, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    return np.hstack(tiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--out", default="results")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--clahe", action="store_true", help="use OpenCV 16-bit CLAHE (slower)")
    a = ap.parse_args()

    if a.synthetic or not a.input:
        syn = SyntheticCine(512, seed=5)
        frames = (syn.frame(i)[0] for i in range(a.frames))
    else:
        frames = load_cine(a.input)

    out_dir = Path(a.out); out_dir.mkdir(exist_ok=True)
    pipe = CoronaryPipeline(PipelineConfig(use_clahe=a.clahe))
    writer, last, n, tot = None, None, 0, 0.0
    for f in frames:
        t0 = time.perf_counter()
        last = pipe.process(f)
        tot += time.perf_counter() - t0
        n += 1
        pn = panel(last)
        if a.video:
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         15, (pn.shape[1], pn.shape[0]))
            writer.write(pn)
        if a.show:
            cv2.imshow("raw | processed | enhanced", pn)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer:
        writer.release()
    if last is None:
        sys.exit("no frames found")

    cv2.imwrite(str(out_dir / "comparison.png"), panel(last))
    for k in ("processed", "enhanced", "vesselness"):
        cv2.imwrite(str(out_dir / f"{k}_16bit.png"), last[k])            # true 16-bit PNGs
    print(f"{n} frames, mean pipeline latency {tot / n * 1e3:.2f} ms/frame -> saved to {out_dir}/")


if __name__ == "__main__":
    main()
