"""Latency benchmark: input frame received -> final processed frame produced.

Usage:  python scripts/benchmark.py                 # synthetic 512x512 and 1024x1024, 300 frames each
        python scripts/benchmark.py --input my.dcm  # your own cine (DICOM / image folder)
        python scripts/benchmark.py --frames 1000 --threads 4
"""
import argparse, gc, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coronary_enhance import CoronaryPipeline, PipelineConfig          # noqa: E402
from coronary_enhance.synth import SyntheticCine                        # noqa: E402
from coronary_enhance.io import load_cine                               # noqa: E402

LIMIT_MS = 36.0


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware():
    return dict(cpu=cpu_name(), logical_cores=os.cpu_count(), os=platform.platform(),
                python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                cv2_threads=cv2.getNumThreads(), gpu="none (CPU only)")


def bench(frames, warmup=30, cfg=None):
    pipe = CoronaryPipeline(cfg or PipelineConfig())
    for f in frames[:warmup]:
        pipe.process(f)                                  # warm-up: allocations, caches
    pipe.reset()
    lat = []
    gc.collect()
    for f in frames:
        t0 = time.perf_counter()                         # frame received by the pipeline ...
        pipe.process(f)
        lat.append((time.perf_counter() - t0) * 1e3)     # ... final processed frame produced
    lat = np.array(lat)
    return dict(frames=len(lat), avg_ms=float(lat.mean()), min_ms=float(lat.min()),
                max_ms=float(lat.max()), p95_ms=float(np.percentile(lat, 95)),
                p99_ms=float(np.percentile(lat, 99)), fps=float(1000.0 / lat.mean()),
                passed=bool(lat.max() <= LIMIT_MS), frames_over_limit=int((lat > LIMIT_MS).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="DICOM / folder of frames (default: synthetic)")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--threads", type=int, default=None, help="OpenCV threads (default: all cores)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.threads:
        cv2.setNumThreads(a.threads)

    sets = {}
    if a.input:
        fr = list(load_cine(a.input))
        fr = (fr * (a.frames // len(fr) + 1))[: max(a.frames, 60)]
        sets[f"{fr[0].shape[1]}x{fr[0].shape[0]} (user data)"] = fr
    else:
        for size in (512, 1024):
            syn = SyntheticCine(size, seed=7)
            sets[f"{size}x{size} 16-bit synthetic"] = [syn.frame(i)[0] for i in range(a.frames + 30)]

    hw = hardware()
    res = {}
    print("Hardware:", json.dumps(hw, indent=2))
    for name, frames in sets.items():
        r = bench(frames)
        res[name] = r
        print(f"\n[{name}]  avg {r['avg_ms']:.2f} ms | min {r['min_ms']:.2f} | max {r['max_ms']:.2f} | "
              f"p95 {r['p95_ms']:.2f} | p99 {r['p99_ms']:.2f} | {r['fps']:.0f} FPS | "
              f"{'PASS' if r['passed'] else 'FAIL'} (limit {LIMIT_MS} ms, {r['frames_over_limit']} frames over)")

    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(dict(hardware=hw, limit_ms=LIMIT_MS, results=res), indent=2))
    md = ["| Input | Avg (ms) | Min (ms) | Max (ms) | P95 (ms) | FPS | <= 36 ms |", "|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        md.append(f"| {n} | {r['avg_ms']:.2f} | {r['min_ms']:.2f} | {r['max_ms']:.2f} | {r['p95_ms']:.2f} | "
                  f"{r['fps']:.0f} | {'YES' if r['passed'] else 'NO'} |")
    (out / "benchmark.md").write_text("\n".join(md) + "\n\nHardware: " + json.dumps(hw) + "\n")
    print("\nSaved results/benchmark.json and results/benchmark.md")


if __name__ == "__main__":
    main()
