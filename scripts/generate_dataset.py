"""Write a synthetic 16-bit cine dataset (+ ground-truth vessel masks) to data/synthetic."""
import argparse, sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coronary_enhance.synth import SyntheticCine   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sequences", type=int, default=2)
ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--out", default="data/synthetic")
a = ap.parse_args()

for s in range(a.sequences):
    syn = SyntheticCine(a.size, seed=s + 10)
    d = Path(a.out) / f"seq{s:02d}"
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(exist_ok=True)
    for i in range(a.frames):
        raw, gt = syn.frame(i)
        cv2.imwrite(str(d / "frames" / f"{i:03d}.png"), raw)                    # uint16 PNG
        cv2.imwrite(str(d / "masks" / f"{i:03d}.png"), gt.astype("uint8") * 255)
print("done ->", a.out)
