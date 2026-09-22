"""Quality evaluation on the synthetic set (ground-truth vessel masks available)."""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coronary_enhance import CoronaryPipeline, PipelineConfig          # noqa: E402
from coronary_enhance.synth import SyntheticCine                        # noqa: E402
from coronary_enhance.metrics import auc, best_dice, cnr, background_clutter   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    rows = {k: [] for k in ("raw", "processed", "enhanced", "vesselness")}
    for seed in range(a.seeds):
        syn = SyntheticCine(512, seed=seed)
        pipe = CoronaryPipeline(PipelineConfig())
        for i in range(a.frames):
            raw, gt = syn.frame(i)
            out = pipe.process(raw)
            if i < 8:                       # let the temporal filter settle
                continue
            for k in rows:
                img = out[k].astype(np.float32)
                score = 65535.0 - img       # vessels are dark in every image
                rows[k].append((auc(score, gt), best_dice(score, gt), cnr(img, gt),
                                background_clutter(img, gt)))
    res = {}
    print(f"{'image':<12}{'AUC':>8}{'best Dice':>11}{'CNR':>8}{'clutter (lower=better)':>26}")
    for k, v in rows.items():
        m = np.nanmean(np.array(v), axis=0)
        res[k] = dict(auc=m[0], best_dice=m[1], cnr=m[2], clutter=m[3])
        print(f"{k:<12}{m[0]:>8.3f}{m[1]:>11.3f}{m[2]:>8.2f}{m[3]:>26.3f}")
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "quality.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
