import time
import numpy as np
from coronary_enhance import CoronaryPipeline, PipelineConfig
from coronary_enhance.synth import SyntheticCine
from coronary_enhance.metrics import auc


def test_16bit_in_16bit_out():
    syn = SyntheticCine(512, seed=1)
    raw, _ = syn.frame(0)
    assert raw.dtype == np.uint16
    out = CoronaryPipeline().process(raw)
    for k in ("processed", "enhanced", "vesselness"):
        assert out[k].dtype == np.uint16 and out[k].shape == raw.shape
    assert out["enhanced"].max() > 256          # genuinely uses the 16-bit range


def test_vessels_are_detected():
    syn = SyntheticCine(512, seed=2)
    pipe = CoronaryPipeline()
    for i in range(10):
        raw, gt = syn.frame(i)
        out = pipe.process(raw)
    assert auc(65535.0 - out["enhanced"].astype(np.float32), gt) > 0.93


def test_latency_budget():
    syn = SyntheticCine(512, seed=3)
    frames = [syn.frame(i)[0] for i in range(40)]
    pipe = CoronaryPipeline()
    for f in frames[:10]:
        pipe.process(f)
    lat = []
    for f in frames[10:]:
        t = time.perf_counter(); pipe.process(f); lat.append((time.perf_counter() - t) * 1e3)
    assert max(lat) <= 36.0, f"max latency {max(lat):.1f} ms"


def test_non_square_and_large_input():
    rng = np.random.default_rng(0)
    f = (rng.random((768, 1024)) * 60000 + 2000).astype(np.uint16)
    out = CoronaryPipeline().process(f)
    assert out["enhanced"].shape == f.shape
