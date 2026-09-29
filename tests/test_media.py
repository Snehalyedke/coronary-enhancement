import cv2
import numpy as np
import pytest

from coronary_enhance import process_image, process_video
from coronary_enhance.io import to_uint16
from coronary_enhance.synth import SyntheticCine


def _make_video(path, n=12, size=256):
    syn = SyntheticCine(size, seed=4)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size), False)
    for i in range(n):
        vw.write((syn.frame(i)[0] >> 8).astype(np.uint8))
    vw.release()


def _count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n, shape = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n, shape = n + 1, f.shape
    cap.release()
    return n, shape


@pytest.mark.parametrize("mode,width_mult", [("enhanced", 1), ("compare", 2)])
def test_video_in_video_out(tmp_path, mode, width_mult):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_video(src)
    info = process_video(src, dst, mode=mode)
    assert info["frames"] == 12
    n, shape = _count_frames(dst)
    assert n == 12 and shape[1] == 256 * width_mult


def test_video_progress_and_max_frames(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_video(src)
    calls = []
    info = process_video(src, dst, progress=lambda d, t: calls.append((d, t)), max_frames=5)
    assert info["frames"] == 5 and calls[-1][0] == 5


def test_bad_video_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        process_video(bad, tmp_path / "out.mp4")


def test_image_outputs(tmp_path):
    raw, _ = SyntheticCine(256, seed=1).frame(0)
    src = tmp_path / "x.png"
    cv2.imwrite(str(src), raw)
    r = process_image(src, tmp_path / "res")
    e16 = cv2.imread(str(tmp_path / "res" / r["enhanced16"]), cv2.IMREAD_UNCHANGED)
    e8 = cv2.imread(str(tmp_path / "res" / r["enhanced"]), cv2.IMREAD_UNCHANGED)
    assert e16.dtype == np.uint16 and e8.dtype == np.uint8 and e16.shape == raw.shape


def test_colour_input_is_converted_to_gray():
    bgr = np.zeros((32, 32, 3), np.uint8)
    bgr[..., 1] = 100
    out = to_uint16(bgr)
    assert out.shape == (32, 32) and out.dtype == np.uint16
