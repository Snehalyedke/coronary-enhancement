import io
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("flask")
from coronary_enhance.synth import SyntheticCine          # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))
    import app as webapp
    monkeypatch.setattr(webapp, "RUNS", tmp_path)
    return webapp.app.test_client()


def test_index(client):
    assert client.get("/").status_code == 200


def test_image_upload(client):
    raw, _ = SyntheticCine(256, seed=1).frame(0)
    ok, buf = cv2.imencode(".png", raw)
    r = client.post("/api/image", data={"file": (io.BytesIO(buf.tobytes()), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert client.get(d["enhanced"]).status_code == 200
    assert client.get(d["enhanced16"]).status_code == 200


def test_rejects_wrong_type(client):
    r = client.post("/api/image", data={"file": (io.BytesIO(b"x"), "a.exe")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_video_upload_and_poll(client, tmp_path):
    syn = SyntheticCine(256, seed=4)
    p = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (256, 256), False)
    for i in range(8):
        vw.write((syn.frame(i)[0] >> 8).astype(np.uint8))
    vw.release()
    r = client.post("/api/video", data={"file": (io.BytesIO(p.read_bytes()), "v.mp4"), "mode": "compare"},
                    content_type="multipart/form-data")
    assert r.status_code == 202
    jid = r.get_json()["job_id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").get_json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    assert client.get(j["video"]).status_code == 200
