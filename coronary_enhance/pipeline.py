"""Real-time coronary artery enhancement pipeline (classical, CPU, 16-bit).

Idea (all in the log-attenuation domain, where structures ADD instead of multiply):

    L = -log(I)                              I = 16-bit X-ray frame
    B = opening(L, SE > vessel width)        ribs / spine / lungs / heart shadow are
                                             *wider* than vessels -> they survive the
                                             opening and form the background layer B
    T = L - B                                thin bright structures only (vessels+noise)
    T = motion-adaptive temporal filter(T)   cine noise reduction, no vessel ghosting
    V = multi-scale Hessian ridge filter(T)  vesselness (tubular-structure likelihood)

    processed = CLAHE(T)                     background-suppressed image
    enhanced  = CLAHE(T * (a + (1-a) * V))   vessel-emphasised image

No training data / model file is needed, which also keeps latency deterministic.
"""
from dataclasses import dataclass, field
import time
import numpy as np
import cv2

cv2.setUseOptimized(True)


@dataclass
class PipelineConfig:
    analysis_max_side: int = 512        # frames larger than this are analysed at reduced size
    vessel_max_width: float = 10.0      # px @ analysis resolution; SE for opening = ~1.6x this
    sigmas: tuple = (1.3, 2.2, 3.2)     # ridge-filter scales (px @ analysis resolution)
    half_res_from: float = 2.0          # scales >= this are computed at 1/2 resolution (speed)
    bg_downsample: int = 2              # background (opening) computed at 1/bg_downsample
    temporal_alpha: float = 0.45        # weight of new frame in static areas (1 = no temporal filter)
    motion_thr: float = 0.05            # |dT| (log units) above which pixels are treated as moving
    vessel_floor: float = 0.12          # min gain of `enhanced` outside vessels (0..1)
    pre_smooth: float = 0.8             # px, light Gaussian before background estimation
    context: float = 0.06               # fraction of raw structure kept in `processed` for orientation
    use_clahe: bool = False             # True = OpenCV 16-bit CLAHE (accurate but ~14 ms/call)
    local_strength: float = 3.0         # >0 compresses dense-vessel regions, boosts sparse ones
    clahe_clip: float = 2.0
    clahe_tiles: int = 8
    vessels_dark: bool = True           # output convention: contrast-filled vessels appear dark
    profile: bool = False               # collect per-stage timings in result["timings"]


class CoronaryPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        c = self.cfg
        k = int(round(c.vessel_max_width * 1.6 / c.bg_downsample)) | 1   # odd
        self._se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._clahe = cv2.createCLAHE(clipLimit=c.clahe_clip,
                                      tileGridSize=(c.clahe_tiles, c.clahe_tiles))
        self._kernels = []
        for s in c.sigmas:
            half = s >= c.half_res_from
            self._kernels.append((half, self._make_kernels(s / 2.0 if half else s)))
        self._prev_T = None
        self._scale_T = None
        self._scale_V = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _make_kernels(sigma):
        r = int(np.ceil(3 * sigma))
        x = np.arange(-r, r + 1, dtype=np.float32)
        g = np.exp(-x ** 2 / (2 * sigma ** 2))
        g /= g.sum()
        g1 = -x / sigma ** 2 * g
        g2 = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
        # sepFilter2D(kernelX, kernelY)
        return dict(sigma=sigma,
                    xx=(g2.reshape(1, -1), g.reshape(-1, 1)),
                    yy=(g.reshape(1, -1), g2.reshape(-1, 1)),
                    xy=(g1.reshape(1, -1), g1.reshape(-1, 1)))

    def reset(self):
        """Call at the start of every new cine run."""
        self._prev_T = self._scale_T = self._scale_V = None

    @staticmethod
    def _ema(old, new, a=0.3):
        return new if old is None else (1 - a) * old + a * new

    def _finish(self, a):
        """float [0,1] -> uint16 with adaptive local contrast (fast CLAHE substitute).

        OpenCV's 16-bit CLAHE needs ~14 ms/frame per call (65536-bin histograms), which would
        eat 80 % of the budget, so contrast is equalised in float with a smooth local gain map:
        dense-vessel regions are gently compressed, sparse regions are boosted.
        Set cfg.use_clahe=True to use true CLAHE instead (slower).
        """
        c = self.cfg
        if c.use_clahe:
            a16 = self._clahe.apply((a * 65535.0).astype(np.uint16))
        else:
            h, w = a.shape
            m = cv2.resize(a, (max(w // 16, 4), max(h // 16, 4)), interpolation=cv2.INTER_AREA)
            m = cv2.GaussianBlur(m, (0, 0), 1.5)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            m *= c.local_strength
            m += 0.6
            gain = cv2.divide(np.float32(1.0), m)
            out = np.clip(a * gain, 0.0, 1.0)
            np.sqrt(out, out=out)                        # gamma 0.5: lifts faint distal vessels
            a16 = (out * 65535.0).astype(np.uint16)
        if c.vessels_dark:
            a16 = 65535 - a16
        return a16

    # ------------------------------------------------------------------ main
    def process(self, frame: np.ndarray) -> dict:
        """frame: uint16 (H, W) -> dict(raw, processed, enhanced, vesselness, mask?)."""
        c = self.cfg
        tm = {}
        t = time.perf_counter()

        def tick(name):
            nonlocal t
            if c.profile:
                n = time.perf_counter()
                tm[name] = (n - t) * 1e3
                t = n

        h0, w0 = frame.shape
        s = max(h0, w0) / c.analysis_max_side
        if s > 1.0:
            x = cv2.resize(frame, (int(round(w0 / s)), int(round(h0 / s))),
                           interpolation=cv2.INTER_AREA)
        else:
            x = frame
        xf = x.astype(np.float32)
        tick("resize+convert")

        # 1) log-attenuation domain: vessels & bones become additive bright structures
        np.maximum(xf, 64.0, out=xf)                      # avoid log(0)
        L = cv2.log(xf)
        np.negative(L, out=L)                             # L = -log(I)
        if c.pre_smooth > 0:
            L = cv2.GaussianBlur(L, (0, 0), c.pre_smooth)
        tick("log+smooth")

        # 2) background estimate at reduced resolution (opening), then subtract
        d = c.bg_downsample
        Ls = cv2.resize(L, (L.shape[1] // d, L.shape[0] // d), interpolation=cv2.INTER_AREA) if d > 1 else L
        B = cv2.dilate(cv2.erode(Ls, self._se), self._se)
        if d > 1:
            B = cv2.resize(B, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
        B = cv2.GaussianBlur(B, (0, 0), 2.0)              # remove blockiness of the up-sampled background
        T = L - B
        T -= float(np.median(T[::6, ::6]))                # remove the small positive noise bias of opening
        np.maximum(T, 0.0, out=T)
        tick("background_subtract")

        # 3) motion-adaptive recursive temporal filter (cine noise suppression)
        P = self._prev_T
        if P is not None and P.shape == T.shape and c.temporal_alpha < 1.0:
            w = cv2.absdiff(T, P)
            w *= (1.0 - c.temporal_alpha) / c.motion_thr
            w += c.temporal_alpha
            np.minimum(w, 1.0, out=w)                     # 1 = trust new frame (motion)
            T = P + w * (T - P)
        self._prev_T = T
        tick("temporal")

        # 4) multi-scale Hessian ridge filter (vesselness) on the background-free layer
        V = None
        T2 = None
        for half, kn in self._kernels:
            if half:
                if T2 is None:
                    T2 = cv2.resize(T, (T.shape[1] // 2, T.shape[0] // 2), interpolation=cv2.INTER_AREA)
                src, sg = T2, kn["sigma"] * 2.0
            else:
                src, sg = T, kn["sigma"]
            Ixx = cv2.sepFilter2D(src, cv2.CV_32F, *kn["xx"])
            Iyy = cv2.sepFilter2D(src, cv2.CV_32F, *kn["yy"])
            Ixy = cv2.sepFilter2D(src, cv2.CV_32F, *kn["xy"])
            hs = cv2.add(Ixx, Iyy)
            hs *= 0.5
            df = cv2.subtract(Ixx, Iyy)
            df *= 0.5
            disc = cv2.magnitude(df, Ixy)
            lam_s = hs - disc                             # strongly negative across a bright ridge
            lam_b = hs + disc                             # ~0 along the ridge, ~lam_s for blobs
            ridge = np.maximum(-lam_s, 0.0)
            lin = cv2.divide(np.abs(lam_b), ridge + 1e-6)         # 0 = line, >=1 = blob
            np.subtract(1.0, lin, out=lin)
            np.clip(lin, 0.0, 1.0, out=lin)
            ridge *= lin
            ridge *= sg * sg                              # scale normalisation
            if half:
                ridge = cv2.resize(ridge, (T.shape[1], T.shape[0]), interpolation=cv2.INTER_LINEAR)
            V = ridge if V is None else np.maximum(V, ridge)
        tick("vesselness")

        # 5) robust, flicker-free normalisation (sub-sampled percentile + EMA)
        sub_T = T[::6, ::6]
        sub_V = V[::6, ::6]
        self._scale_T = self._ema(self._scale_T, float(np.percentile(sub_T, 99.7)) + 1e-6)
        self._scale_V = self._ema(self._scale_V, float(np.percentile(sub_V, 99.5)) + 1e-9)
        Tn = np.clip(T * (1.0 / self._scale_T), 0.0, 1.0)
        Vn = np.clip(V * (1.0 / self._scale_V), 0.0, 1.0)
        tick("normalise")

        # 6) compose outputs
        # processed: background-suppressed, with a faint trace of anatomy for orientation
        if c.context > 0:
            bs = B[::6, ::6]
            bmin, bmax = float(bs.min()), float(bs.max())
            proc = np.clip(Tn + c.context * (B - bmin) * (1.0 / max(bmax - bmin, 1e-6)), 0.0, 1.0)
        else:
            proc = Tn
        enh = Tn * (c.vessel_floor + (1.0 - c.vessel_floor) * Vn)
        enh *= 1.8
        np.clip(enh, 0.0, 1.0, out=enh)

        processed, enhanced = self._finish(proc), self._finish(enh)
        vess16 = (Vn * 65535.0).astype(np.uint16)
        if c.vessels_dark:
            vess16 = 65535 - vess16
        tick("compose+clahe")

        if s > 1.0:                                       # back to the native frame size
            up = lambda a: cv2.resize(a, (w0, h0), interpolation=cv2.INTER_LINEAR)
            processed, enhanced, vess16 = up(processed), up(enhanced), up(vess16)
            tick("upsample")

        out = dict(raw=frame, processed=processed, enhanced=enhanced, vesselness=vess16)
        if c.profile:
            out["timings"] = tm
        return out
