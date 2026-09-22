"""16-bit grayscale I/O helpers (PNG/TIFF/NPY/DICOM)."""
from pathlib import Path
import numpy as np
import cv2


def to_uint16(img: np.ndarray) -> np.ndarray:
    """Convert any grayscale array to uint16, scaling the full range."""
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype == np.uint16:
        return img
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 257
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return ((img - lo) / max(hi - lo, 1e-6) * 65535).astype(np.uint16)


def load_frame(path) -> np.ndarray:
    """Load one 16-bit grayscale frame (uint16 HxW)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npy":
        return to_uint16(np.load(p))
    if ext in (".dcm", ".dicom"):
        import pydicom
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array
        if arr.ndim == 3:            # cine -> first frame; use load_cine for all
            arr = arr[0]
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        return to_uint16(arr)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    return to_uint16(img)


def load_cine(path):
    """Yield frames from a multi-frame DICOM or a folder of images."""
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.glob("*")):
            if f.suffix.lower() in (".png", ".tif", ".tiff", ".npy"):
                yield load_frame(f)
    elif p.suffix.lower() in (".dcm", ".dicom"):
        import pydicom
        arr = pydicom.dcmread(str(p)).pixel_array
        for fr in (arr if arr.ndim == 3 else arr[None]):
            yield to_uint16(fr)
    else:
        yield load_frame(p)
