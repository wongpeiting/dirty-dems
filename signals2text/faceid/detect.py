"""Stage 3 core: detect + embed every face in a frame. NO naming here."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import config


@dataclass
class Face:
    embedding: np.ndarray
    box: tuple
    face_px: int
    face_area_pct: float
    det_score: float


@lru_cache(maxsize=1)
def get_model():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def model_available() -> bool:
    try:
        get_model()
        return True
    except Exception:
        return False


def detect_faces(frame_bgr: np.ndarray, min_face_px: int = config.MIN_FACE_PX) -> list[Face]:
    app = get_model()
    H, W = frame_bgr.shape[:2]
    out = []
    for f in app.get(frame_bgr):
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if h < min_face_px:
            continue
        emb = np.asarray(f.normed_embedding, dtype=np.float32)  # already L2-normed
        out.append(
            Face(
                embedding=emb,
                box=(x1, y1, w, h),
                face_px=h,
                face_area_pct=min(1.0, (w * h) / float(W * H)),
                det_score=float(f.det_score),
            )
        )
    return out
