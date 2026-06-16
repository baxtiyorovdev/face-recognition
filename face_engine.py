# =============================================================
#  face_engine.py — InsightFace (SCRFD + ArcFace)
# =============================================================
"""
Стек:
  Детекция:      SCRFD-500MF  (ONNX, ~5 мс на CPU)
  Эмбеддинги:   ArcFace R18  (ONNX, ~8 мс на CPU)
  Антиспуфинг:  Laplacian variance (текстура кожи)
  Трекинг:      IoU-трекер — стабилизирует ID между кадрами
  База лиц:     numpy cosine-distance

Сравнение с dlib HOG + face_recognition:
  dlib HOG detect:   ~150-400 мс / кадр
  InsightFace SCRFD: ~5-15 мс / кадр   (в 20-30 раз быстрее)
  dlib encode:       ~50-100 мс / лицо
  ArcFace R18:       ~8-15 мс / лицо   (в 5-10 раз быстрее)

Установка (один раз, модели скачиваются автоматически ~300 МБ):
  pip install insightface onnxruntime

Если нет интернета — скачайте buffalo_sc вручную:
  https://github.com/deepinsight/insightface/releases
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger("FaceEngine")

# ── Lazy import InsightFace ───────────────────────────────────
_app = None   # insightface.app.FaceAnalysis — инициализируется один раз


def _get_app():
    global _app
    if _app is not None:
        return _app

    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        raise RuntimeError(
            "InsightFace не установлен.\n"
            "Выполните: pip install insightface onnxruntime"
        )

    log.info("Загрузка InsightFace (buffalo_sc)...")
    _app = FaceAnalysis(
        name="buffalo_sc",           # SCRFD-500MF + ArcFace R18 — лёгкий+быстрый
        providers=_onnx_providers(), # CPU или CUDA автоматически
    )
    # det_size: размер входа детектора — 320×320 достаточно для домофона
    _app.prepare(ctx_id=0, det_size=(320, 320))
    log.info("InsightFace готов (providers=%s)", _onnx_providers())
    return _app


def _onnx_providers() -> list[str]:
    """Возвращает список ONNX providers: CUDA если есть, иначе CPU."""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            log.info("ONNX: используется CUDA GPU")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


# ── Типы данных ───────────────────────────────────────────────

@dataclass
class FaceResult:
    name:         str           # имя или "Unknown"
    confidence:   float         # 0..1 (cosine similarity)
    is_live:      bool          # прошёл антиспуфинг
    bbox:         tuple         # (top, right, bottom, left) — пиксели
    held_seconds: float = 0.0
    door_ready:   bool  = False


# ── IoU-трекер ────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """IoU двух bbox в формате (top, right, bottom, left)."""
    at, ar, ab, al = a
    bt, br, bb, bl = b
    ix1 = max(al, bl); iy1 = max(at, bt)
    ix2 = min(ar, br); iy2 = min(ab, bb)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ar - al) * (ab - at) + (br - bl) * (bb - bt) - inter
    return inter / (ua + 1e-6)


class _IoUTracker:
    """
    Простой трекер на основе IoU.
    Сопоставляет bbox текущего кадра с предыдущим по перекрытию.
    Даёт стабильный int-ID каждому лицу → таймер удержания работает корректно.
    """
    IOU_THRES = 0.35
    MAX_LOST  = 10   # кадров без совпадения → удалить трек

    def __init__(self):
        self._tracks: dict[int, dict] = {}  # id → {bbox, lost}
        self._next_id = 0

    def update(self, bboxes: list[tuple]) -> list[int]:
        """Возвращает список track-id для каждого bbox (порядок совпадает)."""
        used_tracks = set()
        result = [-1] * len(bboxes)

        # Сопоставление жадным алгоритмом по IoU
        for bi, bbox in enumerate(bboxes):
            best_id, best_iou = -1, self.IOU_THRES
            for tid, tr in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou(bbox, tr["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id >= 0:
                result[bi] = best_id
                used_tracks.add(best_id)
                self._tracks[best_id]["bbox"] = bbox
                self._tracks[best_id]["lost"] = 0
            else:
                # Новый трек
                self._tracks[self._next_id] = {"bbox": bbox, "lost": 0}
                result[bi] = self._next_id
                used_tracks.add(self._next_id)
                self._next_id += 1

        # Увеличиваем счётчик lost для невидимых треков
        for tid in list(self._tracks.keys()):
            if tid not in used_tracks:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.MAX_LOST:
                    del self._tracks[tid]

        return result


# ── Антиспуфинг ───────────────────────────────────────────────

def _laplacian(bgr: np.ndarray, bbox: tuple) -> float:
    top, right, bottom, left = bbox
    roi = bgr[top:bottom, left:right]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Основной класс ────────────────────────────────────────────

class FaceEngine:

    def __init__(self):
        self._known_emb:  list[np.ndarray] = []
        self._known_name: list[str]        = []
        self._tracker   = _IoUTracker()
        self._hold_ts:  dict[str, float]   = {}   # name → первый момент совпадения
        _get_app()   # прогрев при старте
        self._load_faces()

    # ── База лиц ──────────────────────────────────────────────

    def _load_faces(self):
        p = Path(config.KNOWN_FACES_DIR)
        if not p.exists():
            log.warning("Папка '%s' не найдена", config.KNOWN_FACES_DIR)
            return
        app = _get_app()
        loaded = 0
        for f in p.glob("*.[jp][pn][ge]*"):
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            faces = app.get(bgr)
            if not faces:
                log.warning("Лицо не найдено в %s", f.name)
                continue
            # Берём лицо с наибольшей det_score
            face = max(faces, key=lambda x: x.det_score)
            emb  = face.normed_embedding           # уже L2-нормирован
            name = f.stem.replace("_", " ").title()
            self._known_emb.append(emb)
            self._known_name.append(name)
            loaded += 1
        log.info("Загружено %d лиц из базы (%s)", loaded, config.KNOWN_FACES_DIR)

    def reload(self):
        self._known_emb.clear()
        self._known_name.clear()
        self._load_faces()

    @property
    def known_count(self) -> int:
        return len(self._known_name)

    # ── Обработка кадра ───────────────────────────────────────

    def process(self, bgr: np.ndarray) -> list[FaceResult]:
        """
        Принимает BGR-кадр.
        Возвращает список FaceResult.
        Скорость: ~15-30 мс на кадр на CPU (vs 200-500 мс у dlib HOG).
        """
        app = _get_app()

        # InsightFace принимает BGR напрямую — конвертация не нужна
        # Уменьшаем до 640px по ширине для скорости (детектор входной 320px)
        h, w = bgr.shape[:2]
        scale = min(1.0, 640 / w)
        if scale < 1.0:
            small = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = bgr
            scale = 1.0

        faces = app.get(small)   # детекция + эмбеддинги за один вызов

        if not faces:
            self._tracker.update([])
            return []

        # Конвертируем bbox InsightFace [x1,y1,x2,y2] → (top,right,bottom,left)
        bboxes = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            # Масштаб обратно к оригиналу
            x1 = int(x1 / scale); y1 = int(y1 / scale)
            x2 = int(x2 / scale); y2 = int(y2 / scale)
            # Ограничиваем по размеру кадра
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            bboxes.append((y1, x2, y2, x1))   # top, right, bottom, left

        track_ids = self._tracker.update(bboxes)

        now     = time.time()
        results = []

        for face, bbox, tid in zip(faces, bboxes, track_ids):
            # ── Антиспуфинг: Laplacian ──────────────────────
            lap    = _laplacian(bgr, bbox)
            is_live = lap >= config.LIVENESS_LAPLACIAN_THRES

            # ── Распознавание ────────────────────────────────
            name, conf = self._recognize(face.normed_embedding)

            # ── Таймер удержания (по track-id, не по имени) ──
            held = 0.0
            door_ready = False
            key = f"{name}_{tid}" if name != "Unknown" else None

            if key and is_live:
                if key not in self._hold_ts:
                    self._hold_ts[key] = now
                held = now - self._hold_ts[key]
                door_ready = held >= config.HOLD_SECONDS
            elif key:
                self._hold_ts.pop(key, None)

            results.append(FaceResult(
                name=name,
                confidence=conf,
                is_live=is_live,
                bbox=bbox,
                held_seconds=held,
                door_ready=door_ready,
            ))

        # Чистим старые ключи hold_ts
        active_keys = {f"{r.name}_{tid}" for r, tid in zip(results, track_ids) if r.name != "Unknown"}
        for k in list(self._hold_ts.keys()):
            if k not in active_keys:
                del self._hold_ts[k]

        return results

    # ── Распознавание ─────────────────────────────────────────

    def _recognize(self, emb: np.ndarray) -> tuple[str, float]:
        if not self._known_emb:
            return "Unknown", 0.0
        # Cosine similarity = dot(emb, known) когда оба L2-нормированы
        sims  = np.dot(self._known_emb, emb)   # shape (N,)
        best_i = int(np.argmax(sims))
        sim    = float(sims[best_i])
        # sim ∈ [-1, 1] → нормируем в [0, 1]
        conf = (sim + 1.0) / 2.0
        if conf >= config.FACE_CONFIDENCE_THRES:
            return self._known_name[best_i], conf
        return "Unknown", conf

    # ── Совместимость с gui.py ────────────────────────────────

    def reset_hold(self, name: str):
        """Сбрасывает все таймеры для данного имени (вызывается из GUI)."""
        for k in list(self._hold_ts.keys()):
            if k.startswith(f"{name}_"):
                del self._hold_ts[k]
