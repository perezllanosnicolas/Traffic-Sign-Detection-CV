"""
Detector alternativo de paneles de autopista.

Pipeline:
  1. Máscara HSV de azul saturado + morfología → blobs candidatos
  2. Filtro geométrico (tamaño, aspect ratio)
  3. Expansión acotada del bounding box para incluir el borde blanco
  4. HoughLinesP dentro del ROI: confirma estructura rectangular (H + V)
     y modula el score sin alterar las coordenadas
  5. Score = correlación F1+especificidad con máscara azul ideal 40×80,
     multiplicado por el factor Hough [0.8 – 1.0]
"""

import cv2
import numpy as np
from typing import List, Tuple


class PanelDetectorAlt:
    """
    Detecta paneles rectangulares azules combinando segmentación de color
    con la Transformada de Hough probabilística (HoughLinesP).

    El azul saturado localiza los candidatos; HoughLinesP confirma que
    el ROI tiene la estructura rectangular del panel (borde H y V).

    Interfaz:
        detect(image: np.ndarray) -> List[[x1, y1, x2, y2, score]]
    """

    def __init__(self):
        # ── Máscara de color azul saturado ────────────────────────────────
        self._blue_lower = np.array([100, 200,  70], dtype=np.uint8)
        self._blue_upper = np.array([128, 255, 255], dtype=np.uint8)

        # ── Morfología para limpiar la máscara azul ───────────────────────
        self._morph_open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._morph_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # ── Filtros geométricos sobre el blob azul ────────────────────────
        self._min_blob_area   = 200
        self._min_blob_width  = 15
        self._min_blob_height = 10
        self._min_aspect      = 0.5   # ancho / alto
        self._max_aspect      = 8.0

        # ── Expansión del bbox para capturar el borde blanco ──────────────
        # Se usa max(min_px, ratio × lado) acotado por max_px, para que
        # funcione bien tanto en paneles pequeños como en grandes.
        self._expand_min_px  = 5
        self._expand_max_px  = 25
        self._expand_ratio   = 0.12

        # ── HoughLinesP dentro del ROI (validación de estructura) ─────────
        self._canny_low       = 50
        self._canny_high      = 150
        self._hough_threshold = 12
        self._min_line_ratio  = 0.18  # min_length = ratio × min(w, h)
        self._max_gap_ratio   = 0.08
        self._angle_tol_deg   = 20.0

        # ── Score por correlación de máscara azul ideal ───────────────────
        self._mask_h          = 40
        self._mask_w          = 80
        self._score_threshold = 0.25
        self._ideal_mask      = self._build_ideal_mask()

    # ─────────────────────────────────────────────────────────────────────
    # Máscara ideal (interior del panel = 1, borde blanco = 0)
    # ─────────────────────────────────────────────────────────────────────

    def _build_ideal_mask(self) -> np.ndarray:
        mask = np.zeros((self._mask_h, self._mask_w), dtype=np.uint8)
        by = int(self._mask_h * 0.20)
        bx = int(self._mask_w * 0.16)
        mask[by : self._mask_h - by, bx : self._mask_w - bx] = 1
        return mask

    # ─────────────────────────────────────────────────────────────────────
    # Fase 1 – Máscara azul y candidatos
    # ─────────────────────────────────────────────────────────────────────

    def _blue_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blue_lower, self._blue_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_close_k,
                                iterations=2)
        return mask

    def _expand(self, v: int, delta: int, lo: int, hi: int) -> int:
        return int(np.clip(v + delta, lo, hi))

    def _candidate_boxes(
        self, mask: np.ndarray, img_h: int, img_w: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Extrae blobs del azul, filtra por geometría y devuelve bboxes
        con expansión acotada para incluir el borde blanco.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < self._min_blob_width or h < self._min_blob_height:
                continue
            if w * h < self._min_blob_area:
                continue
            aspect = w / float(h)
            if not (self._min_aspect <= aspect <= self._max_aspect):
                continue

            # Expansión acotada: ni demasiado pequeña ni demasiado grande
            ew = int(np.clip(w * self._expand_ratio,
                             self._expand_min_px, self._expand_max_px))
            eh = int(np.clip(h * self._expand_ratio,
                             self._expand_min_px, self._expand_max_px))

            x1 = self._expand(x,     -ew, 0,     img_w)
            y1 = self._expand(y,     -eh, 0,     img_h)
            x2 = self._expand(x + w,  ew, 0,     img_w)
            y2 = self._expand(y + h,  eh, 0,     img_h)
            boxes.append((x1, y1, x2, y2))

        return boxes

    # ─────────────────────────────────────────────────────────────────────
    # Fase 2 – Validación Hough (factor de modulación del score)
    # ─────────────────────────────────────────────────────────────────────

    def _hough_factor(self, roi_bgr: np.ndarray) -> float:
        """
        Detecta líneas H y V en el ROI mediante HoughLinesP.
        Devuelve un factor:
          1.0 → líneas H y V presentes (estructura rectangular confirmada)
          0.9 → solo un tipo de línea
          0.8 → ninguna línea detectada
        """
        if roi_bgr.size == 0:
            return 0.8

        h, w = roi_bgr.shape[:2]
        min_side = min(w, h)
        min_len  = max(8, int(min_side * self._min_line_ratio))
        max_gap  = max(3, int(min_side * self._max_gap_ratio))

        gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        if lines is None:
            return 0.8

        tol   = self._angle_tol_deg
        has_h = has_v = False
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            angle = abs(np.degrees(
                np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1) + 1e-9)
            ))
            if angle <= tol:
                has_h = True
            if angle >= (90.0 - tol):
                has_v = True
            if has_h and has_v:
                break

        if has_h and has_v:
            return 1.0
        if has_h or has_v:
            return 0.9
        return 0.8

    # ─────────────────────────────────────────────────────────────────────
    # Fase 3 – Score por correlación de máscara azul
    # ─────────────────────────────────────────────────────────────────────

    def _compute_score(
        self,
        img_bgr: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> float:
        """
        Redimensiona el ROI a 40×80, extrae la máscara binaria de azul
        y la correlaciona con la máscara ideal del panel.

        score_base = 0.55 · F1 + 0.45 · specificity
          F1         → azul bien colocado en la zona interior
          specificity → no-azul en la zona de borde (borde blanco)

        score final = score_base × hough_factor
        """
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        hf = self._hough_factor(roi)

        resized = cv2.resize(
            roi, (self._mask_w, self._mask_h), interpolation=cv2.INTER_AREA
        )
        hsv_roi = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        M = (cv2.inRange(hsv_roi, self._blue_lower, self._blue_upper) > 0
             ).astype(np.uint8)

        ideal     = self._ideal_mask
        tp        = float(np.sum((M == 1) & (ideal == 1)))
        fp        = float(np.sum((M == 1) & (ideal == 0)))
        tn        = float(np.sum((M == 0) & (ideal == 0)))
        positives = float(np.sum(ideal == 1))
        negatives = float(np.sum(ideal == 0))

        if positives <= 0:
            return 0.0

        recall      = tp / positives
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1          = (2.0 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)
        specificity = tn / negatives if negatives > 0 else 0.0

        score_base = 0.55 * f1 + 0.45 * specificity
        return float(np.clip(score_base * hf, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[List]:
        """
        Detecta paneles en una imagen BGR.

        Devuelve lista de [x1, y1, x2, y2, score].
        El NMS se aplica externamente en main.py.
        """
        img_h, img_w = image.shape[:2]

        mask       = self._blue_mask(image)
        candidates = self._candidate_boxes(mask, img_h, img_w)

        detections = []
        for (x1, y1, x2, y2) in candidates:
            score = self._compute_score(image, x1, y1, x2, y2)
            if score >= self._score_threshold:
                detections.append([x1, y1, x2, y2, score])

        return detections
