"""
Detector Híbrido de paneles de autopista.

Combina lo mejor de ambos enfoques:
  1. MÁSCARA AZUL → detecta candidatos (muy preciso, específico a paneles)
  2. HOUGH → refina los bordes exactos (encuentra líneas H/V del marco blanco)

Pipeline:
  1. HSV azul → blobs candidatos
  2. Filtro geométrico
  3. Para cada candidato: expandir zona de búsqueda
  4. HoughLinesP en zona expandida → detectar bordes blancos (H/V)
  5. Refinar bbox usando esas líneas
  6. Score por correlación de máscara ideal 40×80
"""

import cv2
import numpy as np
from typing import List, Tuple


class PanelDetectorHybrid:
    """
    Detector híbrido que combina:
    - Máscara azul HSV (detector primario - detecta dónde están los paneles)
    - HoughLinesP (refinador - encuentra los bordes exactos)

    Interfaz:
        detect(image: np.ndarray) -> List[[x1, y1, x2, y2, score]]
    """

    def __init__(self):
        # ── Máscara de color azul saturado (DETECTOR PRIMARIO) ───────────
        self._blue_lower = np.array([100, 200,  70], dtype=np.uint8)
        self._blue_upper = np.array([128, 255, 255], dtype=np.uint8)

        # ── Morfología para limpiar la máscara azul ────────────────────────
        self._morph_open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._morph_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # ── Filtros geométricos sobre el blob azul ─────────────────────────
        self._min_blob_area   = 200
        self._min_blob_width  = 15
        self._min_blob_height = 10
        self._min_aspect      = 0.5   # ancho / alto
        self._max_aspect      = 8.0

        # ── Expansión para zona de búsqueda Hough ──────────────────────────
        # Se expande el blob azul para que el borde blanco quede en el ROI
        self._search_expand_px = 35

        # ── HoughLinesP (REFINADOR - dentro del ROI expandido) ────────────
        self._canny_low       = 50
        self._canny_high      = 150
        self._hough_threshold = 15
        self._min_line_ratio  = 0.18
        self._max_gap_ratio   = 0.08
        self._angle_tol_deg   = 20.0

        # ── Score por correlación de máscara ideal ────────────────────────
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
    # FASE 1 – Detector Primario: Máscara azul + blobs
    # ─────────────────────────────────────────────────────────────────────

    def _blue_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extrae máscara de píxeles azul saturado."""
        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blue_lower, self._blue_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_close_k,
                                iterations=2)
        return mask

    def _blob_candidates(
        self, mask: np.ndarray, img_h: int, img_w: int
    ) -> List[Tuple[int, int, int, int, Tuple[int, int, int, int]]]:
        """
        Extrae blobs azules, filtra por geometría.
        Devuelve lista de (search_x1, search_y1, search_x2, search_y2, blob_bbox)
        donde blob_bbox es (blob_x1, blob_y1, blob_x2, blob_y2)
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        ep = self._search_expand_px

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < self._min_blob_width or h < self._min_blob_height:
                continue
            if w * h < self._min_blob_area:
                continue
            aspect = w / float(h)
            if not (self._min_aspect <= aspect <= self._max_aspect):
                continue

            # Blob interior
            bx1, by1 = x, y
            bx2, by2 = x + w, y + h

            # Zona de búsqueda expandida (para que el borde blanco quede dentro)
            sx1 = max(0,     bx1 - ep)
            sy1 = max(0,     by1 - ep)
            sx2 = min(img_w, bx2 + ep)
            sy2 = min(img_h, by2 + ep)

            candidates.append((sx1, sy1, sx2, sy2, (bx1, by1, bx2, by2)))

        return candidates

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2 – Refinador: HoughLinesP para ajustar bordes
    # ─────────────────────────────────────────────────────────────────────

    def _refine_with_hough(
        self,
        img_bgr: np.ndarray,
        sx1: int, sy1: int, sx2: int, sy2: int,
        blob: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Ejecuta Hough en la zona de búsqueda y ajusta el bbox a las
        líneas H/V detectadas (bordes blancos del panel).

        Devuelve el bbox refinado (fx1, fy1, fx2, fy2) en coordenadas de imagen.
        Si no hay líneas suficientes, devuelve una expansión fija del blob.
        """
        roi = img_bgr[sy1:sy2, sx1:sx2]
        roi_h, roi_w = roi.shape[:2]

        if roi.size == 0:
            return blob  # Fallback: devolver blob original

        # Preprocesamiento
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)

        min_len = max(8, int(min(roi_w, roi_h) * self._min_line_ratio))
        max_gap = max(3, int(min(roi_w, roi_h) * self._max_gap_ratio))

        # HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        # Blob en coordenadas ROI
        bx1_roi = blob[0] - sx1
        by1_roi = blob[1] - sy1
        bx2_roi = blob[2] - sx1
        by2_roi = blob[3] - sy1

        # Fallback: expansión pequeña del blob si no hay líneas
        fallback_px = 10
        fy1_roi = max(0, by1_roi - fallback_px)
        fy2_roi = min(roi_h, by2_roi + fallback_px)
        fx1_roi = max(0, bx1_roi - fallback_px)
        fx2_roi = min(roi_w, bx2_roi + fallback_px)

        # Si hay líneas Hough, usarlas para refinar
        if lines is not None:
            tol = np.deg2rad(self._angle_tol_deg)
            h_lines = []
            v_lines = []

            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = abs(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1) + 1e-9))

                if angle <= tol:
                    y_mid = (ly1 + ly2) / 2.0
                    h_lines.append(y_mid)
                elif angle >= (np.pi / 2.0 - tol):
                    x_mid = (lx1 + lx2) / 2.0
                    v_lines.append(x_mid)

            # Refinar bordes usando las líneas encontradas
            # Buscar líneas H en la zona superior e inferior del blob
            if h_lines:
                h_above = [y for y in h_lines if y < by1_roi - 5]
                h_below = [y for y in h_lines if y > by2_roi + 5]

                if h_above:
                    fy1_roi = max(0, int(max(h_above)))  # línea más cercana arriba
                if h_below:
                    fy2_roi = min(roi_h, int(min(h_below)))  # línea más cercana abajo

            # Refinar bordes V
            if v_lines:
                v_left  = [x for x in v_lines if x < bx1_roi - 5]
                v_right = [x for x in v_lines if x > bx2_roi + 5]

                if v_left:
                    fx1_roi = max(0, int(max(v_left)))
                if v_right:
                    fx2_roi = min(roi_w, int(min(v_right)))

        # Convertir a coordenadas de imagen
        fx1 = sx1 + fx1_roi
        fy1 = sy1 + fy1_roi
        fx2 = sx1 + fx2_roi
        fy2 = sy1 + fy2_roi

        return (fx1, fy1, fx2, fy2)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3 – Score por correlación de máscara ideal
    # ─────────────────────────────────────────────────────────────────────

    def _compute_score(self, img_bgr: np.ndarray, x1: int, y1: int,
                       x2: int, y2: int) -> float:
        """
        Redimensiona ROI a 40×80, extrae máscara azul y correlaciona
        con la máscara ideal del panel.

        score = 0.55 * F1 + 0.45 * specificity
        """
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        resized = cv2.resize(
            roi, (self._mask_w, self._mask_h), interpolation=cv2.INTER_AREA
        )
        hsv_roi = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        M = (cv2.inRange(hsv_roi, self._blue_lower, self._blue_upper) > 0
             ).astype(np.uint8)

        ideal = self._ideal_mask
        tp = float(np.sum((M == 1) & (ideal == 1)))
        fp = float(np.sum((M == 1) & (ideal == 0)))
        tn = float(np.sum((M == 0) & (ideal == 0)))
        positives = float(np.sum(ideal == 1))
        negatives = float(np.sum(ideal == 0))

        if positives <= 0:
            return 0.0

        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        specificity = tn / negatives if negatives > 0 else 0.0

        return float(np.clip(0.55 * f1 + 0.45 * specificity, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[List]:
        """
        Detecta paneles usando enfoque híbrido:
        1. Máscara azul → candidatos (detector primario)
        2. Hough → refina bordes (refinador)
        3. Score por correlación

        Devuelve [[x1, y1, x2, y2, score], ...]
        """
        img_h, img_w = image.shape[:2]

        # Fase 1: Detectar blobs azules
        mask = self._blue_mask(image)
        candidates = self._blob_candidates(mask, img_h, img_w)

        # Fases 2 y 3: Refinar y puntuar
        detections = []
        for (sx1, sy1, sx2, sy2, blob) in candidates:
            # Refinar con Hough
            (x1, y1, x2, y2) = self._refine_with_hough(
                image, sx1, sy1, sx2, sy2, blob
            )

            # Calcular score
            score = self._compute_score(image, x1, y1, x2, y2)
            if score >= self._score_threshold:
                detections.append([x1, y1, x2, y2, score])

        return detections
