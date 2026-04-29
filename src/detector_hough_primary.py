"""
Detector de paneles de autopista - Enfoque HOUGH PRIMARY.

Sigue exactamente el orden especificado en el enunciado:
  1. HoughLinesP en toda la imagen → detectar estructura rectangular (H×V líneas)
  2. Formar candidatos a panel (intersecciones de líneas H×V)
  3. Para cada candidato: validar con HSV azul saturado
  4. Calcular score con correlación de máscara ideal 40×80
"""

import cv2
import numpy as np
from typing import List, Tuple


class PanelDetectorHoughPrimary:
    """
    Detector de paneles usando HoughLinesP como mecanismo PRIMARY.

    Pipeline:
      1. Preprocesamiento (CLAHE + Canny)
      2. HoughLinesP en imagen completa → H y V líneas
      3. Agrupar líneas paralelas cercanas
      4. Formar candidatos rectangulares (H×V intersecciones)
      5. Validar con color azul HSV
      6. Score mediante correlación de máscara ideal
    """

    def __init__(self):
        # ── Preprocesamiento ──────────────────────────────────────────────
        self._clahe       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._blur_ksize  = 5
        self._canny_low   = 40
        self._canny_high  = 120

        # ── HoughLinesP en imagen completa ────────────────────────────────
        self._hough_threshold    = 30    # votos mínimos (más restrictivo que antes)
        self._min_line_length    = 40    # línea mínima 40px (más restrictivo)
        self._max_line_gap       = 15
        self._angle_tol_deg      = 18.0
        self._group_dist         = 12    # agrupar paralelas a ≤12px

        # ── Filtro geométrico de candidatos rectangulares ──────────────────
        self._min_w      = 25
        self._max_w      = 800
        self._min_h      = 20
        self._max_h      = 400
        self._min_aspect = 1.0
        self._max_aspect = 6.0

        # ── Validación HSV azul ───────────────────────────────────────────
        self._blue_lower      = np.array([100, 200, 70], dtype=np.uint8)
        self._blue_upper      = np.array([128, 255, 255], dtype=np.uint8)
        self._min_blue_ratio  = 0.20    # proporción mínima de azul en candidato

        # ── Score por correlación de máscara ideal ────────────────────────
        self._mask_h          = 40
        self._mask_w          = 80
        self._score_threshold = 0.30
        self._ideal_mask      = self._build_ideal_mask()

    # ─────────────────────────────────────────────────────────────────────
    # Máscara ideal
    # ─────────────────────────────────────────────────────────────────────

    def _build_ideal_mask(self) -> np.ndarray:
        """Máscara ideal 40×80: interior=1, borde blanco=0."""
        mask = np.zeros((self._mask_h, self._mask_w), dtype=np.uint8)
        by = int(self._mask_h * 0.20)
        bx = int(self._mask_w * 0.16)
        mask[by : self._mask_h - by, bx : self._mask_w - bx] = 1
        return mask

    # ─────────────────────────────────────────────────────────────────────
    # Fase 1 – Preprocesamiento y detección de líneas Hough
    # ─────────────────────────────────────────────────────────────────────

    def _detect_hough_lines(
        self, img_bgr: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """
        Ejecuta el pipeline completo de Hough en la imagen:
        1. Preprocesamiento (CLAHE + Gaussian + Canny)
        2. HoughLinesP
        3. Clasificación H/V
        4. Agrupación de paralelas

        Devuelve listas de posiciones (en píxeles) de líneas H y V.
        """
        # Preprocesamiento
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (self._blur_ksize, self._blur_ksize), 1)
        edges   = cv2.Canny(blurred, self._canny_low, self._canny_high,
                           apertureSize=3)

        # HoughLinesP en imagen completa
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=self._min_line_length,
            maxLineGap=self._max_line_gap,
        )

        if raw is None:
            return [], []

        h_pos: List[float] = []
        v_pos: List[float] = []

        tol = np.deg2rad(self._angle_tol_deg)
        for line in raw:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-9))

            if angle <= tol:
                # Horizontal: guardar y media
                h_pos.append((y1 + y2) / 2.0)
            elif angle >= (np.pi / 2.0 - tol):
                # Vertical: guardar x media
                v_pos.append((x1 + x2) / 2.0)

        # Agrupar líneas paralelas cercanas
        h_lines = self._group_positions(h_pos)
        v_lines = self._group_positions(v_pos)

        return h_lines, v_lines

    def _group_positions(self, positions: List[float]) -> List[float]:
        """Agrupa posiciones cuya diferencia sea ≤ group_dist."""
        if not positions:
            return []

        positions = sorted(positions)
        groups: List[List[float]] = []
        current = [positions[0]]

        for p in positions[1:]:
            if p - current[-1] <= self._group_dist:
                current.append(p)
            else:
                groups.append(current)
                current = [p]
        groups.append(current)

        return [float(np.median(g)) for g in groups]

    # ─────────────────────────────────────────────────────────────────────
    # Fase 2 – Formar candidatos rectangulares
    # ─────────────────────────────────────────────────────────────────────

    def _form_candidates(
        self,
        h_lines: List[float],
        v_lines: List[float],
        img_h: int,
        img_w: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Forma candidatos rectangulares a partir de pares (Hi, Hj) × (Vk, Vl).
        Filtra por geometría (tamaño, aspect ratio).
        """
        candidates = []

        for i in range(len(h_lines)):
            for j in range(i + 1, len(h_lines)):
                y_top = h_lines[i]
                y_bot = h_lines[j]
                height = y_bot - y_top

                if not (self._min_h <= height <= self._max_h):
                    continue

                for k in range(len(v_lines)):
                    for l in range(k + 1, len(v_lines)):
                        x_left = v_lines[k]
                        x_right = v_lines[l]
                        width = x_right - x_left

                        if not (self._min_w <= width <= self._max_w):
                            continue

                        aspect = width / (height + 1e-6)
                        if not (self._min_aspect <= aspect <= self._max_aspect):
                            continue

                        x1 = int(max(0, x_left))
                        y1 = int(max(0, y_top))
                        x2 = int(min(img_w - 1, x_right))
                        y2 = int(min(img_h - 1, y_bot))

                        candidates.append((x1, y1, x2, y2))

        return candidates

    # ─────────────────────────────────────────────────────────────────────
    # Fase 3 – Validación HSV azul (filtro rápido)
    # ─────────────────────────────────────────────────────────────────────

    def _blue_ratio(self, img_bgr: np.ndarray, x1: int, y1: int,
                    x2: int, y2: int) -> float:
        """Computa la proporción de píxeles azul saturado en la región."""
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blue_lower, self._blue_upper)
        total = roi.shape[0] * roi.shape[1]
        return np.sum(mask > 0) / total

    # ─────────────────────────────────────────────────────────────────────
    # Fase 4 – Score mediante correlación de máscara ideal
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
        Detecta paneles siguiendo el pipeline del enunciado:
        1. HoughLinesP → H y V líneas
        2. Candidatos rectangulares (H×V)
        3. Filtro azul HSV
        4. Score correlación

        Devuelve [[x1, y1, x2, y2, score], ...]
        """
        img_h, img_w = image.shape[:2]

        # 1. Detectar líneas H y V
        h_lines, v_lines = self._detect_hough_lines(image)

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        # 2. Formar candidatos
        candidates = self._form_candidates(h_lines, v_lines, img_h, img_w)

        # 3 y 4. Validar y puntuar
        detections = []
        for (x1, y1, x2, y2) in candidates:
            # Filtro rápido: azul mínimo
            if self._blue_ratio(image, x1, y1, x2, y2) < self._min_blue_ratio:
                continue

            # Score detallado
            score = self._compute_score(image, x1, y1, x2, y2)
            if score >= self._score_threshold:
                detections.append([x1, y1, x2, y2, score])

        return detections
