"""
Script de debug del detector Hough.
Exporta imágenes intermedias en carpeta 'debug/' mostrando cada fase:
  - Máscara de azul (bruta)
  - Máscara tras morfología (OPEN + CLOSE)
  - Contornos detectados
  - Para cada candidato: Canny edges, HoughLinesP, resultado final
"""

import cv2
import numpy as np
import os
from src.detector_alt import PanelDetectorAlt

def debug_detector(img_path: str, output_dir: str = "debug"):
    """Ejecuta el detector paso a paso exportando imágenes intermedias."""

    os.makedirs(output_dir, exist_ok=True)

    # Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: no se puede leer {img_path}")
        return

    img_name = os.path.basename(img_path).replace(".png", "")
    img_h, img_w = img.shape[:2]
    print(f"\n=== DEBUG: {img_name} ({img_w}×{img_h}) ===\n")

    # Instanciar detector
    det = PanelDetectorAlt()

    # ── FASE 1: Máscara HSV bruta ──────────────────────────────────────
    print("[1] Generando máscara HSV de azul...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_raw = cv2.inRange(hsv, det._blue_lower, det._blue_upper)
    cv2.imwrite(f"{output_dir}/01_mask_raw.png", mask_raw)
    print(f"    → Píxeles azules: {np.sum(mask_raw > 0)}")

    # ── FASE 2: Morfología (OPEN) ──────────────────────────────────────
    print("[2] Aplicando MORPH_OPEN...")
    mask_open = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, det._morph_open_k)
    cv2.imwrite(f"{output_dir}/02_mask_after_open.png", mask_open)
    print(f"    → Píxeles después de OPEN: {np.sum(mask_open > 0)}")

    # ── FASE 3: Morfología (CLOSE) ─────────────────────────────────────
    print("[3] Aplicando MORPH_CLOSE (2 iteraciones)...")
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, det._morph_close_k,
                                   iterations=2)
    cv2.imwrite(f"{output_dir}/03_mask_after_close.png", mask_closed)
    print(f"    → Píxeles después de CLOSE: {np.sum(mask_closed > 0)}")

    # ── FASE 4: Extracción de contornos ────────────────────────────────
    print("[4] Extrayendo contornos...")
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    print(f"    → Contornos encontrados: {len(contours)}")

    # Dibujar contornos
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/04_contours.png", img_contours)

    # ── FASE 5: Filtro geométrico y candidatos ─────────────────────────
    print("[5] Filtrando por geometría...")
    candidates = det._candidate_boxes(mask_closed, img_h, img_w)
    print(f"    → Candidatos después del filtro: {len(candidates)}")

    # Dibujar candidatos
    img_candidates = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(candidates):
        cv2.rectangle(img_candidates, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_candidates, str(i), (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite(f"{output_dir}/05_candidates.png", img_candidates)

    # ── FASE 6: Para cada candidato: Canny + HoughLinesP ───────────────
    print("[6] Analizando cada candidato con Canny + Hough...\n")

    final_detections = []
    for idx, (x1, y1, x2, y2) in enumerate(candidates):
        print(f"   Candidato {idx}: ({x1},{y1})-({x2},{y2}) = {x2-x1}×{y2-y1}")

        roi = img[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Canny
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, det._canny_low, det._canny_high)

        subdir = f"{output_dir}/06_cand_{idx:02d}"
        os.makedirs(subdir, exist_ok=True)

        cv2.imwrite(f"{subdir}/a_roi_gray.png", gray)
        cv2.imwrite(f"{subdir}/b_canny_edges.png", edges)

        # HoughLinesP
        min_len = max(8, int(min(roi_w, roi_h) * det._min_line_ratio))
        max_gap = max(3, int(min(roi_w, roi_h) * det._max_gap_ratio))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=det._hough_threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        # Dibujar líneas Hough
        roi_lines = roi.copy()
        h_count = v_count = 0
        if lines is not None:
            tol = det._angle_tol_deg
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = abs(np.degrees(
                    np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1) + 1e-9)
                ))
                if angle <= tol:
                    cv2.line(roi_lines, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                    h_count += 1
                elif angle >= (90.0 - tol):
                    cv2.line(roi_lines, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
                    v_count += 1

        cv2.imwrite(f"{subdir}/c_hough_lines.png", roi_lines)
        print(f"      Líneas H: {h_count}, Líneas V: {v_count}")

        # Score
        score = det._compute_score(img, x1, y1, x2, y2)
        print(f"      Score: {score:.3f}")

        if score >= det._score_threshold:
            final_detections.append([x1, y1, x2, y2, score])
            print(f"      ✓ ACEPTADO")
        else:
            print(f"      ✗ Rechazado (score < {det._score_threshold})")
        print()

    # ── FASE 7: Resultado final ────────────────────────────────────────
    print(f"[7] Resultado final: {len(final_detections)} detecciones\n")
    img_final = img.copy()
    for x1, y1, x2, y2, score in final_detections:
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_final, f"{score:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/07_final_detections.png", img_final)

    print(f"✓ Imágenes debug exportadas a: {output_dir}/")


if __name__ == "__main__":
    # Debug de una imagen con panel grande
    debug_detector("data/test_detection/00006.png")
