"""
Script de debug del detector Hough Primary (Hough como detector primario).
Exporta imágenes intermedias en carpeta 'debug_hough_primary/' mostrando cada fase:
  - Preprocesamiento (CLAHE, Canny)
  - HoughLinesP en imagen completa
  - Líneas H y V detectadas y agrupadas
  - Candidatos rectangulares formados
  - Validación HSV azul
  - Resultado final
"""

import cv2
import numpy as np
import os
from src.detector_hough_primary import PanelDetectorHoughPrimary

def debug_detector_hough_primary(img_path: str, output_dir: str = "debug_hough_primary"):
    """Ejecuta el detector Hough Primary paso a paso exportando imágenes intermedias."""

    os.makedirs(output_dir, exist_ok=True)

    # Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: no se puede leer {img_path}")
        return

    img_name = os.path.basename(img_path).replace(".png", "")
    img_h, img_w = img.shape[:2]
    print(f"\n=== DEBUG HOUGH PRIMARY: {img_name} ({img_w}×{img_h}) ===\n")

    # Instanciar detector
    det = PanelDetectorHoughPrimary()

    # ── FASE 1: Preprocesamiento ───────────────────────────────────────
    print("[1] Preprocesamiento (CLAHE + GaussianBlur + Canny)...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = det._clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (det._blur_ksize, det._blur_ksize), 1)
    edges = cv2.Canny(blurred, det._canny_low, det._canny_high, apertureSize=3)

    cv2.imwrite(f"{output_dir}/01_gray.png", gray)
    cv2.imwrite(f"{output_dir}/02_clahe_enhanced.png", enhanced)
    cv2.imwrite(f"{output_dir}/03_blurred.png", blurred)
    cv2.imwrite(f"{output_dir}/04_canny_edges.png", edges)
    print(f"    → Píxeles de borde: {np.sum(edges > 0)}")

    # ── FASE 2: HoughLinesP en imagen completa ─────────────────────────
    print("[2] Ejecutando HoughLinesP en imagen completa...")
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=det._hough_threshold,
        minLineLength=det._min_line_length,
        maxLineGap=det._max_line_gap,
    )

    if raw is None:
        print("    → No se detectaron líneas!")
        return

    print(f"    → Líneas brutas detectadas: {len(raw)}")

    # Dibujar líneas brutas
    img_lines_raw = img.copy()
    for line in raw:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines_raw, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.imwrite(f"{output_dir}/05_hough_lines_raw.png", img_lines_raw)

    # ── FASE 3: Clasificación H/V y agrupación ─────────────────────────
    print("[3] Clasificando líneas en H/V y agrupando paralelas...")
    h_pos = []
    v_pos = []
    tol = np.deg2rad(det._angle_tol_deg)

    for line in raw:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-9))

        if angle <= tol:
            h_pos.append((y1 + y2) / 2.0)
        elif angle >= (np.pi / 2.0 - tol):
            v_pos.append((x1 + x2) / 2.0)

    print(f"    → Posiciones H brutas: {len(h_pos)}")
    print(f"    → Posiciones V brutas: {len(v_pos)}")

    h_lines = det._group_positions(h_pos)
    v_lines = det._group_positions(v_pos)

    print(f"    → Líneas H agrupadas: {len(h_lines)} → {[round(y) for y in h_lines]}")
    print(f"    → Líneas V agrupadas: {len(v_lines)} → {[round(x) for x in v_lines]}")

    # Dibujar líneas agrupadas
    img_lines_grouped = img.copy()
    # H lines (rojo)
    for y in h_lines:
        cv2.line(img_lines_grouped, (0, int(y)), (img_w, int(y)), (0, 0, 255), 2)
    # V lines (verde)
    for x in v_lines:
        cv2.line(img_lines_grouped, (int(x), 0), (int(x), img_h), (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/06_hough_lines_grouped.png", img_lines_grouped)

    # ── FASE 4: Formar candidatos rectangulares ────────────────────────
    print("[4] Formando candidatos rectangulares (H×V intersecciones)...")
    candidates = det._form_candidates(h_lines, v_lines, img_h, img_w)
    print(f"    → Candidatos después filtro geométrico: {len(candidates)}")

    # Dibujar candidatos
    img_candidates = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(candidates):
        cv2.rectangle(img_candidates, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_candidates, str(i), (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite(f"{output_dir}/07_candidates.png", img_candidates)

    # ── FASE 5: Validación HSV azul (filtro rápido) ───────────────────
    print("[5] Validando candidatos con HSV azul...")
    blue_filtered = []
    for i, (x1, y1, x2, y2) in enumerate(candidates):
        ratio = det._blue_ratio(img, x1, y1, x2, y2)
        if ratio >= det._min_blue_ratio:
            blue_filtered.append((x1, y1, x2, y2, ratio))
            print(f"    Candidato {i}: blue_ratio={ratio:.2%} ✓")
        else:
            print(f"    Candidato {i}: blue_ratio={ratio:.2%} ✗ (< {det._min_blue_ratio})")

    print(f"\n    → Pasan filtro azul: {len(blue_filtered)}/{len(candidates)}")

    # Dibujar candidatos que pasan filtro azul
    img_blue_filtered = img.copy()
    for i, (x1, y1, x2, y2, ratio) in enumerate(blue_filtered):
        cv2.rectangle(img_blue_filtered, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_blue_filtered, f"{ratio:.1%}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(f"{output_dir}/08_blue_filtered.png", img_blue_filtered)

    # ── FASE 6: Score mediante correlación de máscara ──────────────────
    print("[6] Calculando score por correlación de máscara ideal...")
    final_detections = []
    for i, (x1, y1, x2, y2, ratio) in enumerate(blue_filtered):
        score = det._compute_score(img, x1, y1, x2, y2)
        print(f"    Candidato {i}: ({x1},{y1})-({x2},{y2}) score={score:.3f}")

        if score >= det._score_threshold:
            final_detections.append([x1, y1, x2, y2, score])
            print(f"        ✓ ACEPTADO")
        else:
            print(f"        ✗ Rechazado (score < {det._score_threshold})")

    # ── FASE 7: Resultado final ────────────────────────────────────────
    print(f"\n[7] Resultado final: {len(final_detections)} detecciones\n")
    img_final = img.copy()
    for x1, y1, x2, y2, score in final_detections:
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_final, f"{score:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/09_final_detections.png", img_final)

    print(f"✓ Imágenes debug exportadas a: {output_dir}/")


if __name__ == "__main__":
    # Debug de una imagen con panel grande
    debug_detector_hough_primary("data/test_detection/00006.png")
