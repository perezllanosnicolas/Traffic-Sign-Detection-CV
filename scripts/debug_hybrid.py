"""
Debugging script for the Hybrid Detector (Blue color + Hough refinement).

This script executes the detection pipeline step-by-step and exports 
intermediate images to a specified output directory to visualize each phase:
  - Blue mask generation
  - Blob candidates extraction
  - Search zone and HoughLinesP analysis per blob
  - Bounding box refinement using Hough lines
  - Final scoring and validation
"""

import argparse
import os
import cv2
import numpy as np
from src.detector_hybrid import PanelDetectorHybrid


def debug_detector_hybrid(img_path: str, output_dir: str = "debug_hybrid") -> None:
    """
    Executes the Hybrid detector step-by-step, exporting intermediate phase images.

    Args:
        img_path (str): Path to the input test image.
        output_dir (str): Directory where intermediate debug images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image at {img_path}")
        return

    img_name = os.path.basename(img_path).replace(".png", "")
    img_h, img_w = img.shape[:2]
    print(f"\n=== DEBUG HYBRID: {img_name} ({img_w}x{img_h}) ===\n")

    # Instantiate detector
    det = PanelDetectorHybrid()

    # ── PHASE 1: Blue Mask ─────────────────────────────────────────────
    print("[1] Detecting blue mask...")
    mask = det._blue_mask(img)
    cv2.imwrite(os.path.join(output_dir, "01_blue_mask.png"), mask)
    print(f"    -> Blue pixels: {np.sum(mask > 0)}")

    # ── PHASE 2: Blob Extraction ───────────────────────────────────────
    print("[2] Extracting blob candidates...")
    # UPDATE: Using the new refactored method name
    candidates = det._extract_blob_candidates(mask, img_h, img_w)
    print(f"    -> Candidates found: {len(candidates)}")

    # Draw blobs and search zones
    img_blobs = img.copy()
    for i, (sx1, sy1, sx2, sy2, (bx1, by1, bx2, by2)) in enumerate(candidates):
        # Inner blob (blue)
        cv2.rectangle(img_blobs, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
        # Search zone (green)
        cv2.rectangle(img_blobs, (sx1, sy1), (sx2, sy2), (0, 255, 0), 1)
        cv2.putText(
            img_blobs, 
            f"B{i}", 
            (bx1, by1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 0, 0), 
            1
        )
    cv2.imwrite(os.path.join(output_dir, "02_blobs_and_search_zones.png"), img_blobs)

    # ── PHASE 3: Hough Refinement per Candidate ────────────────────────
    print("[3] Refining each blob using HoughLinesP...\n")

    final_detections = []
    for idx, (sx1, sy1, sx2, sy2, blob) in enumerate(candidates):
        bx1, by1, bx2, by2 = blob
        print(
            f"   Blob {idx}: blob=({bx1},{by1})-({bx2},{by2}) | "
            f"search_zone=({sx1},{sy1})-({sx2},{sy2})"
        )

        roi = img[sy1:sy2, sx1:sx2]
        roi_h, roi_w = roi.shape[:2]

        # Preprocessing within ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, det._canny_low, det._canny_high)

        subdir = os.path.join(output_dir, f"03_blob_{idx:02d}")
        os.makedirs(subdir, exist_ok=True)

        cv2.imwrite(os.path.join(subdir, "a_roi_gray.png"), gray)
        cv2.imwrite(os.path.join(subdir, "b_canny_edges.png"), edges)

        # HoughLinesP computation
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

        # Blob in ROI coordinates
        bx1_roi = bx1 - sx1
        by1_roi = by1 - sy1
        bx2_roi = bx2 - sx1
        by2_roi = by2 - sy1

        # Draw blob in the ROI
        roi_with_blob = roi.copy()
        cv2.rectangle(roi_with_blob, (bx1_roi, by1_roi), (bx2_roi, by2_roi), (255, 0, 0), 2)
        cv2.putText(
            roi_with_blob, 
            "Blue Blob", 
            (bx1_roi, by1_roi-10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 0, 0), 
            1
        )

        # Process and draw Hough lines
        roi_with_lines = roi_with_blob.copy()
        h_count = v_count = 0
        h_above = []
        h_below = []
        v_left = []
        v_right = []

        if lines is not None:
            tol = np.deg2rad(det._angle_tol_deg)
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = abs(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1) + 1e-9))

                if angle <= tol:
                    y_mid = (ly1 + ly2) / 2.0
                    cv2.line(roi_with_lines, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                    h_count += 1
                    if y_mid < by1_roi - 5:
                        h_above.append(y_mid)
                    elif y_mid > by2_roi + 5:
                        h_below.append(y_mid)

                elif angle >= (np.pi / 2.0 - tol):
                    x_mid = (lx1 + lx2) / 2.0
                    cv2.line(roi_with_lines, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
                    v_count += 1
                    if x_mid < bx1_roi - 5:
                        v_left.append(x_mid)
                    elif x_mid > bx2_roi + 5:
                        v_right.append(x_mid)

        cv2.imwrite(os.path.join(subdir, "c_hough_lines.png"), roi_with_lines)
        print(
            f"      Lines: H={h_count} (above={len(h_above)}, below={len(h_below)}), "
            f"V={v_count} (left={len(v_left)}, right={len(v_right)})"
        )

        # Geometric refinement
        fx1, fy1, fx2, fy2 = det._refine_with_hough(img, sx1, sy1, sx2, sy2, blob)

        print(f"      Original Bbox: ({bx1},{by1})-({bx2},{by2}) = {bx2-bx1}x{by2-by1}")
        print(f"      Refined Bbox:  ({fx1},{fy1})-({fx2},{fy2}) = {fx2-fx1}x{fy2-fy1}")

        # Draw original vs refined comparison
        roi_comparison = roi.copy()
        cv2.rectangle(roi_comparison, (bx1_roi, by1_roi), (bx2_roi, by2_roi), (255, 0, 0), 2)
        
        fx1_roi, fy1_roi = fx1 - sx1, fy1 - sy1
        fx2_roi, fy2_roi = fx2 - sx1, fy2 - sy1
        cv2.rectangle(roi_comparison, (fx1_roi, fy1_roi), (fx2_roi, fy2_roi), (0, 255, 0), 2)
        
        cv2.putText(
            roi_comparison, 
            "Blue=Orig Green=Refined", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        cv2.imwrite(os.path.join(subdir, "d_bbox_comparison.png"), roi_comparison)

        # Evaluate score
        score = det._compute_score(img, fx1, fy1, fx2, fy2)
        print(f"      Score: {score:.3f}")

        if score >= det._score_threshold:
            final_detections.append([fx1, fy1, fx2, fy2, score])
            print("      [✓] ACCEPTED\n")
        else:
            print(f"      [✗] REJECTED (score < {det._score_threshold})\n")

    # ── PHASE 4: Final Result ──────────────────────────────────────────
    print(f"[4] Final Result: {len(final_detections)} detections\n")
    img_final = img.copy()
    for x1, y1, x2, y2, score in final_detections:
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_final, 
            f"{score:.2f}", 
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
    cv2.imwrite(os.path.join(output_dir, "04_final_detections.png"), img_final)

    print(f"[SUCCESS] Debug images exported to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug script for the Hybrid Detector.")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--output", type=str, default="debug_hybrid", help="Output directory for debug images")
    
    args = parser.parse_args()
    debug_detector_hybrid(args.image, args.output)