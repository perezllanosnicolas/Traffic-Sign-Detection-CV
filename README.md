# Highway Traffic Sign Detection | Classical Computer Vision

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-blue.svg)

> **Note:** This repository is **Phase 1** of an End-to-End Traffic Sign Recognition system. Once the panels are detected and cropped, they are passed to our OCR pipeline. Check out the OCR implementation in my [Traffic-Sign-OCR-System repository](https://github.com/perezllanosnicolas/Traffic-Sign-OCR-System).

## Overview
This repository contains a robust Computer Vision pipeline designed to detect highway information panels in real-world driving scenarios. 

Unlike modern approaches that rely entirely on heavy Deep Learning models, this project demonstrates the power of **Classical Computer Vision** algorithms. By combining Maximally Stable Extremal Regions (MSER), strict geometric filtering, and morphological color correlation, the system achieves high precision in real-time without the need for GPU inference.

## Core Architecture
The detection pipeline is modularized and heavily optimized for varying lighting conditions, partial occlusions, and perspective distortions. The primary engine (`PanelDetectorMSER`) operates through the following steps:

1. **High-Contrast Extraction:** Uses MSER to detect highly stable regions (such as white text on a dark background).
2. **Geometric Constraints:** Filters regions based on bounding box solidity, minimum area, and strict aspect ratios to discard vehicles and roadside artifacts.
3. **Asymmetric Padding:** Expands the bounding box (5% horizontal, 8% vertical) to ensure the white outer border is captured within the Region of Interest (ROI).
4. **Color & Topology Validation:** Normalizes the ROI to a 40x80 matrix, extracts the saturated blue mask in the HSV color space, and computes a correlation score against an ideal mathematical mask (which heavily penalizes border over-segmentation).
5. **Non-Maximum Suppression (NMS):** Resolves overlapping predictions using custom Intersection over Union (IoU) and Intersection over Minimum (IoM) logic.

>  **Deep Dive:** Want to know why pure Hough Transforms fail globally but excel locally? Check out our detailed [Technical Evaluation Report](docs/technical_report.md) comparing MSER, Color-First, and Hybrid approaches.

## Performance & Results
The pipeline was evaluated against an established Baseline across a diverse test dataset. The custom implementation significantly outperformed the Baseline, especially under strict overlap requirements (IoU > 0.7).

* **Average Precision (IoU > 0.5):** 83.4% (vs 68.7% Baseline)
* **Average Precision (IoU > 0.7):** 81.9% (vs 60.9% Baseline)

<p align="center">
  <img src="docs/assets/pr_curve_iou_70.png" width="600" alt="Precision-Recall Curve IoU 0.7">
  <img src="docs/assets/pr_curve_iou_50.png" width="600" alt="Precision-Recall Curve IoU 0.5">
</p>


## Installation & Usage

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/tu-usuario/Traffic-Sign-Detection-CV.git](https://github.com/tu-usuario/Traffic-Sign-Detection-CV.git)
cd Traffic-Sign-Detection-CV
pip install -r requirements.txt
```

**2. Run the detecion pipeline on a test directory:**
```bash
python main.py --test_path data/test_detection --detector mser --output_dir results/images --output_txt results/detections.txt
```

**3.Evaluate the model:**
```bash
python -m scripts.evaluar_resultados --test_path data/test_detection --predictions_file results/detections.txt --baseline_file data/baseline_detections.txt
```

## Debugging & Tools
The repository includes a suite of visualization tools for algorithmic debugging. You can generate step-by-step visual pipelines for any given image:

```bash
python -m scripts.debug_hybrid --image data/test_detection/00005.png --output docs/assets/debug
```
## Authors
This project was co-developed as part of a joint Computer Engineering research initiative by:
- Nicolás Pérez  - [www.linkedin.com/in/nicolasperezllanos]      | [https://github.com/perezllanosnicolas]
- Rubén Pisonero - [https://www.linkedin.com/in/ruben-pisonero/] | [https://github.com/rpisoner]
