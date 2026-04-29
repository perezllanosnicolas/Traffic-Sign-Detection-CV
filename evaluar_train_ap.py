import argparse
from evaluar_resultados import load_results_file, precision_recall_curve, draw_PR_fast


def compute_ap(detections_file, dataset_path, iou):
    _, det_dbboxes = load_results_file(detections_file, dataset_path)
    _, gt_dbboxes = load_results_file(f"{dataset_path}/gt.txt", dataset_path)

    tp, fp, _, tot = precision_recall_curve(
        gt_dbboxes,
        det_dbboxes,
        show=False,
        ovr=iou,
    )
    _, _, ap = draw_PR_fast(tp, fp, tot, show=False)
    return ap


def main():
    parser = argparse.ArgumentParser(
        description="Calcula AP en train/test para IoU 0.5 y 0.7"
    )
    parser.add_argument(
        "--detections_file",
        default="./resultado.txt",
        help="Ruta al fichero de detecciones",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Ruta al directorio con imagenes y gt.txt (train_detection o test_detection)",
    )

    args = parser.parse_args()

    ap05 = compute_ap(args.detections_file, args.dataset_path, iou=0.5)
    ap07 = compute_ap(args.detections_file, args.dataset_path, iou=0.7)

    print(f"Dataset: {args.dataset_path}")
    print(f"Detections: {args.detections_file}")
    print(f"AP@0.5: {ap05 * 100:.2f}")
    print(f"AP@0.7: {ap07 * 100:.2f}")


if __name__ == "__main__":
    main()
