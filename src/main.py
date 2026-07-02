from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.detector import DiskDetector, DiskDetectorConfig
    from src.utils import annotate_image, collect_image_paths, ensure_dir, read_image, write_csv, write_image
else:
    from .detector import DiskDetector, DiskDetectorConfig
    from .utils import annotate_image, collect_image_paths, ensure_dir, read_image, write_csv, write_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Battery disk front/back recognition")
    parser.add_argument("--input", required=True, help="Image path, image folder, or camera index.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument(
        "--mode",
        choices=("image", "folder", "camera", "auto"),
        default="auto",
        help="Input mode. Default: auto.",
    )
    parser.add_argument("--save", dest="save", action="store_true", default=True, help="Save annotated images.")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save annotated images.")
    parser.add_argument("--debug", action="store_true", help="Reserved for debugging outputs.")
    parser.add_argument("--min-radius-ratio", type=float, default=0.022, help="Minimum disk radius ratio.")
    parser.add_argument("--max-radius-ratio", type=float, default=0.090, help="Maximum disk radius ratio.")
    parser.add_argument("--camera-frames", type=int, default=1, help="Frames to process in camera mode.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    save_images = args.save
    output_dir = ensure_dir(args.output)
    image_output_dir = ensure_dir(output_dir / "images")

    detector = DiskDetector(
        DiskDetectorConfig(
            min_radius_ratio=args.min_radius_ratio,
            max_radius_ratio=args.max_radius_ratio,
        )
    )

    if _resolve_mode(args.input, args.mode) == "camera":
        rows = process_camera(args.input, args.camera_frames, detector, image_output_dir, save_images)
    else:
        rows = process_files(args.input, detector, image_output_dir, save_images)

    csv_path = output_dir / "results.csv"
    write_csv(csv_path, rows)
    _print_summary(rows, csv_path)
    return 0


def process_files(
    input_path: str,
    detector: DiskDetector,
    image_output_dir: Path,
    save_images: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    image_paths = collect_image_paths(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_path}")

    for image_path in image_paths:
        image = read_image(image_path)
        detections = detector.detect(image)
        rows.extend(detection.to_csv_row(image_path.name) for detection in detections)

        if save_images:
            annotated = annotate_image(image, detections)
            output_name = f"{image_path.stem}_annotated{image_path.suffix or '.jpg'}"
            write_image(image_output_dir / output_name, annotated)

        print(f"{image_path.name}: detected={len(detections)}")
    return rows


def process_camera(
    camera_input: str,
    frame_count: int,
    detector: DiskDetector,
    image_output_dir: Path,
    save_images: bool,
) -> list[dict[str, object]]:
    camera_index = int(camera_input)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {camera_index}")

    rows: list[dict[str, object]] = []
    try:
        for frame_idx in range(max(frame_count, 1)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read camera frame.")
            image_name = f"camera_{camera_index}_{frame_idx + 1:03d}.jpg"
            detections = detector.detect(frame)
            rows.extend(detection.to_csv_row(image_name) for detection in detections)
            if save_images:
                annotated = annotate_image(frame, detections)
                write_image(image_output_dir / image_name, annotated)
            print(f"{image_name}: detected={len(detections)}")
    finally:
        cap.release()
    return rows


def _resolve_mode(input_value: str, mode: str) -> str:
    if mode != "auto":
        return mode
    path = Path(input_value)
    if path.exists():
        return "folder" if path.is_dir() else "image"
    if input_value.isdigit():
        return "camera"
    return "image"


def _print_summary(rows: list[dict[str, object]], csv_path: Path) -> None:
    counts = Counter(str(row["label"]) for row in rows)
    total = len(rows)
    print(
        "summary: "
        f"total={total}, "
        f"front={counts.get('front', 0)}, "
        f"back={counts.get('back', 0)}, "
        f"unknown={counts.get('unknown', 0)}"
    )
    print(f"results saved: {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
