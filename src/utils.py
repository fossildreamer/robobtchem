from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .models import DiskDetection


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def collect_image_paths(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {path}")


def read_image(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    data = np.fromfile(str(file_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {file_path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    suffix = file_path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {file_path}")
    encoded.tofile(str(file_path))


def annotate_image(image_bgr: np.ndarray, detections: list[DiskDetection]) -> np.ndarray:
    annotated = image_bgr.copy()
    for detection in detections:
        center = (int(round(detection.center_x)), int(round(detection.center_y)))
        radius = int(round(detection.radius))
        color = _label_color(detection.label)
        cv2.circle(annotated, center, radius, color, 3)
        cv2.circle(annotated, center, 4, color, -1)

        text = f"{detection.index}:{detection.label} {detection.confidence:.2f}"
        text_x = max(0, center[0] - radius)
        text_y = max(24, center[1] - radius - 8)
        cv2.putText(
            annotated,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    fieldnames = [
        "image",
        "index",
        "center_x",
        "center_y",
        "radius",
        "label",
        "label_cn",
        "confidence",
        "source",
        "mean_b",
        "mean_g",
        "mean_r",
        "mean_h",
        "mean_s",
        "mean_v",
        "std_v",
        "dark_ratio",
        "bright_ratio",
        "metallic_ratio",
        "copper_ratio",
    ]
    with file_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _label_color(label: str) -> tuple[int, int, int]:
    if label == "front":
        return (0, 180, 0)
    if label == "back":
        return (0, 0, 255)
    return (0, 210, 255)
