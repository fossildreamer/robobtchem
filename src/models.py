from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DiskLabel = Literal["front", "back", "unknown"]


LABEL_CN: dict[DiskLabel, str] = {
    "front": "正面",
    "back": "反面",
    "unknown": "未识别",
}


@dataclass(frozen=True)
class ColorFeatures:
    mean_b: float
    mean_g: float
    mean_r: float
    mean_h: float
    mean_s: float
    mean_v: float
    std_v: float
    dark_ratio: float
    bright_ratio: float
    metallic_ratio: float
    copper_ratio: float


@dataclass(frozen=True)
class DiskDetection:
    index: int
    center_x: float
    center_y: float
    radius: float
    label: DiskLabel
    label_cn: str
    confidence: float
    source: str
    features: ColorFeatures

    def to_csv_row(self, image_name: str) -> dict[str, object]:
        return {
            "image": image_name,
            "index": self.index,
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "radius": round(self.radius, 2),
            "label": self.label,
            "label_cn": self.label_cn,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "mean_b": round(self.features.mean_b, 2),
            "mean_g": round(self.features.mean_g, 2),
            "mean_r": round(self.features.mean_r, 2),
            "mean_h": round(self.features.mean_h, 2),
            "mean_s": round(self.features.mean_s, 2),
            "mean_v": round(self.features.mean_v, 2),
            "std_v": round(self.features.std_v, 2),
            "dark_ratio": round(self.features.dark_ratio, 4),
            "bright_ratio": round(self.features.bright_ratio, 4),
            "metallic_ratio": round(self.features.metallic_ratio, 4),
            "copper_ratio": round(self.features.copper_ratio, 4),
        }
