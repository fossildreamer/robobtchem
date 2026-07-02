from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import ColorFeatures, DiskLabel


@dataclass(frozen=True)
class FaceClassifierConfig:
    back_v_max: int = 82
    back_dark_ratio_min: float = 0.42
    front_v_min: int = 118
    front_bright_ratio_min: float = 0.16
    metallic_s_max: int = 105
    metallic_v_min: int = 135
    copper_h_min: int = 4
    copper_h_max: int = 35
    copper_s_min: int = 60
    copper_v_min: int = 75
    metallic_lab_b_max: int = 128


class FaceClassifier:
    """Classify a detected battery disk as front/back from color statistics."""

    def __init__(self, config: FaceClassifierConfig | None = None) -> None:
        self.config = config or FaceClassifierConfig()

    def classify(self, image_bgr: np.ndarray, center: tuple[float, float], radius: float) -> tuple[DiskLabel, float, ColorFeatures]:
        mask = self._disk_mask(image_bgr.shape[:2], center, radius)
        features = self.extract_features(image_bgr, mask)

        cfg = self.config
        back_score = self._clamp01(
            0.65 * (cfg.back_v_max - features.mean_v) / max(cfg.back_v_max, 1)
            + 0.35 * features.dark_ratio / max(cfg.back_dark_ratio_min, 0.01)
        )
        front_score = self._clamp01(
            0.35 * (features.mean_v - cfg.front_v_min) / max(255 - cfg.front_v_min, 1)
            + 0.35 * features.bright_ratio / max(cfg.front_bright_ratio_min, 0.01)
            + 0.20 * features.metallic_ratio
            + 0.10 * features.copper_ratio
        )

        if features.dark_ratio >= cfg.back_dark_ratio_min and features.mean_v <= cfg.back_v_max:
            return "back", max(0.5, back_score), features

        front_like = (
            features.mean_v >= cfg.front_v_min
            and (
                features.metallic_ratio >= 0.18
                or features.copper_ratio >= 0.10
            )
        )
        if front_like:
            return "front", max(0.5, front_score), features

        if back_score >= 0.58 and back_score > front_score:
            return "back", back_score, features
        if front_score >= 0.50 and front_score > back_score:
            return "front", front_score, features

        confidence = max(back_score, front_score, 0.05)
        return "unknown", min(confidence, 0.49), features

    def extract_features(self, image_bgr: np.ndarray, mask: np.ndarray) -> ColorFeatures:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        pixels_bgr = image_bgr[mask > 0]
        pixels_hsv = hsv[mask > 0]
        pixels_lab = lab[mask > 0]
        if pixels_bgr.size == 0:
            return ColorFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        mean_b, mean_g, mean_r = pixels_bgr.mean(axis=0)
        mean_h, mean_s, mean_v = pixels_hsv.mean(axis=0)
        v = pixels_hsv[:, 2]
        s = pixels_hsv[:, 1]
        h = pixels_hsv[:, 0]
        lab_b = pixels_lab[:, 2]
        cfg = self.config

        dark_ratio = float(np.mean(v <= cfg.back_v_max))
        bright_ratio = float(np.mean(v >= cfg.front_v_min))
        metallic_ratio = float(
            np.mean(
                (s <= cfg.metallic_s_max)
                & (v >= cfg.metallic_v_min)
                & (lab_b <= cfg.metallic_lab_b_max)
            )
        )
        copper_ratio = float(
            np.mean(
                (h >= cfg.copper_h_min)
                & (h <= cfg.copper_h_max)
                & (s >= cfg.copper_s_min)
                & (v >= cfg.copper_v_min)
            )
        )

        return ColorFeatures(
            float(mean_b),
            float(mean_g),
            float(mean_r),
            float(mean_h),
            float(mean_s),
            float(mean_v),
            float(v.std()),
            dark_ratio,
            bright_ratio,
            metallic_ratio,
            copper_ratio,
        )

    @staticmethod
    def _disk_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cx, cy = int(round(center[0])), int(round(center[1]))
        inner_radius = max(2, int(round(radius * 0.72)))
        cv2.circle(mask, (cx, cy), inner_radius, 255, -1)
        return mask

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))
