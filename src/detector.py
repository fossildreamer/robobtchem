from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .classifier import FaceClassifier
from .models import DiskDetection, LABEL_CN


@dataclass(frozen=True)
class DiskDetectorConfig:
    min_radius_ratio: float = 0.022
    max_radius_ratio: float = 0.090
    min_area_ratio: float = 0.0009
    min_circularity: float = 0.48
    merge_center_ratio: float = 0.82
    min_circle_iou: float = 0.16
    hough_dp: float = 1.2
    hough_param1: float = 90
    hough_param2: float = 28


@dataclass(frozen=True)
class Candidate:
    center_x: float
    center_y: float
    radius: float
    source: str
    score: float


class DiskDetector:
    """Detect circular battery disks and classify each detected region."""

    def __init__(
        self,
        config: DiskDetectorConfig | None = None,
        classifier: FaceClassifier | None = None,
    ) -> None:
        self.config = config or DiskDetectorConfig()
        self.classifier = classifier or FaceClassifier()

    def detect(self, image_bgr: np.ndarray) -> list[DiskDetection]:
        candidates = self._detect_candidates(image_bgr)
        detections: list[DiskDetection] = []
        for idx, candidate in enumerate(candidates, start=1):
            label, confidence, features = self.classifier.classify(
                image_bgr,
                (candidate.center_x, candidate.center_y),
                candidate.radius,
            )
            detections.append(
                DiskDetection(
                    index=idx,
                    center_x=candidate.center_x,
                    center_y=candidate.center_y,
                    radius=candidate.radius,
                    label=label,
                    label_cn=LABEL_CN[label],
                    confidence=confidence,
                    source=candidate.source,
                    features=features,
                )
            )
        return detections

    def _detect_candidates(self, image_bgr: np.ndarray) -> list[Candidate]:
        height, width = image_bgr.shape[:2]
        min_dim = min(height, width)
        min_radius = max(8, int(round(min_dim * self.config.min_radius_ratio)))
        max_radius = max(min_radius + 2, int(round(min_dim * self.config.max_radius_ratio)))

        candidates: list[Candidate] = []
        candidates.extend(self._hough_candidates(image_bgr, min_radius, max_radius))
        candidates.extend(self._mask_candidates(image_bgr, min_radius, max_radius))
        candidates = self._filter_and_score_candidates(image_bgr, candidates)
        merged = self._merge_candidates(candidates)
        return sorted(merged, key=lambda item: (item.center_y, item.center_x))

    def _hough_candidates(self, image_bgr: np.ndarray, min_radius: int, max_radius: int) -> list[Candidate]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 1.8)
        min_dist = max(min_radius * 1.65, 12)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.config.hough_dp,
            minDist=min_dist,
            param1=self.config.hough_param1,
            param2=self.config.hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return []

        result: list[Candidate] = []
        for x, y, radius in np.round(circles[0]).astype(float):
            if self._inside_image(image_bgr.shape[:2], x, y, radius):
                result.append(Candidate(float(x), float(y), float(radius), "hough", 0.50))
        return result

    def _mask_candidates(self, image_bgr: np.ndarray, min_radius: int, max_radius: int) -> list[Candidate]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        h, w = image_bgr.shape[:2]
        min_area = h * w * self.config.min_area_ratio

        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 86]))
        l_chan, _, b_chan = cv2.split(lab)
        _, s_chan, v_chan = cv2.split(hsv)
        metal_mask = (
            (l_chan > 110)
            & (b_chan < 128)
            & (s_chan < 130)
            & (v_chan > 110)
        ).astype(np.uint8) * 255
        candidates: list[Candidate] = []
        candidates.extend(
            self._contour_candidates_from_mask(
                image_bgr,
                dark_mask,
                min_area,
                min_radius,
                max_radius,
                "mask_dark",
                kernel_size=11,
            )
        )
        candidates.extend(
            self._contour_candidates_from_mask(
                image_bgr,
                metal_mask,
                min_area,
                min_radius,
                max_radius,
                "mask_metal",
                kernel_size=15,
            )
        )
        return candidates

    def _contour_candidates_from_mask(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        min_area: float,
        min_radius: int,
        max_radius: int,
        source: str,
        kernel_size: int,
    ) -> list[Candidate]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Candidate] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.config.min_circularity:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if not (min_radius <= radius <= max_radius):
                continue
            if not self._inside_image(image_bgr.shape[:2], x, y, radius):
                continue
            candidates.append(Candidate(float(x), float(y), float(radius), source, 0.70 + float(circularity)))
        return candidates

    def _filter_and_score_candidates(self, image_bgr: np.ndarray, candidates: list[Candidate]) -> list[Candidate]:
        filtered: list[Candidate] = []
        for candidate in candidates:
            label, confidence, features = self.classifier.classify(
                image_bgr,
                (candidate.center_x, candidate.center_y),
                candidate.radius,
            )
            center_mask = self._circle_mask(
                image_bgr.shape[:2],
                candidate.center_x,
                candidate.center_y,
                max(3.0, candidate.radius * 0.24),
            )
            center_features = self.classifier.extract_features(image_bgr, center_mask)

            score = None
            if label == "back":
                if center_features.dark_ratio >= 0.55 or features.dark_ratio >= 0.78:
                    score = 0.55 * features.dark_ratio + 0.35 * center_features.dark_ratio + 0.10 * confidence
            elif label == "front":
                metallic_center = center_features.metallic_ratio >= 0.26
                copper_center = (
                    center_features.copper_ratio >= 0.22
                    and center_features.mean_s >= 60
                    and center_features.mean_r > center_features.mean_g + 3
                    and center_features.mean_g >= center_features.mean_b
                )
                silver_center = (
                    center_features.mean_v >= 145
                    and center_features.mean_s <= 115
                    and center_features.bright_ratio >= 0.58
                )
                white_paper_like = (
                    center_features.mean_v >= 205
                    and center_features.mean_s <= 20
                    and center_features.copper_ratio <= 0.04
                )
                if (metallic_center or copper_center or silver_center) and not white_paper_like:
                    material_ratio = max(features.metallic_ratio, features.copper_ratio)
                    center_ratio = max(center_features.metallic_ratio, center_features.copper_ratio)
                    score = 0.45 * material_ratio + 0.40 * center_ratio + 0.15 * confidence

            if score is None or score < 0.34:
                continue
            source_bonus = 0.0
            if candidate.source == "mask_metal":
                source_bonus = 0.45
            elif candidate.source == "mask_dark":
                source_bonus = 0.25
            filtered.append(
                Candidate(
                    candidate.center_x,
                    candidate.center_y,
                    candidate.radius,
                    candidate.source,
                    float(score + source_bonus + candidate.score * 0.03),
                )
            )
        return filtered

    def _merge_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
        merged: list[Candidate] = []
        for candidate in ordered:
            duplicate_index = self._find_duplicate(merged, candidate)
            if duplicate_index is None:
                merged.append(candidate)
                continue
            old = merged[duplicate_index]
            if candidate.score > old.score:
                merged[duplicate_index] = candidate
        return merged

    def _find_duplicate(self, existing: list[Candidate], candidate: Candidate) -> int | None:
        for idx, item in enumerate(existing):
            center_distance = float(np.hypot(item.center_x - candidate.center_x, item.center_y - candidate.center_y))
            radius_ref = max(item.radius, candidate.radius, 1.0)
            if center_distance <= radius_ref * self.config.merge_center_ratio:
                return idx
            if self._circle_iou(item, candidate) >= self.config.min_circle_iou:
                return idx
        return None

    @staticmethod
    def _inside_image(shape: tuple[int, int], x: float, y: float, radius: float) -> bool:
        height, width = shape
        margin = max(3.0, radius * 0.12)
        return margin <= x < width - margin and margin <= y < height - margin

    @staticmethod
    def _circle_mask(shape: tuple[int, int], x: float, y: float, radius: float) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius)), 255, -1)
        return mask

    @staticmethod
    def _circle_iou(first: Candidate, second: Candidate) -> float:
        distance = float(np.hypot(first.center_x - second.center_x, first.center_y - second.center_y))
        r1 = float(first.radius)
        r2 = float(second.radius)
        if distance >= r1 + r2:
            return 0.0
        if distance <= abs(r1 - r2):
            intersection = np.pi * min(r1, r2) ** 2
        else:
            a1 = r1**2 * np.arccos((distance**2 + r1**2 - r2**2) / (2 * distance * r1))
            a2 = r2**2 * np.arccos((distance**2 + r2**2 - r1**2) / (2 * distance * r2))
            a3 = 0.5 * np.sqrt(
                max(
                    0.0,
                    (-distance + r1 + r2)
                    * (distance + r1 - r2)
                    * (distance - r1 + r2)
                    * (distance + r1 + r2),
                )
            )
            intersection = a1 + a2 - a3
        union = np.pi * r1**2 + np.pi * r2**2 - intersection
        return float(intersection / union) if union > 0 else 0.0
