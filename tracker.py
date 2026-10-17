"""Поиск плоского объекта относительно неизменного эталона первого кадра."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrackerConfig:
    nfeatures: int = 2500
    scale_factor: float = 1.2
    nlevels: int = 8
    fast_threshold: int = 15
    ratio: float = 0.75
    min_matches: int = 12
    min_inliers: int = 10
    min_inlier_ratio: float = 0.45
    ransac_threshold: float = 3.0
    min_area_ratio: float = 0.001
    max_area_ratio: float = 1.5

    def validate(self):
        values = vars(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Параметры трекера должны быть конечными числами.")
        if self.nfeatures < 4 or self.nlevels < 1 or self.fast_threshold < 0:
            raise ValueError("Некорректные параметры ORB.")
        if self.scale_factor <= 1 or not 0 < self.ratio < 1:
            raise ValueError("scale-factor должен быть > 1, ratio — между 0 и 1.")
        if min(self.min_matches, self.min_inliers) < 4:
            raise ValueError("Для гомографии требуется не менее 4 соответствий.")
        if not 0 < self.min_inlier_ratio <= 1 or self.ransac_threshold <= 0:
            raise ValueError("Некорректные пороги RANSAC/inlier.")
        if not 0 < self.min_area_ratio < self.max_area_ratio:
            raise ValueError("Требуется 0 < min-area-ratio < max-area-ratio.")


@dataclass
class Detection:
    corners: np.ndarray | None = None
    matches: int = 0
    inliers: int = 0
    reason: str = "no_descriptors"

    @property
    def status(self):
        return "TRACKING" if self.corners is not None else "LOST"

    @property
    def inlier_ratio(self):
        return self.inliers / self.matches if self.matches else 0.0


def valid_quad(corners, frame_shape, min_area_ratio=0.001, max_area_ratio=1.5):
    """Выпуклая рамка с разумной площадью; допускается частичный выход за кадр."""
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    height, width = frame_shape[:2]
    # Ограничение также защищает преобразование координат в int32 при рисовании.
    if np.max(np.abs(points)) > 10 * max(height, width):
        return False
    if not cv2.isContourConvex(points):
        return False
    area_ratio = abs(cv2.contourArea(points)) / (width * height)
    if not min_area_ratio <= area_ratio <= max_area_ratio:
        return False
    frame = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    visible_area, _ = cv2.intersectConvexConvex(points, frame)
    return visible_area > 0


class KeypointTracker:
    def __init__(self, first_frame, corners, config=None):
        self.config = config or TrackerConfig()
        self.config.validate()
        self.corners = np.asarray(corners, dtype=np.float32)
        if not valid_quad(self.corners, first_frame.shape):
            raise ValueError("Недопустимая геометрия эталонного объекта.")
        self.orb = cv2.ORB_create(
            nfeatures=self.config.nfeatures,
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.nlevels,
            fastThreshold=self.config.fast_threshold,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(self.corners).astype(np.int32), 255)
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # Маска отсекает фон, а координаты остаются в системе полного кадра.
        self.keypoints, self.descriptors = self.orb.detectAndCompute(gray, mask)
        if self.descriptors is None or len(self.descriptors) < self.config.min_matches:
            raise ValueError("На объекте недостаточно ключевых точек. Выберите чёткий объект с деталями.")

    def detect(self, frame):
        result = Detection()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return result
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        candidates = [pair[0] for pair in pairs if len(pair) == 2
                      and pair[0].distance < self.config.ratio * pair[1].distance]
        # Одной точке текущего кадра соответствует только одна точка эталона.
        matches, used = [], set()
        for match in sorted(candidates, key=lambda item: item.distance):
            if match.trainIdx not in used:
                matches.append(match)
                used.add(match.trainIdx)
        result.matches = len(matches)
        result.reason = "few_matches"
        if result.matches < self.config.min_matches:
            return result
        source = np.float32([self.keypoints[m.queryIdx].pt for m in matches])
        target = np.float32([keypoints[m.trainIdx].pt for m in matches])
        result.reason = "homography_failed"
        try:
            homography, mask = cv2.findHomography(
                source, target, cv2.RANSAC, self.config.ransac_threshold
            )
        except cv2.error:
            return result
        if homography is None or mask is None or not np.isfinite(homography).all():
            return result
        result.inliers = int(mask.sum())
        result.reason = "few_inliers"
        if (result.inliers < self.config.min_inliers
                or result.inlier_ratio < self.config.min_inlier_ratio):
            return result
        # Не допускаем пересечения проективного горизонта внутри объекта.
        homogeneous = np.column_stack((self.corners, np.ones(4))) @ homography.T
        denominators = homogeneous[:, 2]
        result.reason = "invalid_geometry"
        if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
            return result
        projected = cv2.perspectiveTransform(self.corners[None], homography)[0]
        if not valid_quad(projected, frame.shape, self.config.min_area_ratio,
                          self.config.max_area_ratio):
            return result
        result.corners = projected
        result.reason = "ok"
        return result


def draw_detection(frame, detection, label="Object", reference=False):
    image = frame.copy()
    color = (70, 220, 100) if detection.corners is not None else (70, 80, 245)
    if detection.corners is not None:
        points = np.rint(detection.corners).astype(np.int32)
        cv2.polylines(image, [points], True, color, 3, cv2.LINE_AA)
        x, y = points[0]
        position = (int(np.clip(x, 0, frame.shape[1] - 1)),
                    int(np.clip(y - 10, 70, max(70, frame.shape[0] - 10))))
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    status = "REFERENCE" if reference else detection.status
    cv2.rectangle(image, (0, 0), (min(frame.shape[1] - 1, 520), 48), (20, 25, 30), -1)
    cv2.putText(image, f"{status} | inliers {detection.inliers}/{detection.matches}",
                (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return image
