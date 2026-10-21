"""Поиск объекта по эталону ORB и сопровождение точек оптическим потоком."""

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
    optical_flow: bool = True

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
    method: str = "ORB"
    keypoints: np.ndarray | None = None
    matched_points: np.ndarray | None = None
    inlier_points: np.ndarray | None = None
    homography: np.ndarray | None = None

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
        self.previous_gray = None
        self.flow_points = None
        self.flow_origins = None
        if self.config.optical_flow:
            self._seed_flow(gray, self.corners, np.eye(3))

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        points = np.float32([point.pt for point in keypoints]).reshape(-1, 2)
        result = self._detect_reference(frame, keypoints, descriptors)
        if self.config.optical_flow:
            if result.corners is None:
                flow = self._detect_flow(gray, frame.shape)
                if flow is not None and flow.corners is not None:
                    result = flow
            if result.corners is not None:
                self._seed_flow(gray, result.corners, result.homography)
            else:
                # После потери не переносим старые точки через неизвестный участок.
                self.previous_gray = self.flow_points = self.flow_origins = None
        result.keypoints = points
        return result

    def _detect_reference(self, frame, keypoints, descriptors):
        result = Detection()
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
        result.matched_points = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        result.reason = "few_matches"
        if result.matches < self.config.min_matches:
            return result
        source = np.float32([self.keypoints[m.queryIdx].pt for m in matches])
        target = result.matched_points
        return self._estimate(source, target, frame.shape, result)

    def _estimate(self, source, target, frame_shape, result):
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
        result.inlier_points = target[mask.ravel().astype(bool)]
        # Не допускаем пересечения проективного горизонта внутри объекта.
        homogeneous = np.column_stack((self.corners, np.ones(4))) @ homography.T
        denominators = homogeneous[:, 2]
        result.reason = "invalid_geometry"
        if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
            return result
        projected = cv2.perspectiveTransform(self.corners[None], homography)[0]
        if not valid_quad(projected, frame_shape, self.config.min_area_ratio,
                          self.config.max_area_ratio):
            return result
        if result.method == "FLOW":
            # Точки в одном маленьком пятне не определяют устойчивую рамку всего объекта.
            support = abs(cv2.contourArea(cv2.convexHull(source[mask.ravel().astype(bool)])))
            if support < 0.03 * abs(cv2.contourArea(self.corners)):
                return result
        result.corners = projected
        result.homography = homography
        result.reason = "ok"
        return result

    def _seed_flow(self, gray, corners, homography):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        points = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.005,
                                        minDistance=3, mask=mask)
        self.previous_gray = gray
        self.flow_points = self.flow_origins = None
        if points is None or len(points) < self.config.min_matches:
            return
        try:
            inverse = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return
        origins = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
        if not np.isfinite(origins).all():
            return
        self.flow_points = points.reshape(-1, 2)
        self.flow_origins = origins

    def _detect_flow(self, gray, frame_shape):
        if self.previous_gray is None or self.flow_points is None:
            return None
        parameters = dict(winSize=(21, 21), maxLevel=3,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.flow_points.reshape(-1, 1, 2), None, **parameters)
        if moved is None or status is None:
            return None
        backwards, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, moved, None, **parameters)
        if backwards is None or back_status is None:
            return None
        target = moved.reshape(-1, 2)
        error = np.linalg.norm(backwards.reshape(-1, 2) - self.flow_points, axis=1)
        height, width = gray.shape
        good = ((status.ravel() > 0) & (back_status.ravel() > 0) & (error < 1.0)
                & np.isfinite(target).all(axis=1) & (target[:, 0] >= 0) & (target[:, 0] < width)
                & (target[:, 1] >= 0) & (target[:, 1] < height))
        target = target[good]
        result = Detection(matches=len(target), matched_points=target, method="FLOW", reason="few_matches")
        if result.matches < self.config.min_matches:
            return result
        return self._estimate(self.flow_origins[good], target, frame_shape, result)


def draw_detection(frame, detection, label="Object", reference=False, show_keypoints=True):
    image = frame.copy()
    color = (70, 220, 100) if detection.corners is not None else (70, 80, 245)
    if detection.corners is not None and detection.method == "FLOW":
        color = (245, 200, 70)
    point_groups = ((detection.keypoints, (230, 150, 70), 1),
                    (detection.matched_points, (40, 210, 245), 2),
                    (detection.inlier_points, (70, 220, 100), 3))
    if show_keypoints:
        for points, point_color, radius in point_groups:
            if points is not None:
                for x, y in np.rint(points).astype(np.int32):
                    cv2.circle(image, (int(x), int(y)), radius, point_color, -1, cv2.LINE_AA)
    if detection.corners is not None:
        points = np.rint(detection.corners).astype(np.int32)
        cv2.polylines(image, [points], True, color, 3, cv2.LINE_AA)
        x, y = points[0]
        position = (int(np.clip(x, 0, frame.shape[1] - 1)),
                    int(np.clip(y - 10, 70, max(70, frame.shape[0] - 10))))
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    status = "REFERENCE" if reference else detection.status
    cv2.rectangle(image, (0, 0), (frame.shape[1] - 1, 48), (20, 25, 30), -1)
    cv2.putText(image, f"{status} | {detection.method} | inliers {detection.inliers}/{detection.matches}",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    count = len(detection.keypoints) if detection.keypoints is not None else 0
    cv2.putText(image, f"features {count} | {detection.reason}",
                (12, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 225, 230), 1, cv2.LINE_AA)
    if show_keypoints:
        height, width = frame.shape[:2]
        cv2.rectangle(image, (0, height - 23), (width - 1, height - 1), (20, 25, 30), -1)
        for x, text, point_color in ((12, "all points", (230, 150, 70)),
                                     (180, "matches", (40, 210, 245)),
                                     (345, "RANSAC inliers", (70, 220, 100))):
            cv2.circle(image, (x, height - 11), 3, point_color, -1)
            cv2.putText(image, text, (x + 9, height - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, point_color, 1, cv2.LINE_AA)
    return image
