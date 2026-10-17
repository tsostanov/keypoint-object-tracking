"""Выбор четырёх углов и воспроизводимое сохранение в JSON."""

import json
from pathlib import Path

import cv2
import numpy as np

from tracker import valid_quad


def validate_selection(corners, frame_shape):
    try:
        points = np.asarray(corners, dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError("Углы должны быть массивом из четырёх пар чисел.") from error
    if not valid_quad(points, frame_shape, max_area_ratio=1.0):
        raise ValueError("Нужны 4 различных угла по контуру выпуклого объекта без пересечений.")
    height, width = frame_shape[:2]
    if (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
        raise ValueError("Выбранные углы должны находиться внутри первого кадра.")
    return points


def save_selection(path, corners, frame_shape):
    points = validate_selection(corners, frame_shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"frame_size": [frame_shape[1], frame_shape[0]], "corners": points.tolist()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_selection(path, frame_shape):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Некорректный JSON выбора: {error}") from error
    if not isinstance(payload, dict) or "corners" not in payload:
        raise ValueError("JSON должен содержать объект с полем corners.")
    if payload.get("frame_size") != [frame_shape[1], frame_shape[0]]:
        raise ValueError("frame_size в JSON не соответствует размерам видео.")
    return validate_selection(payload["corners"], frame_shape)


def choose_corners(frame):
    window = "Select object: Enter confirm | R reset | Q/Esc cancel"
    # Масштабируем только отображение; JSON и трекер используют исходные координаты.
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 800 / height)
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    scale_xy = np.array([display_size[0] / width, display_size[1] / height])
    points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((np.array([x, y]) / scale_xy).tolist())

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(window, on_mouse)
        while True:
            image = cv2.resize(frame, display_size)
            display_points = np.rint(np.array(points) * scale_xy).astype(np.int32)
            if len(points):
                cv2.polylines(image, [display_points], len(points) == 4, (0, 220, 255), 2)
                for index, point in enumerate(display_points):
                    cv2.circle(image, tuple(point), 5, (0, 220, 255), -1)
                    cv2.putText(image, str(index + 1), tuple(point + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.imshow(window, image)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13):
                try:
                    return validate_selection(points, frame.shape)
                except ValueError as error:
                    print(f"Выбор не принят: {error}")
    finally:
        cv2.destroyWindow(window)
