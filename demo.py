"""Воспроизводимый синтетический ролик. Не заменяет видео из задания."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from main import build_parser, process_video
from selection import save_selection


def reference_scene(seed=2026):
    rng = np.random.default_rng(seed)
    frame = np.full((480, 640, 3), (38, 32, 28), dtype=np.uint8)
    corners = np.float32([[120, 90], [520, 90], [520, 390], [120, 390]])
    cv2.rectangle(frame, (120, 90), (520, 390), (225, 235, 240), -1)
    for _ in range(240):
        x, y = rng.integers([140, 110], [500, 370])
        color = tuple(int(value) for value in rng.integers(20, 200, 3))
        cv2.circle(frame, (int(x), int(y)), int(rng.integers(2, 9)), color, -1)
    cv2.rectangle(frame, (165, 185), (475, 287), (32, 44, 65), -1)
    cv2.putText(frame, "KEYPOINT", (180, 229), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (240, 245, 255), 3)
    cv2.putText(frame, "LAB 02 / ORB", (188, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 240), 2)
    return frame, corners


def synthetic_frame(reference, corners, index):
    if 45 <= index < 60:
        return np.full_like(reference, (38, 32, 28)), None
    phase = index / 89
    angle = np.deg2rad(18 * np.sin(phase * 2 * np.pi))
    scale = 1 - 0.22 * np.sin(phase * np.pi)
    rotation = scale * np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    target = (corners - [320, 240]) @ rotation.T + [320 + 65 * np.sin(phase * 4), 240]
    target += np.array([[0, 0], [-24 * phase, 18 * phase], [15 * phase, 0], [0, -20 * phase]])
    target = target.astype(np.float32)
    transform = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(reference, transform, (640, 480), borderValue=(38, 32, 28)), target


def create_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "synthetic.mp4"
    selection = directory / "synthetic_selection.json"
    reference, corners = reference_scene()
    save_selection(selection, corners, reference.shape)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
    if not writer.isOpened():
        writer.release()
        raise ValueError("Не удалось создать демонстрационное видео.")
    truth = []
    try:
        for index in range(90):
            frame, target = (reference, corners) if index == 0 else synthetic_frame(reference, corners, index)
            writer.write(frame)
            truth.append({"frame": index, "corners": target.tolist() if target is not None else None})
    finally:
        writer.release()
    (directory / "synthetic_ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    return source, selection


def make_contact_sheet(video, destination):
    capture = cv2.VideoCapture(str(video))
    images = []
    try:
        for index in (0, 25, 50, 75):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Не читается демонстрационный кадр {index}.")
            cv2.putText(frame, f"SYNTHETIC DEMO / frame {index}", (12, 463),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
            images.append(frame)
    finally:
        capture.release()
    sheet = np.vstack((np.hstack(images[:2]), np.hstack(images[2:])))
    if not cv2.imwrite(str(destination), sheet):
        raise ValueError("Не удалось записать иллюстрацию.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/demo"))
    args = parser.parse_args()
    source, selection = create_demo(args.directory)
    output = args.directory / "synthetic_tracked.mp4"
    process_video(build_parser().parse_args([
        "--input", str(source), "--output", str(output), "--selection", str(selection),
        "--no-preview", "--label", "Demo card",
    ]))
    make_contact_sheet(output, args.directory / "preview.jpg")


if __name__ == "__main__":
    main()
