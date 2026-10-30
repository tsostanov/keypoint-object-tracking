"""Проверить полные MP4 и статистику; при наличии AVI сравнить исходники."""

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/videos"
NAMES = ("mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit", "test_cv")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_outputs():
    report = read_json(RESULTS / "playback_verification.json")
    require(len(report["videos"]) == len(NAMES), "Expected four output videos")
    hashes = set()
    evaluations = {item["output"]: item["verified_output_frames"]
                   for item in read_json(RESULTS / "evaluation.json")["videos"]}
    evaluations.update({item["path"]: item["verified_frames"]
                        for item in read_json(RESULTS / "test_cv_evaluation.json")["verified_videos"]})
    for name, record in zip(NAMES, report["videos"]):
        path = RESULTS / "playback" / f"{name}_tracked.mp4"
        relative = path.relative_to(ROOT).as_posix()
        require(record["output"] == relative, f"Wrong output path: {name}")
        summary = read_json(RESULTS / f"{name}_tracked_summary.json")
        require(Path(summary["output"]).as_posix() == relative, f"Wrong summary path: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest not in hashes, f"Duplicate output: {relative}")
        hashes.add(digest)
        with (RESULTS / f"{name}_tracked_frames.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        require([int(row["frame"]) for row in rows] == list(range(summary["written_frames"])),
                f"Missing or repeated CSV frames: {name}")
        detected = sum(row["status"] == "TRACKING" for row in rows)
        require(detected == summary["detected_frames"], f"Detection count mismatch: {name}")
        capture = cv2.VideoCapture(str(path))
        try:
            require(capture.isOpened(), f"Cannot open {relative}")
            require(abs(capture.get(cv2.CAP_PROP_FPS) - record["fps"]) < 0.01,
                    f"FPS mismatch: {name}")
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                require(list(frame.shape[1::-1]) == record["frame_size"], f"Frame size mismatch: {name}")
                count += 1
        finally:
            capture.release()
        require(count == summary["written_frames"] == len(rows) == record["verified_frames"],
                f"Decoded frame count mismatch: {name}")
        require(evaluations[relative] == count, f"Evaluation frame count mismatch: {name}")
        record.update(sha256=digest, bytes=path.stat().st_size,
                      csv_frames=len(rows), detected_frames=detected, availability="tracked_in_git")
        print(f"OK: {relative}: {count} frames, SHA-256 {digest}")
    report["date"] = datetime.now(timezone(timedelta(hours=3))).date().isoformat()
    write_json(RESULTS / "playback_verification.json", report)


def compare_sources():
    names = NAMES[:3]
    paths = [ROOT / "data/videos" / f"{name}.avi" for name in names]
    pairs = list(combinations(range(len(names)), 2))
    intervals = {pair: [] for pair in pairs}
    counts = {pair: 0 for pair in pairs}
    with ExitStack() as stack:
        captures = []
        for path in paths:
            require(path.is_file(), f"Source video is available only locally: {path}")
            capture = cv2.VideoCapture(str(path))
            stack.callback(capture.release)
            require(capture.isOpened(), f"Cannot open {path}")
            captures.append(capture)
        frame_count = 0
        while True:
            reads = [capture.read() for capture in captures]
            if not any(ok for ok, _ in reads):
                break
            require(all(ok for ok, _ in reads), "Source videos have different frame counts")
            for pair in pairs:
                if not np.array_equal(reads[pair[0]][1], reads[pair[1]][1]):
                    counts[pair] += 1
                    spans = intervals[pair]
                    if spans and spans[-1][1] == frame_count - 1:
                        spans[-1][1] = frame_count
                    else:
                        spans.append([frame_count, frame_count])
            frame_count += 1
    require(frame_count > 0, "Source videos have no readable frames")
    report = {
        "comparison": "Exact equality of decoded BGR pixels; zero-based inclusive frame intervals",
        "opencv": cv2.__version__,
        "source_availability": "local_only",
        "sources": [{"path": path.relative_to(ROOT).as_posix(),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "compared_frames": frame_count,
        "pairs": [{"first": names[first], "second": names[second],
                   "identical_frames": frame_count - counts[first, second],
                   "different_frames": counts[first, second],
                   "different_intervals": intervals[first, second]} for first, second in pairs],
    }
    write_json(RESULTS / "source_comparison.json", report)
    print(json.dumps(report["pairs"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-sources", action="store_true", help="Also compare local source AVI files")
    args = parser.parse_args()
    verify_outputs()
    if args.compare_sources:
        compare_sources()


if __name__ == "__main__":
    main()
