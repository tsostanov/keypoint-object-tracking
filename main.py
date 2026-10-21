"""Консольное приложение: видео → ORB + LK + RANSAC → видео, CSV и JSON."""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import cv2

from selection import choose_corners, load_selection, save_selection
from tracker import Detection, KeypointTracker, TrackerConfig, draw_detection


def build_parser():
    parser = argparse.ArgumentParser(description="Трекинг плоского объекта по ключевым точкам ORB.")
    parser.add_argument("--input", type=Path, required=True, help="Исходное видео")
    parser.add_argument("--output", type=Path, required=True, help="Результат .mp4 или .avi")
    parser.add_argument("--label", default="Object", help="Подпись латиницей")
    parser.add_argument("--selection", type=Path, help="JSON с четырьмя углами и frame_size")
    parser.add_argument("--no-preview", action="store_true", help="Без окон; нужен --selection")
    parser.add_argument("--show-keypoints", action=argparse.BooleanOptionalAction, default=True,
                        help="Показывать все ключевые точки, совпадения и inlier (по умолчанию включено)")
    descriptions = {
        "nfeatures": "Максимум точек ORB", "scale_factor": "Масштаб между уровнями пирамиды",
        "nlevels": "Число уровней пирамиды", "fast_threshold": "Порог FAST",
        "ratio": "Порог отношения расстояний KNN", "min_matches": "Минимум совпадений",
        "min_inliers": "Минимум inlier", "min_inlier_ratio": "Минимальная доля inlier",
        "ransac_threshold": "Порог ошибки репроекции, пиксели",
        "min_area_ratio": "Минимальная площадь / площадь кадра",
        "max_area_ratio": "Максимальная площадь / площадь кадра",
        "optical_flow": "Сопровождение точек между кадрами при недостатке совпадений ORB",
    }
    for name, default in asdict(TrackerConfig()).items():
        options = ({"action": argparse.BooleanOptionalAction} if isinstance(default, bool)
                   else {"type": type(default)})
        parser.add_argument("--" + name.replace("_", "-"), default=default, **options,
                            help=f"{descriptions[name]} (по умолчанию {default})")
    return parser


def same_file(first, second):
    return (first.resolve() == second.resolve()
            or (first.exists() and second.exists() and first.samefile(second)))


def output_paths(args):
    output = args.output
    paths = {
        "video": output,
        "csv": output.with_name(output.stem + "_frames.csv"),
        "summary": output.with_name(output.stem + "_summary.json"),
        "selection": output.with_name(output.stem.removesuffix("_tracked") + "_selection.json"),
    }
    destinations = [paths[name] for name in ("video", "csv", "summary")]
    if args.selection is None:
        destinations.append(paths["selection"])
    sources = [args.input] + ([args.selection] if args.selection else [])
    for index, destination in enumerate(destinations):
        if any(same_file(destination, source) for source in sources + destinations[:index]):
            raise ValueError("Пути входных файлов и выходных результатов не должны совпадать.")
    return paths


def video_fps(value):
    if not math.isfinite(value) or not 0 < value <= 1000:
        print("Предупреждение: некорректный FPS; используется 25 FPS.", file=sys.stderr)
        return 25.0
    return value


def process_video(args):
    if args.no_preview and args.selection is None:
        raise ValueError("Для --no-preview укажите --selection с сохранёнными углами.")
    if not args.label or any(ord(char) < 32 or ord(char) > 126 for char in args.label):
        raise ValueError("--label должен содержать печатные символы ASCII (латиница).")
    config = TrackerConfig(**{name: getattr(args, name) for name in asdict(TrackerConfig())})
    config.validate()
    paths = output_paths(args)
    if args.output.suffix.lower() not in (".mp4", ".avi"):
        raise ValueError("Выходное видео должно иметь расширение .mp4 или .avi.")
    if not args.input.is_file():
        raise ValueError(f"Входной файл не найден: {args.input}")
    capture = cv2.VideoCapture(str(args.input))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Не удалось открыть видео: {args.input}")
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise ValueError("Не удалось прочитать первый кадр видео.")
        height, width = first_frame.shape[:2]
        # FFmpeg/OpenCV может молча обрезать нечётный размер на один пиксель.
        if width % 2 or height % 2:
            raise ValueError("Для записи без обрезки нужны чётные ширина и высота видео.")
        fps = video_fps(capture.get(cv2.CAP_PROP_FPS))
        corners = (load_selection(args.selection, first_frame.shape) if args.selection
                   else choose_corners(first_frame))
        if corners is None:
            print("Выбор отменён. Обработка не начата.")
            return None
        tracker = KeypointTracker(first_frame, corners, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.selection is None:
            save_selection(paths["selection"], corners, first_frame.shape)
            print(f"Углы сохранены: {paths['selection']}")
        codec = "mp4v" if args.output.suffix.lower() == ".mp4" else "MJPG"
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Не удалось открыть VideoWriter: {args.output} (кодек {codec})")
        processed = detected = 0
        tracking_methods = {"ORB": 0, "FLOW": 0}
        processing_seconds = 0.0
        stopped_early = False
        preview_window = "Keypoint tracking | Q/Esc: stop"
        if not args.no_preview:
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        started = time.perf_counter()
        with paths["csv"].open("w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["frame", "video_time_s", "status", "matches", "inliers",
                                 "inlier_ratio", "processing_ms", "reason", "method", "keypoints"])
            reference_points = [point.pt for point in tracker.keypoints]
            reference_detection = Detection(corners=corners, keypoints=reference_points,
                                            method="REFERENCE", reason="reference")
            reference = draw_detection(first_frame, reference_detection, args.label,
                                       reference=True, show_keypoints=args.show_keypoints)
            writer.write(reference)
            csv_writer.writerow([0, "0.000000", "REFERENCE", 0, 0, "0.000000", "0.000", "reference",
                                 "REFERENCE", len(reference_points)])
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape != first_frame.shape:
                    raise ValueError("Размер кадра изменился внутри видео.")
                tick = time.perf_counter()
                detection = tracker.detect(frame)
                image = draw_detection(frame, detection, args.label, show_keypoints=args.show_keypoints)
                elapsed = time.perf_counter() - tick
                processed += 1
                detected += detection.status == "TRACKING"
                if detection.status == "TRACKING":
                    tracking_methods[detection.method] += 1
                processing_seconds += elapsed
                writer.write(image)
                csv_writer.writerow([
                    processed, f"{processed / fps:.6f}", detection.status, detection.matches,
                    detection.inliers, f"{detection.inlier_ratio:.6f}", f"{elapsed * 1000:.3f}",
                    detection.reason, detection.method, len(detection.keypoints),
                ])
                if processed % 100 == 0:
                    print(f"Кадров: {processed}; обнаружено: {detected}; {detection.status}", flush=True)
                if not args.no_preview:
                    cv2.imshow(preview_window, image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                        stopped_early = True
                        break
        writer.release()
        writer = None
        wall_seconds = time.perf_counter() - started
        summary = {
            "input": str(args.input), "output": str(args.output),
            "frame_size": [width, height], "video_fps": fps,
            "written_frames": processed + 1, "processed_frames": processed,
            "detected_frames": detected, "detection_rate": detected / processed if processed else 0.0,
            "mean_processing_ms": 1000 * processing_seconds / processed if processed else 0.0,
            "processing_fps": processed / processing_seconds if processing_seconds else 0.0,
            "pipeline_fps": processed / wall_seconds if wall_seconds else 0.0,
            "stopped_early": stopped_early, "config": asdict(config),
            "show_keypoints": args.show_keypoints, "tracking_methods": tracking_methods,
        }
        paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Готово: {processed} кадров после эталона; обнаружено {detected} "
              f"({summary['detection_rate']:.1%}); {summary['processing_fps']:.1f} кадр/с.")
        print(f"Видео: {args.output}\nCSV: {paths['csv']}\nJSON: {paths['summary']}")
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        process_video(args)
    except (ValueError, OSError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Обработка прервана. Частичное видео и CSV могут быть сохранены.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
