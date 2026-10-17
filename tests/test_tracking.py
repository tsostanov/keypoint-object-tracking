import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from demo import reference_scene, synthetic_frame
from main import build_parser, main, process_video, video_fps
from selection import choose_corners, load_selection, save_selection, validate_selection
from tracker import KeypointTracker, TrackerConfig, valid_quad


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setRNGSeed(2026)
        cls.reference, cls.corners = reference_scene()

    def test_perspective_loss_and_recovery(self):
        tracker = KeypointTracker(self.reference, self.corners)
        for index in (10, 25, 44, 50, 60, 75, 89):
            with self.subTest(frame=index):
                frame, truth = synthetic_frame(self.reference, self.corners, index)
                result = tracker.detect(frame)
                if truth is None:
                    self.assertEqual(result.status, "LOST")
                    self.assertIsNone(result.corners)
                else:
                    self.assertEqual(result.status, "TRACKING", result.reason)
                    error = np.linalg.norm(result.corners - truth, axis=1)
                    self.assertLess(float(error.max()), 5.0)

    def test_partial_outside_frame(self):
        target = self.corners - np.float32([160, 0])
        transform = cv2.getPerspectiveTransform(self.corners, target)
        frame = cv2.warpPerspective(self.reference, transform, (640, 480))
        result = KeypointTracker(self.reference, self.corners).detect(frame)
        self.assertEqual(result.status, "TRACKING", result.reason)
        self.assertLess(float(np.linalg.norm(result.corners - target, axis=1).max()), 5)

    def test_invalid_quadrilaterals(self):
        for corners in ([[0, 0], [200, 200], [200, 0], [0, 200]],
                        [[0, 0], [1, 0], [1, 1], [0, 1]],
                        [[0, 0], [10, 0], [20, 0], [30, 0]],
                        [[0, 0], [200, 0], [200, 200], [0, float("nan")]],
                        [[0, 0], [20000, 0], [20000, 20000], [0, 20000]]):
            with self.subTest(corners=corners):
                self.assertFalse(valid_quad(corners, self.reference.shape))

    def test_empty_reference(self):
        with self.assertRaisesRegex(ValueError, "ключевых точек"):
            KeypointTracker(np.zeros_like(self.reference), self.corners)

    def test_insufficient_matches(self):
        tracker = KeypointTracker(self.reference, self.corners)
        tracker.config.min_matches = 100000
        result = tracker.detect(self.reference)
        self.assertEqual(result.status, "LOST")
        self.assertEqual(result.reason, "few_matches")

    def test_homography_failure(self):
        tracker = KeypointTracker(self.reference, self.corners)
        with patch("tracker.cv2.findHomography", return_value=(None, None)):
            self.assertEqual(tracker.detect(self.reference).reason, "homography_failed")

    def test_inlier_rejection(self):
        tracker = KeypointTracker(self.reference, self.corners)
        with patch("tracker.cv2.findHomography", return_value=(np.eye(3), np.zeros((20, 1), np.uint8))):
            result = tracker.detect(self.reference)
            self.assertEqual(result.status, "LOST")
            self.assertEqual(result.reason, "few_inliers")

    def test_config_validation(self):
        for values in ({"ratio": 1.1}, {"ratio": float("nan")}, {"min_matches": 3},
                       {"scale_factor": 1}, {"ransac_threshold": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrackerConfig(**values).validate()


class SelectionTests(unittest.TestCase):
    def test_json_roundtrip_and_errors(self):
        frame, corners = reference_scene()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corners.json"
            save_selection(path, corners, frame.shape)
            np.testing.assert_array_equal(load_selection(path, frame.shape), corners)
            with self.assertRaises(ValueError):
                load_selection(path, (240, 320, 3))
            for content in ("{broken", "[]", '{"corners": null}',
                            '{"frame_size": [640, 480], "corners": [[1, 2]]}'):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_selection(path, frame.shape)
        with self.assertRaises(ValueError):
            validate_selection(corners - [200, 0], frame.shape)

    def test_selection_mouse_reset_confirm_and_cancel(self):
        frame, corners = reference_scene()
        callback = None

        def register(window, function):
            nonlocal callback
            callback = function

        def select(key):
            for x, y in corners.astype(int):
                callback(cv2.EVENT_LBUTTONDOWN, int(x), int(y), None, None)
            return key

        with contextlib.ExitStack() as stack:
            for name in ("namedWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch("selection.cv2." + name))
            stack.enter_context(patch("selection.cv2.setMouseCallback", side_effect=register))
            stack.enter_context(patch("selection.cv2.getWindowProperty", return_value=1))
            # Первая итерация рисует пустой выбор; вторая — точки; R сбрасывает их.
            with patch("selection.cv2.waitKey", side_effect=[ord("r"), 27]):
                self.assertIsNone(choose_corners(frame))
            with patch("selection.cv2.waitKey", side_effect=lambda delay: select(13)):
                np.testing.assert_array_equal(choose_corners(frame), corners)


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "input.avi"
        self.output = self.directory / "nested" / "tracked.mp4"
        self.selection = self.directory / "selection.json"
        reference, corners = reference_scene()
        save_selection(self.selection, corners, reference.shape)
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 12, (640, 480))
        self.assertTrue(writer.isOpened())
        try:
            for frame in (reference, synthetic_frame(reference, corners, 25)[0],
                          np.zeros_like(reference), synthetic_frame(reference, corners, 75)[0]):
                writer.write(frame)
        finally:
            writer.release()

    def arguments(self, *extra):
        return ["--input", str(self.source), "--output", str(self.output),
                "--selection", str(self.selection), "--no-preview", *extra]

    def test_pipeline_video_csv_summary(self):
        with patch("main.cv2.imshow", side_effect=AssertionError("GUI in headless mode")):
            summary = process_video(build_parser().parse_args(self.arguments()))
        self.assertEqual(summary["processed_frames"], 3)
        self.assertEqual(summary["detected_frames"], 2)
        self.assertEqual(summary["written_frames"], 4)
        self.assertAlmostEqual(summary["detection_rate"], 2 / 3)
        saved = json.loads(self.output.with_name("tracked_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, summary)
        with self.output.with_name("tracked_frames.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["status"] for row in rows], ["REFERENCE", "TRACKING", "LOST", "TRACKING"])
        self.assertAlmostEqual(float(rows[-1]["video_time_s"]), 3 / 12)
        capture = cv2.VideoCapture(str(self.output))
        try:
            self.assertTrue(capture.isOpened())
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12)
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (480, 640, 3))
                count += 1
            self.assertEqual(count, 4)
        finally:
            capture.release()

    def test_bad_inputs(self):
        cases = [self.arguments("--input", str(self.directory / "missing.mp4")),
                 self.arguments("--output", str(self.source)),
                 self.arguments("--output", str(self.selection)),
                 self.arguments("--ratio", "nan"), self.arguments("--label", "Объект"),
                 ["--input", str(self.source), "--output", str(self.output), "--no-preview"]]
        broken = self.directory / "broken.mp4"
        broken.write_text("not a video", encoding="utf-8")
        cases.append(self.arguments("--input", str(broken)))
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(args), 1)
                self.assertIn("Ошибка:", error.getvalue())
        self.assertFalse(self.output.exists())

    def test_writer_failure(self):
        with patch("main.cv2.VideoWriter") as writer:
            writer.return_value.isOpened.return_value = False
            with self.assertRaisesRegex(ValueError, "VideoWriter"):
                process_video(build_parser().parse_args(self.arguments()))
            writer.return_value.release.assert_called_once()

    def test_fps_fallback(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with contextlib.redirect_stderr(io.StringIO()) as warning:
                self.assertEqual(video_fps(fps), 25)
                self.assertIn("25 FPS", warning.getvalue())
        self.assertEqual(video_fps(29.97), 29.97)


if __name__ == "__main__":
    unittest.main()
