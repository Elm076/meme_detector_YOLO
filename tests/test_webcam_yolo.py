import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from webcam_yolo import (
    OverlayHoldTracker,
    OverlayObservation,
    build_parser,
    class_name,
    expand_box,
    load_assets,
    normalize_label,
    overlay_asset,
    resolve_checkpoint,
)


class LabelTests(unittest.TestCase):
    def test_expression_confidence_defaults_to_70_percent(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.expression_confidence, 0.70)
        self.assertEqual(args.asset_hold_seconds, 1.0)

    def test_normalizes_model_and_asset_names(self):
        self.assertEqual(normalize_label("Happy"), "happy")
        self.assertEqual(normalize_label("happy-dog"), "happy")
        self.assertEqual(normalize_label("Very Confused"), "very_confused")

    def test_reads_list_and_dictionary_class_names(self):
        self.assertEqual(class_name(["angry", "happy"], 1), "happy")
        self.assertEqual(class_name({0: "angry", 1: "happy"}, 1), "happy")


class AssetTests(unittest.TestCase):
    def test_loads_assets_by_normalized_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (8, 8), "red").save(root / "Happy Dog.png")
            assets = load_assets(root, cv2)
            self.assertEqual(set(assets), {"happy"})

    def test_alpha_overlay_blends_transparent_asset(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        asset = np.zeros((2, 2, 4), dtype=np.uint8)
        asset[:, :, 2] = 200
        asset[:, :, 3] = 128
        overlay_asset(frame, asset, (1, 1, 3, 3), cv2)
        self.assertGreater(int(frame[1, 1, 2]), 90)
        self.assertLess(int(frame[1, 1, 2]), 110)


class GeometryTests(unittest.TestCase):
    def test_expands_and_clips_box(self):
        self.assertEqual(expand_box((10, 10, 20, 20), 100, 100, 2.0), (5, 5, 25, 25))
        self.assertEqual(expand_box((0, 0, 10, 10), 100, 100, 2.0), (0, 0, 15, 15))


class OverlayHoldTests(unittest.TestCase):
    def test_holds_asset_for_one_second_after_prediction_drops(self):
        tracker = OverlayHoldTracker(hold_seconds=1.0)
        first = tracker.update(
            [OverlayObservation((10, 10, 30, 30), "happy")], now=10.0
        )
        self.assertEqual(first[0].asset_label, "happy")

        held = tracker.update(
            [OverlayObservation((11, 10, 31, 30), None)], now=10.75
        )
        self.assertEqual(held[0].asset_label, "happy")
        self.assertEqual(held[0].box, (11, 10, 31, 30))
        self.assertEqual(tracker.update([], now=11.0)[0].asset_label, "happy")
        self.assertEqual(tracker.update([], now=11.01), [])

    def test_new_confident_label_switches_and_refreshes_hold(self):
        tracker = OverlayHoldTracker(hold_seconds=1.0)
        tracker.update([OverlayObservation((10, 10, 30, 30), "happy")], now=5.0)
        switched = tracker.update(
            [OverlayObservation((10, 10, 30, 30), "angry")], now=5.5
        )
        self.assertEqual(switched[0].asset_label, "angry")
        self.assertEqual(switched[0].expires_at, 6.5)


class CheckpointTests(unittest.TestCase):
    def test_explicit_checkpoint_must_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "best.pt"
            checkpoint.touch()
            self.assertEqual(
                resolve_checkpoint(checkpoint, "unused", "Classifier"), checkpoint.resolve()
            )


if __name__ == "__main__":
    unittest.main()
