import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from trainer_webcam_yolo import (
    WiderBox,
    _to_yolo_box,
    build_parser,
    discover_meme_classes,
    iter_wider_annotations,
    parse_batch,
    prepare_fane_dataset,
    prepare_wider,
    resolve_device,
)


class WiderAnnotationTests(unittest.TestCase):
    def test_parses_boxes_and_zero_count_placeholder(self):
        content = """0--Parade/a.jpg
2
10 20 30 40 0 0 0 0 0 0
1 2 3 4 0 0 0 1 0 0
1--Handshaking/b.jpg
0
0 0 0 0 0 0 0 0 0 0
2--Demonstration/c.jpg
1
5 6 7 8 0 0 0 0 0 0
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            annotation = Path(temp_dir) / "annotations.txt"
            annotation.write_text(content, encoding="utf-8")
            records = list(iter_wider_annotations(annotation))

        self.assertEqual([record[0].as_posix() for record in records], [
            "0--Parade/a.jpg",
            "1--Handshaking/b.jpg",
            "2--Demonstration/c.jpg",
        ])
        self.assertEqual(len(records[0][1]), 2)
        self.assertTrue(records[0][1][1].invalid)
        self.assertEqual(records[1][1], [])
        self.assertEqual(records[2][1][0].width, 7)

    def test_box_is_clipped_and_normalized(self):
        label = _to_yolo_box(WiderBox(-10, 10, 60, 100, False), 100, 100)
        self.assertEqual(label, "0 0.250000 0.550000 0.500000 0.900000")
        self.assertIsNone(_to_yolo_box(WiderBox(20, 20, 0, 4, False), 100, 100))

    def test_prepare_converts_both_splits_and_writes_yaml(self):
        annotation = "event/example.jpg\n2\n10 20 30 40 0 0 0 0 0 0\n1 2 3 4 0 0 0 1 0 0\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_root = root / "wider_face_split"
            split_root.mkdir()
            for split in ("train", "val"):
                image_dir = root / f"WIDER_{split}" / "images" / "event"
                image_dir.mkdir(parents=True)
                Image.new("RGB", (100, 100)).save(image_dir / "example.jpg")
                (split_root / f"wider_face_{split}_bbx_gt.txt").write_text(
                    annotation, encoding="utf-8"
                )

            yaml_path = prepare_wider(root, download=False, include_invalid=False)
            label = root / "WIDER_train" / "labels" / "event" / "example.txt"
            self.assertTrue(yaml_path.is_file())
            self.assertEqual(
                label.read_text(encoding="utf-8"),
                "0 0.250000 0.400000 0.300000 0.400000\n",
            )
            self.assertIn("0: face", yaml_path.read_text(encoding="utf-8"))


class MemeClassTests(unittest.TestCase):
    def test_discovers_class_names_from_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            assets = Path(temp_dir)
            (assets / "shocked_dog.jpeg").touch()
            (assets / "smiling_dog.jfif").touch()
            (assets / "notes.txt").touch()
            self.assertEqual(discover_meme_classes(assets), ["shocked", "smiling"])


class FaneDatasetTests(unittest.TestCase):
    def test_balances_labels_and_builds_train_val_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fane"
            output = root / "balanced"
            for label in ("angry", "happy"):
                label_dir = source / label
                label_dir.mkdir(parents=True)
                for index in range(8):
                    Image.new("RGB", (8, 8), color=(index, 0, 0)).save(
                        label_dir / f"{label}{index}.jpg"
                    )

            prepare_fane_dataset(source, output, images_per_label=5, val_fraction=0.4, seed=7)

            for label in ("angry", "happy"):
                self.assertEqual(len(list((output / "train" / label).glob("*.jpg"))), 3)
                self.assertEqual(len(list((output / "val" / label).glob("*.jpg"))), 2)
                self.assertEqual(len(list((source / label).glob("*.jpg"))), 8)
            self.assertTrue((output / "dataset.json").is_file())

    def test_skips_files_that_have_an_image_extension_but_are_not_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_dir = root / "fane" / "happy"
            label_dir.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (8, 8)).save(label_dir / f"valid{index}.jpg")
            (label_dir / "not_really_an_image.jpg").write_text(
                '{"cells": []}', encoding="utf-8"
            )

            output = prepare_fane_dataset(
                root / "fane", root / "balanced", images_per_label=2
            )

            selected = list((output / "train" / "happy").glob("*.jpg"))
            selected += list((output / "val" / "happy").glob("*.jpg"))
            self.assertEqual(len(selected), 2)
            self.assertNotIn("not_really_an_image.jpg", {path.name for path in selected})

    def test_rejects_labels_with_too_few_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_dir = root / "fane" / "angry"
            label_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(label_dir / "one.jpg")
            with self.assertRaisesRegex(Exception, "Not enough images"):
                prepare_fane_dataset(root / "fane", root / "balanced", images_per_label=2)


class CommandLineTests(unittest.TestCase):
    def test_real_time_defaults_use_yolo26_nano(self):
        detector = build_parser().parse_args(["train-detector"])
        classifier = build_parser().parse_args(["train-expression"])
        self.assertEqual(detector.model, "yolo26n.pt")
        self.assertEqual(detector.name, "yolo26n_widerface")
        self.assertEqual(classifier.model, "yolo26n-cls.pt")
        self.assertEqual(classifier.imgsz, 224)
        self.assertEqual(classifier.batch, 128)
        self.assertEqual(classifier.workers, 0)

    def test_fane_defaults_to_500_images_per_label(self):
        fane = build_parser().parse_args(["prepare-fane"])
        self.assertEqual(fane.images_per_label, 500)
        self.assertEqual(fane.val_fraction, 0.2)

    def test_batch_parser_preserves_integer_and_fraction_forms(self):
        self.assertEqual(parse_batch("16"), 16)
        self.assertIsInstance(parse_batch("16"), int)
        self.assertEqual(parse_batch("-1"), -1)
        self.assertEqual(parse_batch("0.75"), 0.75)
        with self.assertRaises(Exception):
            parse_batch("1.5")

    @patch("torch.cuda.get_device_name", return_value="Test GPU")
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_defaults_to_first_cuda_gpu(self, _available, _name):
        self.assertEqual(resolve_device(None), "0")
        self.assertEqual(resolve_device("cuda:0"), "0")

    @patch("torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_has_actionable_error_on_cpu_build(self, _available):
        with self.assertRaisesRegex(Exception, "requirements-cuda.txt"):
            resolve_device("0")

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_falls_back_to_cpu_when_not_requested(self, _available):
        self.assertEqual(resolve_device(None), "cpu")


if __name__ == "__main__":
    unittest.main()
