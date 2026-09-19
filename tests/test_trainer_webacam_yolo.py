import tempfile
import unittest
from pathlib import Path

from PIL import Image

from trainer_webacam_yolo import (
    WiderBox,
    _to_yolo_box,
    build_parser,
    discover_meme_classes,
    iter_wider_annotations,
    parse_batch,
    prepare_wider,
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


class CommandLineTests(unittest.TestCase):
    def test_real_time_defaults_use_yolo26_nano(self):
        detector = build_parser().parse_args(["train-detector"])
        classifier = build_parser().parse_args(["train-expression"])
        self.assertEqual(detector.model, "yolo26n.pt")
        self.assertEqual(detector.name, "yolo26n_widerface")
        self.assertEqual(classifier.model, "yolo26n-cls.pt")
        self.assertEqual(classifier.imgsz, 224)

    def test_batch_parser_preserves_integer_and_fraction_forms(self):
        self.assertEqual(parse_batch("16"), 16)
        self.assertIsInstance(parse_batch("16"), int)
        self.assertEqual(parse_batch("-1"), -1)
        self.assertEqual(parse_batch("0.75"), 0.75)
        with self.assertRaises(Exception):
            parse_batch("1.5")


if __name__ == "__main__":
    unittest.main()
