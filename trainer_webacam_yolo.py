#!/usr/bin/env python3
"""Prepare WIDER FACE and train YOLO26 models for a live meme overlay app.

The final application needs two models:

1. A detector fine-tuned on WIDER FACE to locate faces.
2. A classifier fine-tuned on labelled crops of the user's expressions.

WIDER FACE has face boxes, but it does not contain the six meme-expression
classes used by this project.  Keeping detection and expression classification
separate also makes live inference faster and makes it easy to add expressions.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WIDER_ROOT = PROJECT_ROOT / "data" / "wider_face"
DEFAULT_EXPRESSION_ROOT = PROJECT_ROOT / "data" / "expressions"
DEFAULT_ASSETS_ROOT = PROJECT_ROOT / "assets"

# Keep Ultralytics settings in this project. This avoids surprising writes to
# the user's profile and works in sandboxed or portable Windows environments.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache"))

# The CUHK-CSE Hugging Face repository mirrors the files linked by the
# official WIDER FACE project page.  The test split is intentionally omitted:
# its bounding-box ground truth is not public and it is not needed to train.
WIDER_ARCHIVES = {
    "WIDER_train.zip": (
        "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/"
        "data/WIDER_train.zip"
    ),
    "WIDER_val.zip": (
        "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/"
        "data/WIDER_val.zip"
    ),
    "wider_face_split.zip": (
        "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/"
        "data/wider_face_split.zip"
    ),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


class TrainerError(RuntimeError):
    """An actionable command-line error."""


@dataclass(frozen=True)
class WiderBox:
    x: float
    y: float
    width: float
    height: float
    invalid: bool


@dataclass
class ConversionStats:
    images: int = 0
    faces: int = 0
    invalid_faces: int = 0
    degenerate_faces: int = 0

    def __iadd__(self, other: "ConversionStats") -> "ConversionStats":
        self.images += other.images
        self.faces += other.faces
        self.invalid_faces += other.invalid_faces
        self.degenerate_faces += other.degenerate_faces
        return self


def _next_nonempty(lines: Sequence[str], index: int) -> tuple[str, int]:
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise TrainerError("Unexpected end of WIDER FACE annotation file")
    return lines[index].strip(), index + 1


def _looks_like_box(line: str) -> bool:
    parts = line.split()
    if len(parts) < 4:
        return False
    try:
        [float(value) for value in parts[:4]]
    except ValueError:
        return False
    return True


def iter_wider_annotations(annotation_file: Path) -> Iterator[tuple[Path, list[WiderBox]]]:
    """Yield image-relative paths and boxes from a WIDER FACE text file."""
    if not annotation_file.is_file():
        raise TrainerError(f"Annotation file not found: {annotation_file}")

    lines = annotation_file.read_text(encoding="utf-8-sig").splitlines()
    index = 0
    while index < len(lines):
        image_name, index = _next_nonempty(lines, index)
        count_text, index = _next_nonempty(lines, index)
        try:
            box_count = int(count_text)
        except ValueError as exc:
            raise TrainerError(
                f"Invalid face count {count_text!r} after {image_name!r} "
                f"in {annotation_file}"
            ) from exc

        boxes: list[WiderBox] = []
        for _ in range(box_count):
            box_line, index = _next_nonempty(lines, index)
            values = box_line.split()
            if len(values) < 4:
                raise TrainerError(f"Invalid box for {image_name!r}: {box_line!r}")
            try:
                x, y, width, height = (float(value) for value in values[:4])
                invalid = len(values) > 7 and int(float(values[7])) == 1
            except ValueError as exc:
                raise TrainerError(f"Invalid box for {image_name!r}: {box_line!r}") from exc
            boxes.append(WiderBox(x, y, width, height, invalid))

        # WIDER FACE annotation releases sometimes include one all-zero row
        # after an image whose declared face count is zero.
        if box_count == 0 and index < len(lines) and _looks_like_box(lines[index]):
            index += 1

        yield Path(image_name), boxes


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise TrainerError(
            "Pillow is required to prepare WIDER FACE. Install requirements first."
        ) from exc

    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError as exc:
        raise TrainerError(f"Cannot read image {image_path}: {exc}") from exc


def _to_yolo_box(box: WiderBox, image_width: int, image_height: int) -> str | None:
    x1 = min(max(box.x, 0.0), float(image_width))
    y1 = min(max(box.y, 0.0), float(image_height))
    x2 = min(max(box.x + box.width, 0.0), float(image_width))
    y2 = min(max(box.y + box.height, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None

    center_x = ((x1 + x2) / 2.0) / image_width
    center_y = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def convert_wider_split(
    annotation_file: Path,
    image_dir: Path,
    label_dir: Path,
    include_invalid: bool = False,
) -> ConversionStats:
    """Convert one official WIDER FACE split to Ultralytics YOLO labels."""
    if not image_dir.is_dir():
        raise TrainerError(f"Image directory not found: {image_dir}")

    stats = ConversionStats()
    missing_images: list[Path] = []
    for relative_image, boxes in iter_wider_annotations(annotation_file):
        image_path = image_dir / relative_image
        if not image_path.is_file():
            missing_images.append(image_path)
            continue

        image_width, image_height = _image_size(image_path)
        labels: list[str] = []
        for box in boxes:
            if box.invalid and not include_invalid:
                stats.invalid_faces += 1
                continue
            label = _to_yolo_box(box, image_width, image_height)
            if label is None:
                stats.degenerate_faces += 1
                continue
            labels.append(label)
            stats.faces += 1

        label_path = (label_dir / relative_image).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        stats.images += 1

    if missing_images:
        preview = "\n  ".join(str(path) for path in missing_images[:5])
        raise TrainerError(
            f"{len(missing_images)} annotated images are missing. First examples:\n  {preview}"
        )
    return stats


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_resolved != member_path and destination_resolved not in member_path.parents:
                raise TrainerError(f"Unsafe path in {archive.name}: {member.filename}")
        zip_file.extractall(destination)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name} ...")

    def report(blocks: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = min(blocks * block_size, total_size)
            percent = downloaded * 100.0 / total_size
            print(
                f"\r  {downloaded / 2**20:,.1f} / {total_size / 2**20:,.1f} MiB "
                f"({percent:5.1f}%)",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, partial, reporthook=report)
        print()
        partial.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        partial.unlink(missing_ok=True)
        raise TrainerError(f"Failed to download {url}: {exc}") from exc


def _dataset_layout_exists(dataset_root: Path) -> bool:
    required = (
        dataset_root / "WIDER_train" / "images",
        dataset_root / "WIDER_val" / "images",
        dataset_root / "wider_face_split" / "wider_face_train_bbx_gt.txt",
        dataset_root / "wider_face_split" / "wider_face_val_bbx_gt.txt",
    )
    return all(path.exists() for path in required)


def prepare_wider(dataset_root: Path, download: bool, include_invalid: bool) -> Path:
    dataset_root = dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    if not _dataset_layout_exists(dataset_root):
        for archive_name, url in WIDER_ARCHIVES.items():
            archive = dataset_root / archive_name
            if not archive.is_file():
                if not download:
                    raise TrainerError(
                        f"Missing {archive_name}. Re-run with --download, or place the three "
                        f"official WIDER FACE archives in {dataset_root}."
                    )
                _download(url, archive)
            print(f"Extracting {archive.name} ...")
            _safe_extract(archive, dataset_root)

    train_stats = convert_wider_split(
        dataset_root / "wider_face_split" / "wider_face_train_bbx_gt.txt",
        dataset_root / "WIDER_train" / "images",
        dataset_root / "WIDER_train" / "labels",
        include_invalid,
    )
    val_stats = convert_wider_split(
        dataset_root / "wider_face_split" / "wider_face_val_bbx_gt.txt",
        dataset_root / "WIDER_val" / "images",
        dataset_root / "WIDER_val" / "labels",
        include_invalid,
    )

    yaml_path = dataset_root / "wider_face.yaml"
    yaml_path.write_text(
        "\n".join(
            (
                f'path: "{dataset_root.as_posix()}"',
                "train: WIDER_train/images",
                "val: WIDER_val/images",
                "names:",
                "  0: face",
                "",
            )
        ),
        encoding="utf-8",
    )

    total = ConversionStats()
    total += train_stats
    total += val_stats
    print(
        f"Prepared {total.images:,} images and {total.faces:,} valid face boxes. "
        f"Skipped {total.invalid_faces:,} invalid and {total.degenerate_faces:,} "
        f"degenerate boxes.\nDataset YAML: {yaml_path}"
    )
    return yaml_path


def discover_meme_classes(assets_root: Path) -> list[str]:
    """Derive stable expression names from files such as shocked_dog.jpeg."""
    if not assets_root.is_dir():
        raise TrainerError(f"Meme asset directory not found: {assets_root}")
    classes = {
        path.stem.removesuffix("_dog").lower().replace(" ", "_")
        for path in assets_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not classes:
        raise TrainerError(f"No meme images found in {assets_root}")
    return sorted(classes)


def init_expression_dataset(dataset_root: Path, assets_root: Path) -> None:
    classes = discover_meme_classes(assets_root)
    for split in ("train", "val"):
        for class_name in classes:
            (dataset_root / split / class_name).mkdir(parents=True, exist_ok=True)
    instructions = dataset_root / "README.txt"
    instructions.write_text(
        "Labelled human face crops for YOLO26 expression classification.\n\n"
        "Put cropped webcam photos into the matching train/<expression> and\n"
        "val/<expression> folders. Meme images are overlays, not training data.\n"
        "Aim for at least 200 varied images per class, multiple people, lighting\n"
        "conditions, camera distances, and a neutral/no-trigger class if desired.\n\n"
        f"Classes derived from assets: {', '.join(classes)}\n",
        encoding="utf-8",
    )
    print(f"Created expression dataset folders at {dataset_root.resolve()}")
    print(f"Classes: {', '.join(classes)}")


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainerError(
            "Ultralytics is not installed. Activate .venv and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return YOLO


def resolve_device(device: str | None) -> str:
    """Select GPU 0 automatically and diagnose CPU-only PyTorch installations."""
    try:
        import torch
    except ImportError as exc:
        raise TrainerError("PyTorch is not installed in the active environment.") from exc

    requested = (device or "").strip().lower().replace("cuda:", "")
    if requested == "cuda":
        requested = "0"

    cuda_requested = bool(requested) and requested not in {"cpu", "mps"}
    if cuda_requested and not torch.cuda.is_available():
        raise TrainerError(
            f"CUDA device {device!r} was requested, but PyTorch {torch.__version__} "
            f"was built with CUDA {torch.version.cuda!r} and reports no CUDA devices. "
            "For this RTX 5060 Ti environment, install the CUDA build with "
            "`python -m pip install -r requirements-cuda.txt`."
        )

    selected = requested or ("0" if torch.cuda.is_available() else "cpu")
    if selected == "cpu":
        print("Using CPU because CUDA is not available to PyTorch.")
    elif selected == "mps":
        print("Using Apple MPS.")
    else:
        index_text = selected.split(",", maxsplit=1)[0]
        try:
            index = int(index_text)
            gpu_name = torch.cuda.get_device_name(index)
        except (ValueError, RuntimeError, AssertionError):
            gpu_name = "requested CUDA device"
        print(f"Using CUDA device {selected}: {gpu_name}")
    return selected


def _device_kwargs(device: str | None) -> dict[str, str]:
    return {"device": resolve_device(device)}


def train_detector(args: argparse.Namespace) -> None:
    data = args.data.resolve()
    if not data.is_file():
        raise TrainerError(
            f"Dataset YAML not found: {data}. Run the prepare-wider command first."
        )
    print(
        "Training the face detector. WIDER FACE improves face localization; "
        "it does not provide meme-expression labels."
    )
    YOLO = _load_yolo()
    model = YOLO(args.model)
    result = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        amp=not args.no_amp,
        plots=True,
        close_mosaic=10,
        resume=args.resume,
        **_device_kwargs(args.device),
    )
    print(f"Detector training complete: {result.save_dir}")


def _check_expression_dataset(dataset_root: Path, expected_classes: Sequence[str]) -> None:
    problems: list[str] = []
    for split in ("train", "val"):
        split_root = dataset_root / split
        for class_name in expected_classes:
            class_root = split_root / class_name
            count = (
                sum(
                    1
                    for path in class_root.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
                if class_root.is_dir()
                else 0
            )
            if count == 0:
                problems.append(f"{split}/{class_name} has no images")
    if problems:
        raise TrainerError(
            "Expression dataset is incomplete:\n  "
            + "\n  ".join(problems)
            + "\nRun init-expression-data and add labelled human face crops."
        )


def train_expression_classifier(args: argparse.Namespace) -> None:
    classes = discover_meme_classes(args.assets)
    _check_expression_dataset(args.data, classes)
    YOLO = _load_yolo()
    model = YOLO(args.model)
    result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        amp=not args.no_amp,
        plots=True,
        **_device_kwargs(args.device),
    )
    print(f"Expression-classifier training complete: {result.save_dir}")


def validate_model(args: argparse.Namespace) -> None:
    YOLO = _load_yolo()
    kwargs: dict[str, object] = {"imgsz": args.imgsz, "plots": True}
    if args.data:
        kwargs["data"] = str(args.data.resolve())
    kwargs.update(_device_kwargs(args.device))
    metrics = YOLO(str(args.weights.resolve())).val(**kwargs)
    print(f"Validation complete. Results saved to {metrics.save_dir}")


def export_model(args: argparse.Namespace) -> None:
    YOLO = _load_yolo()
    kwargs: dict[str, object] = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
    }
    kwargs.update(_device_kwargs(args.device))
    output = YOLO(str(args.weights.resolve())).export(**kwargs)
    print(f"Export complete: {output}")


def parse_batch(value: str) -> int | float:
    """Parse Ultralytics batch syntax without turning integer sizes into floats."""
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch must be -1, a positive integer, or 0..1") from exc
    if number == -1:
        return -1
    if number.is_integer() and number >= 1:
        return int(number)
    if 0 < number < 1:
        return number
    raise argparse.ArgumentTypeError("batch must be -1, a positive integer, or a fraction in 0..1")


def _add_shared_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=parse_batch,
        default=-1,
        help="integer batch size, -1 for AutoBatch, or a 0..1 GPU-memory fraction",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--device", help="Ultralytics device, e.g. 0, 0,1, cpu, or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--name", required=False)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare WIDER FACE and fine-tune YOLO26 for live meme expressions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-wider", help="Download/extract WIDER FACE and convert boxes to YOLO format"
    )
    prepare.add_argument("--dataset-root", type=Path, default=DEFAULT_WIDER_ROOT)
    prepare.add_argument(
        "--download",
        action="store_true",
        help="Download the roughly 3.4 GiB train/validation dataset when missing",
    )
    prepare.add_argument(
        "--include-invalid",
        action="store_true",
        help="Keep boxes marked invalid by WIDER FACE (normally undesirable)",
    )

    init_expressions = subparsers.add_parser(
        "init-expression-data", help="Create labelled face-crop folders from meme asset names"
    )
    init_expressions.add_argument("--dataset-root", type=Path, default=DEFAULT_EXPRESSION_ROOT)
    init_expressions.add_argument("--assets", type=Path, default=DEFAULT_ASSETS_ROOT)

    detector = subparsers.add_parser(
        "train-detector", help="Fine-tune a pretrained YOLO26 detector on WIDER FACE"
    )
    detector.add_argument(
        "--data", type=Path, default=DEFAULT_WIDER_ROOT / "wider_face.yaml"
    )
    detector.add_argument("--model", default="yolo26n.pt")
    detector.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume the last run, or pass a checkpoint path",
    )
    _add_shared_training_args(detector)
    detector.set_defaults(name="yolo26n_widerface")

    expression = subparsers.add_parser(
        "train-expression", help="Fine-tune YOLO26-cls on labelled face crops"
    )
    expression.add_argument("--data", type=Path, default=DEFAULT_EXPRESSION_ROOT)
    expression.add_argument("--assets", type=Path, default=DEFAULT_ASSETS_ROOT)
    expression.add_argument("--model", default="yolo26n-cls.pt")
    _add_shared_training_args(expression)
    expression.set_defaults(name="yolo26n_expressions", imgsz=224)

    validate = subparsers.add_parser("validate", help="Validate a trained checkpoint")
    validate.add_argument("weights", type=Path)
    validate.add_argument("--data", type=Path)
    validate.add_argument("--imgsz", type=int, default=640)
    validate.add_argument("--device")

    export = subparsers.add_parser("export", help="Export a checkpoint for live inference")
    export.add_argument("weights", type=Path)
    export.add_argument("--format", default="onnx", choices=("onnx", "engine", "openvino"))
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--device")
    export.add_argument("--half", action="store_true")
    export.add_argument("--dynamic", action="store_true")
    export.add_argument("--no-simplify", dest="simplify", action="store_false")
    export.set_defaults(simplify=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-wider":
            prepare_wider(args.dataset_root, args.download, args.include_invalid)
        elif args.command == "init-expression-data":
            init_expression_dataset(args.dataset_root, args.assets)
        elif args.command == "train-detector":
            train_detector(args)
        elif args.command == "train-expression":
            train_expression_classifier(args)
        elif args.command == "validate":
            validate_model(args)
        elif args.command == "export":
            export_model(args)
        else:  # pragma: no cover - argparse guarantees a known command
            raise TrainerError(f"Unknown command: {args.command}")
    except TrainerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
