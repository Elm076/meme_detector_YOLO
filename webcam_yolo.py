#!/usr/bin/env python3
"""Run live face detection, expression classification, and meme overlays."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR = PROJECT_ROOT / "runs" / "yolo26n_widerface" / "weights" / "best.pt"
DEFAULT_ASSETS = PROJECT_ROOT / "assets"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
NO_OVERLAY_LABELS = {"neutral"}

# Keep Ultralytics settings inside the project, matching the training script.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache"))


class LiveAppError(RuntimeError):
    """An actionable live-application error."""


@dataclass(frozen=True)
class OverlayObservation:
    """A detected face and the asset label eligible in the current frame."""

    box: tuple[int, int, int, int]
    asset_label: str | None


@dataclass
class HeldOverlay:
    """An asset kept visible briefly across unstable classifier results."""

    box: tuple[int, int, int, int]
    asset_label: str
    expires_at: float


def box_iou(first: Sequence[int], second: Sequence[int]) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class OverlayHoldTracker:
    """Match faces between frames and hold their latest qualifying asset."""

    def __init__(self, hold_seconds: float, match_iou: float = 0.1) -> None:
        self.hold_seconds = hold_seconds
        self.match_iou = match_iou
        self._held: list[HeldOverlay] = []

    def update(
        self, observations: Sequence[OverlayObservation], now: float
    ) -> list[HeldOverlay]:
        unmatched = set(range(len(self._held)))
        for observation in observations:
            best_index: int | None = None
            best_iou = self.match_iou
            for index in unmatched:
                overlap = box_iou(observation.box, self._held[index].box)
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = index

            if best_index is not None:
                unmatched.remove(best_index)
                held = self._held[best_index]
                held.box = observation.box
                if observation.asset_label is not None:
                    held.asset_label = observation.asset_label
                    held.expires_at = now + self.hold_seconds
            elif observation.asset_label is not None:
                self._held.append(
                    HeldOverlay(
                        box=observation.box,
                        asset_label=observation.asset_label,
                        expires_at=now + self.hold_seconds,
                    )
                )

        self._held = [held for held in self._held if held.expires_at >= now]
        return list(self._held)


def normalize_label(value: str) -> str:
    """Normalize model class and asset names to the same stable lookup key."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized.removesuffix("_dog")


def resolve_checkpoint(explicit: Path | None, pattern: str, description: str) -> Path:
    """Use an explicit checkpoint or the newest checkpoint matching a run pattern."""
    if explicit is not None:
        checkpoint = explicit.expanduser().resolve()
        if not checkpoint.is_file():
            raise LiveAppError(f"{description} checkpoint not found: {checkpoint}")
        return checkpoint

    candidates = [path for path in PROJECT_ROOT.glob(pattern) if path.is_file()]
    if not candidates:
        raise LiveAppError(
            f"No {description.lower()} checkpoint found under {PROJECT_ROOT / 'runs'}. "
            f"Pass --{description.lower().replace(' ', '-')} explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def class_name(names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
    """Read a class name from either Ultralytics names representation."""
    if isinstance(names, Mapping):
        if class_id in names:
            return str(names[class_id])
        return str(names[str(class_id)])  # type: ignore[index]
    return str(names[class_id])


def expand_box(
    box: Sequence[float], frame_width: int, frame_height: int, scale: float
) -> tuple[int, int, int, int]:
    """Scale a box around its center and clip it to the frame."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = max(1.0, (x2 - x1) * scale / 2.0)
    half_height = max(1.0, (y2 - y1) * scale / 2.0)
    return (
        max(0, int(round(center_x - half_width))),
        max(0, int(round(center_y - half_height))),
        min(frame_width, int(round(center_x + half_width))),
        min(frame_height, int(round(center_y + half_height))),
    )


def load_assets(asset_root: Path, cv2: Any) -> dict[str, Any]:
    """Load image assets, keyed by their normalized filename stems."""
    asset_root = asset_root.expanduser().resolve()
    if not asset_root.is_dir():
        raise LiveAppError(f"Asset directory not found: {asset_root}")

    assets: dict[str, Any] = {}
    for path in sorted(asset_root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = normalize_label(path.stem)
        if key in assets:
            raise LiveAppError(f"Multiple assets map to label {key!r} in {asset_root}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Warning: ignoring unreadable asset: {path}", file=sys.stderr)
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            print(f"Warning: ignoring unsupported asset: {path}", file=sys.stderr)
            continue
        assets[key] = image

    if not assets:
        raise LiveAppError(f"No readable image assets found in {asset_root}")
    return assets


def overlay_asset(frame: Any, asset: Any, box: Sequence[int], cv2: Any) -> None:
    """Cover a frame region with an aspect-preserving BGR/BGRA asset."""
    x1, y1, x2, y2 = (int(value) for value in box)
    target_width = x2 - x1
    target_height = y2 - y1
    if target_width <= 0 or target_height <= 0:
        return

    asset_height, asset_width = asset.shape[:2]
    resize_scale = max(target_width / asset_width, target_height / asset_height)
    resized_width = max(target_width, int(round(asset_width * resize_scale)))
    resized_height = max(target_height, int(round(asset_height * resize_scale)))
    resized = cv2.resize(asset, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    cropped = resized[top : top + target_height, left : left + target_width]

    if cropped.shape[2] == 4:
        alpha = cropped[:, :, 3:4].astype("float32") / 255.0
        foreground = cropped[:, :, :3].astype("float32")
        background = frame[y1:y2, x1:x2].astype("float32")
        frame[y1:y2, x1:x2] = (foreground * alpha + background * (1.0 - alpha)).astype(
            frame.dtype
        )
    else:
        frame[y1:y2, x1:x2] = cropped[:, :, :3]


def _runtime_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise LiveAppError(
            "Live dependencies are missing. Activate .venv and install "
            "requirements-cuda.txt (NVIDIA) or requirements.txt (CPU)."
        ) from exc
    return cv2, YOLO


def _open_camera(cv2: Any, camera_index: int, width: int, height: int) -> Any:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise LiveAppError(
            f"Cannot open camera {camera_index}. Close other camera apps or try --camera 1."
        )
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def _prediction_kwargs(device: str | None) -> dict[str, object]:
    return {"device": device} if device else {}


def run_live(args: argparse.Namespace) -> None:
    cv2, YOLO = _runtime_dependencies()
    detector_path = resolve_checkpoint(args.detector, "runs/yolo26n_widerface/weights/best.pt", "Detector")
    classifier_path = resolve_checkpoint(
        args.classifier, "runs/yolo26n_expressions_fixed/weights/best.pt", "Classifier"
    )
    assets = load_assets(args.assets, cv2)
    detector = YOLO(str(detector_path))
    classifier = YOLO(str(classifier_path))

    model_labels = {
        normalize_label(class_name(classifier.names, class_id))
        for class_id in range(len(classifier.names))
    }
    mapped = sorted(model_labels.intersection(assets))
    missing = sorted(model_labels.difference(assets).difference(NO_OVERLAY_LABELS))
    print(f"Face detector: {detector_path}")
    print(f"Expression classifier: {classifier_path}")
    print(f"Mapped assets: {', '.join(mapped) or 'none'}")
    if missing:
        print(f"Warning: no asset for labels: {', '.join(missing)}", file=sys.stderr)
    print("Press Q or Escape in the webcam window to quit.")

    capture = _open_camera(cv2, args.camera, args.width, args.height)
    overlay_tracker = OverlayHoldTracker(args.asset_hold_seconds)
    window_name = "YOLO26 Live Expressions"
    smoothed_fps: float | None = None
    last_frame_time = time.perf_counter()
    failed_reads = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= 10:
                    raise LiveAppError("Camera stopped returning frames")
                continue
            failed_reads = 0
            if args.mirror:
                frame = cv2.flip(frame, 1)

            detection = detector.predict(
                frame,
                conf=args.face_confidence,
                imgsz=args.detector_imgsz,
                max_det=args.max_faces,
                verbose=False,
                **_prediction_kwargs(args.device),
            )[0]

            face_boxes: list[tuple[int, int, int, int]] = []
            face_crops: list[Any] = []
            if detection.boxes is not None:
                frame_height, frame_width = frame.shape[:2]
                for raw_box in detection.boxes.xyxy.detach().cpu().tolist():
                    x1, y1, x2, y2 = expand_box(raw_box, frame_width, frame_height, 1.0)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    face_boxes.append((x1, y1, x2, y2))
                    face_crops.append(crop.copy())

            overlay_observations: list[OverlayObservation] = []
            label_draws: list[tuple[tuple[int, int, int, int], str, float]] = []
            if face_crops:
                classifications = classifier.predict(
                    face_crops,
                    imgsz=args.classifier_imgsz,
                    verbose=False,
                    **_prediction_kwargs(args.device),
                )
                frame_height, frame_width = frame.shape[:2]
                for box, classification in zip(face_boxes, classifications):
                    if classification.probs is None:
                        overlay_observations.append(OverlayObservation(box, None))
                        continue
                    class_id = int(classification.probs.top1)
                    confidence = float(classification.probs.top1conf)
                    label = normalize_label(class_name(classifier.names, class_id))
                    asset = assets.get(label)
                    qualifying_label = label if (
                        confidence >= args.expression_confidence
                        and label not in NO_OVERLAY_LABELS
                        and asset is not None
                    ) else None
                    overlay_observations.append(OverlayObservation(box, qualifying_label))
                    label_draws.append((box, label, confidence))

            frame_height, frame_width = frame.shape[:2]
            held_overlays = overlay_tracker.update(
                overlay_observations, time.perf_counter()
            )
            for held in held_overlays:
                asset = assets.get(held.asset_label)
                if asset is None:
                    continue
                display_box = expand_box(
                    held.box, frame_width, frame_height, args.overlay_scale
                )
                overlay_asset(frame, asset, display_box, cv2)

            if args.show_labels:
                for box, label, confidence in label_draws:
                    display_box = expand_box(
                        box, frame_width, frame_height, args.overlay_scale
                    )
                    x1, y1, x2, y2 = display_box
                    color = (
                        (80, 220, 80)
                        if confidence >= args.expression_confidence
                        else (0, 180, 255)
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.0%}",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now
            smoothed_fps = instant_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * instant_fps
            cv2.putText(
                frame,
                f"FPS {smoothed_fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live webcam face-expression meme overlay using two YOLO26 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=DEFAULT_DETECTOR,
        help="fine-tuned WIDER FACE detector checkpoint",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        help="expression checkpoint; defaults to the newest yolo26n_expressions* best.pt",
    )
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--device", help="Ultralytics device such as 0 or cpu")
    parser.add_argument("--face-confidence", type=float, default=0.4)
    parser.add_argument("--expression-confidence", type=float, default=0.70)
    parser.add_argument(
        "--asset-hold-seconds",
        type=float,
        default=1.0,
        help="keep the last qualifying asset visible after a missed prediction",
    )
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--classifier-imgsz", type=int, default=224)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--overlay-scale", type=float, default=1.15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--mirror", action=argparse.BooleanOptionalAction, default=True, help="mirror webcam image"
    )
    parser.add_argument(
        "--show-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw predicted expression and confidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 0.0 <= args.face_confidence <= 1.0:
            raise LiveAppError("face-confidence must be between 0 and 1")
        if not 0.0 <= args.expression_confidence <= 1.0:
            raise LiveAppError("expression-confidence must be between 0 and 1")
        if args.overlay_scale <= 0:
            raise LiveAppError("overlay-scale must be greater than 0")
        if args.asset_hold_seconds < 0:
            raise LiveAppError("asset-hold-seconds cannot be negative")
        run_live(args)
    except LiveAppError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
