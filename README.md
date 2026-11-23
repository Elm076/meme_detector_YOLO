# YOLO26 webcam meme App

This project trains the two models needed by a real-time webcam meme overlay:

1. **Face detector:** `yolo26n.pt` fine-tuned on WIDER FACE.
2. **Expression classifier:** `yolo26n-cls.pt` fine-tuned on labelled crops of
   your own expressions. Its class names are derived from the files in
   `assets/` (`bro`, `cooked`, `laughing`, `shocked`, `smiling`, and `sus`).

WIDER FACE contains face bounding boxes, not these expression categories. It
can make face detection much better than the generic COCO checkpoint, but it
cannot teach the model which meme to select. The expression dataset supplies
that missing supervision. In the live app, detect a face, crop it, classify the
crop, and draw the matching asset only when confidence remains high for several
frames.

This is expression classification, not identity recognition. WIDER FACE is
published under a non-commercial Creative Commons license; check the official
dataset terms before distributing a commercial application or trained weights.

## Run the live webcam overlay

After both models are trained, launch the live application with:

```powershell
python webcam_yolo.py --device 0
```

It automatically uses `runs/yolo26n_widerface/weights/best.pt` for face
detection and the newest `runs/yolo26n_expressions*/weights/best.pt` for
expression classification. Each prediction is matched to an image in `assets/`
by filename stem (`happy` uses `assets/happy.jpg`, for example). `neutral` has
no overlay. Press `Q` or Escape to close the window.

Pass checkpoints explicitly when needed:

```powershell
python webcam_yolo.py `
  --detector runs\yolo26n_widerface\weights\best.pt `
  --classifier runs\yolo26n_expressions_fixed\weights\best.pt `
  --device 0
```

If the wrong camera opens, add `--camera 1`. Detection and expression
thresholds can be adjusted with `--face-confidence` and
`--expression-confidence`. Expression overlays require 70% confidence by
default. Once an asset qualifies, it remains visible for one second after a
missed or low-confidence prediction to reduce flicker. Change that duration
with `--asset-hold-seconds`.

## Setup (Windows PowerShell)

Python 3.10-3.13 is recommended. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
```

`requirements-cuda.txt` selects the CUDA 13.0 PyTorch wheels required by the
RTX 50-series/Blackwell architecture. Use `requirements.txt` instead only for
a CPU installation. Installing the unqualified PyTorch wheel can produce a
`+cpu` build that sees no NVIDIA GPU even when `nvidia-smi` works.

For an NVIDIA GPU, verify that PyTorch sees CUDA before starting a long run:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For this machine the expected version suffix is `+cu130`, and the command must
print `True` followed by `NVIDIA GeForce RTX 5060 Ti`. The trainer selects GPU 0
automatically; `--device 0` remains available as an explicit override.

## Prepare WIDER FACE

The train and validation archives total roughly 3.4 GiB. Download and convert
them in one command:

```powershell
python trainer_webacam_yolo.py prepare-wider --download
```

Alternatively, manually put `WIDER_train.zip`, `WIDER_val.zip`, and
`wider_face_split.zip` in `data/wider_face/`, then omit `--download`. Invalid
and degenerate WIDER boxes are excluded by default.

## Train the face detector

The nano model is the best starting point for low-latency webcam inference:

```powershell
python trainer_webacam_yolo.py train-detector --device 0 --epochs 100
```

Use `--model yolo26s.pt` if the GPU has enough memory and accuracy matters more
than latency. `--batch -1` uses Ultralytics AutoBatch. Training output is stored
under `runs/yolo26n_widerface/`. Resume an interrupted run with:

```powershell
python trainer_webacam_yolo.py train-detector --resume runs\yolo26n_widerface\weights\last.pt
```

## Build and train the expression dataset

To build a balanced classification dataset from `data/fane_data`, sample the
same number of images from every label and create an 80/20 train/validation
split:

```powershell
python trainer_webcam_yolo.py prepare-fane
```

The default is 500 total images per label (400 train and 100 validation), saved
under `data/fane_balanced/`; the source images are never modified. Change the
amount, split, or random selection with `--images-per-label`, `--val-fraction`,
and `--seed`, for example:

```powershell
python trainer_webcam_yolo.py prepare-fane --images-per-label 750 --val-fraction 0.2
python trainer_webcam_yolo.py train-expression --data data\fane_balanced --device 0 --epochs 100
```

Expression training defaults to `--batch 128 --workers 0` on Windows. This
avoids page-file-backed DataLoader shared-memory failures that can occur when
AutoBatch selects an extremely large classification batch. Both values remain
configurable if you want to tune throughput.

Use `--overwrite` when intentionally replacing a previously generated balanced
dataset.

For a manually collected expression dataset, create folders based on the
current meme filenames:

```powershell
python trainer_webacam_yolo.py init-expression-data
```

Add tightly cropped human face photos to every generated class folder under
`data/expressions/train/` and `data/expressions/val/`. Do not train on the dog
memes themselves; those are output overlays. Aim for at least 200 varied images
per class, keep people separated across train and validation when possible, and
add a `neutral` meme asset/class if the application needs an explicit no-trigger
state. Then train:

```powershell
python trainer_webacam_yolo.py train-expression --device 0 --epochs 100
```

## Validate and export

```powershell
python trainer_webacam_yolo.py validate runs\yolo26n_widerface\weights\best.pt --data data\wider_face\wider_face.yaml --device 0
python trainer_webacam_yolo.py export runs\yolo26n_widerface\weights\best.pt --format onnx --device 0
```

The script also supports `engine` (TensorRT) and `openvino` exports. Run any
subcommand with `--help` to see all settings.
