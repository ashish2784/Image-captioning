# Image Captioning with CNN + LSTM

This project trains an image captioning model on the [mini COCO 2014 dataset](https://www.kaggle.com/datasets/nagasai524/mini-coco2014-dataset-for-image-captioning). A ResNet-50 encoder extracts visual features, and an attention-enabled LSTM decoder generates captions one token at a time. After training, you can caption new images from the command line (batch or interactive) and optionally visualize the results.

## ✨ Features

- Automatic download of the Kaggle mini COCO 2014 dataset through `kagglehub`
- Flexible vocabulary construction with frequency thresholding and max-size limits
- Attention-based decoder with teacher forcing, scheduled sampling, and gradient clipping
- Checkpointing, metrics logging, and a JSON training summary for quick experiment review
- Command-line tools for single-shot inference and an interactive captioning REPL

## 🧱 Project Layout

```
image_captioning/
├── config.py                # Dataclasses for dataset/model/training configuration
├── data/                    # Dataset parsing, transforms, and DataLoader helpers
├── inference/               # Caption generation service for scripts and notebooks
├── models/                  # CNN encoder + attention LSTM decoder implementation
└── training/                # Training engine, pipeline orchestration, and utilities
scripts/
├── train.py                 # Training entry point (downloads data if needed)
├── infer.py                 # Single-image caption CLI with optional visualization
└── caption_interactive.py   # Interactive CLI for captioning multiple images
artifacts/
├── checkpoints/             # Saved model weights (one per epoch by default)
├── vocab.json               # Serialized vocabulary built during training
└── training_summary.json    # Loss/metric history for the latest run
```

## 🔧 Requirements & Setup

- Python 3.10+
- Kaggle API credentials configured for `kagglehub` (see the [authentication guide](https://github.com/Kaggle/kagglehub#authentication))
- A GPU (CUDA) or Apple Silicon (MPS) device is strongly recommended for training speed

Install the dependencies into your virtual environment:

```bash
pip install -r requirements.txt
```

If you plan to run on Apple Silicon, make sure your PyTorch wheels include MPS support. For CUDA, install a matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

## 🚂 Training

Kick off training (the script downloads the dataset into the workspace if `--dataset-root` is missing):

```bash
python scripts/train.py --epochs 1 --batch-size 64 --workspace artifacts
```

Commonly used flags:

- `--dataset-root`: reuse an existing dataset directory to skip re-downloads
- `--annotations`: provide a custom COCO captions JSON path
- `--unfreeze-encoder`: fine-tune the CNN encoder after the warm-up period
- `--no-pretrained`: skip downloading pretrained ResNet weights (offline environments)
- `--device`: set the compute device (`cuda`, `cuda:0`, `mps`, or `cpu`)

Each run stores checkpoints, `vocab.json`, and `training_summary.json` under the `--workspace` directory. The summary captures train/validation losses for quick inspection or plotting.

## 🖼️ Inference

Generate a caption for a single image with optional visualization:

```bash
python scripts/infer.py path/to/image.jpg --workspace artifacts --show
```

Key options:

- `--checkpoint`: choose a specific checkpoint file instead of the latest
- `--device`: override the default inference device
- `--show`: open a matplotlib window with the image and predicted caption

For an interactive captioning session that accepts multiple paths (or prompts you to paste one), use the REPL-style helper:

```bash
python scripts/caption_interactive.py --workspace artifacts --show
```

You can also pass an initial image path to caption it immediately before entering the loop.

## 🚀 Next Steps

- Experiment with beam search or nucleus sampling for richer caption diversity
- Add BLEU, METEOR, or CIDEr evaluation on a held-out test split
- Integrate TensorBoard or Weights & Biases for live metric dashboards
- Package the inference service behind a small REST API or Gradio demo
