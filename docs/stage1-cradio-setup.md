# Stage 1 C-RADIO training setup

This repository contains Molmo2 code and configuration, not datasets or model weights.
Keep those assets outside Git and link them into the checkout as needed.

## Required local assets

- `pretrained_models/llm/qwen3-4b-instruct.pt`: converted Qwen3-4B-Instruct weights.
- `pretrained_models/llm/qwen3-4b-instruct-2507-modelscope/`: local tokenizer.
- `pretrained_models/vision/c-radio-v4-so400m-hf/`: complete C-RADIOv4-SO400M
  Hugging Face snapshot, including its custom Python modules and weights.
- `dataset/stage1/`: local data links for `DOCCI`, `TextCaps-FineVision`,
  `DCI`, `CoSyn-point`, `molmo2-tulu4-classified`, `pixmo_datasets`,
  `pixmo_images`, and `huggingface`.

On the current server, the dataset targets live under `/tmp/data/stage1` and
the links are under `/tmp/algorithm/molmo2/dataset/stage1`. Neither the data
nor the links are committed. Update the local paths when deploying elsewhere.

## Environment and launch

The current PPU/CUDA-compatible environment is `/opt/molmo2_venv`. The
`requirements-ppu.lock` file records Python package versions, but PPU vendor
packages must still be supplied by the container/SDK; it is not a complete
cross-platform `uv.lock`.

```bash
cd /tmp/algorithm/molmo2
source /usr/local/PPU_SDK/envsetup.sh
source /opt/molmo2_venv/bin/activate
export PYTHONPATH="$PWD"
export MOLMO_DATA_DIR="$PWD"
export HF_HOME="$PWD/dataset/stage1/huggingface"
export PIXMO_IMAGE_DIR="$PWD/dataset/stage1/pixmo_images"

bash run_scripts/stage1/train_stage1_cradio.sh
```

The single-device script and `configs/pretrain/cradio_v4_so400m_stage1.yaml`
contain the current Stage 1 configuration. Check the visible device count
before using the separate 16-process launcher.

## Data migration caveat

The saved Hugging Face `pixmo_datasets/{cap,count,points-counting,points-pointing}`
datasets can contain absolute image paths recorded when they were created.
Moving only the image files and setting `PIXMO_IMAGE_DIR` does **not** rewrite
those saved rows. After relocating images, rewrite and validate the saved
paths, including opening sampled images in the preprocessing pipeline, before
starting training. Merely calling each dataset's `get()` does not load the image.

As of 2026-09-30, the current server's saved PixMo rows still reference the
old `/tmp/data/pixmo_images` path. Its training launch fails on the first
batch until those rows are migrated; the images are available in
`/tmp/data/stage1/pixmo_images`.
