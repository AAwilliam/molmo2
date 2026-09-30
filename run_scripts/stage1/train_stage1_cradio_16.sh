#!/usr/bin/env bash
set -e

# SAVE_FOLDER=${1:?用法: train_stage1_cradio.sh <checkpoint输出目录> [额外训练参数...]}
SAVE_FOLDER=./output/stage1_weight

source /usr/local/PPU_SDK/envsetup.sh
source /opt/molmo2_venv/bin/activate
cd /tmp/algorithm/molmo2
export PYTHONPATH="$PWD"
export MOLMO_DATA_DIR="$PWD"
export HF_HOME="$PWD/dataset/stage1/huggingface"
export PIXMO_IMAGE_DIR="$PWD/dataset/stage1/pixmo_images"
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

exec python -m torch.distributed.run --standalone --nproc-per-node=16 \
  launch_scripts/pretrain.py qwen3_4b_instruct \
  --model molmo2 --vision_backbone radio \
  --config configs/pretrain/cradio_v4_so400m_stage1.yaml \
  --wandb=null --save_folder="$SAVE_FOLDER" "$@"
