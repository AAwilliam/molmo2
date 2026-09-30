#!/usr/bin/env bash
set -e

cd /tmp/algorithm/molmo2
OUTPUT=pretrained_models/llm/qwen3-4b-instruct.pt
if [ -e "$OUTPUT" ]; then
  echo "已存在：$OUTPUT；不会覆盖。"
  exit 1
fi

source /usr/local/PPU_SDK/envsetup.sh
source /opt/molmo2_venv/bin/activate
export PYTHONPATH=$PWD
export MOLMO_DATA_DIR=$PWD
export HF_HOME=$PWD/dataset/stage1/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

python scripts/prepare_pretrained_model.py qwen3_4b_instruct \
  --data_dir "$PWD" --cache_dir "$HF_HOME" --unsharded

test -s "$OUTPUT"
