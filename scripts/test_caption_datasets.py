"""Smoke test for the caption datasets (docci / textcaps / dci).

Checks: row counts, get() structure, image decoding, and end-to-end
formatting with the training DataFormatter settings.

Run from the repo root:
  PYTHONPATH=/tmp/algorithm/molmo2 /opt/molmo2_venv/bin/python \
      scripts/test_caption_datasets.py
"""
import io
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/algorithm/molmo2")

from olmo.data.get_dataset import get_dataset_by_name  # noqa: E402
from olmo.preprocessing.data_formatter import DataFormatter, GENERAL_PROMPTS_V1  # noqa: E402

EXPECT = {"docci": 9647, "textcaps": 21261, "dci": 7561}
OUT_DIR = "/tmp/algorithm/molmo2/dataset/stage1/_conversion/test_out"
os.makedirs(OUT_DIR, exist_ok=True)

formatter = DataFormatter(
    message_format="qwen3",
    system_prompt="style_and_length_v2",
    pointing_format="html-v1",
)

all_ok = True
for name in ["docci", "textcaps", "dci"]:
    print(f"===== {name} =====")
    ds = get_dataset_by_name(name, "train")
    n = len(ds)
    ok = n == EXPECT[name]
    all_ok &= ok
    print(f"rows: {n} (expect {EXPECT[name]}) {'OK' if ok else 'MISMATCH'}")

    rng = np.random.RandomState(0)
    for i in [0, n // 2, n - 1]:
        ex = ds.get(i, rng)
        assert set(ex.keys()) >= {"image", "message_list", "metadata"}, ex.keys()
        msg = ex["message_list"][0]
        image = ex["image"]
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, dict) and image.get("bytes") is not None:
            image = Image.open(io.BytesIO(image["bytes"]))
        elif isinstance(image, dict) and image.get("path") is not None:
            image = Image.open(image["path"])
        else:
            raise TypeError(f"Unsupported image value: {type(image)}")
        image.verify()
        if isinstance(ex["image"], str):
            im = Image.open(ex["image"]).convert("RGB")
        elif ex["image"].get("bytes") is not None:
            im = Image.open(io.BytesIO(ex["image"]["bytes"])).convert("RGB")
        else:
            im = Image.open(ex["image"]["path"]).convert("RGB")
        thumb = os.path.join(OUT_DIR, f"{name}_{i}.jpg")
        im.thumbnail((360, 360))
        im.save(thumb, "JPEG", quality=80)
        print(f"  [{i}] id={ex['metadata']['image_id']} style={msg['style']} "
              f"{im.size[0]}x{im.size[1]} words={len(msg['text'].split())}")
        print(f"      caption: {msg['text'][:110]}...")
        print(f"      thumb:   {thumb}")
        try:
            out1, out2 = formatter(ex, is_training=True, for_inference=False, rng=rng)
            print(f"      formatter keys: {list(out1.keys()) if isinstance(out1, dict) else type(out1)}")
            for k, v in (out1.items() if isinstance(out1, dict) else []):
                print(f"        {k}: {str(v)[:160]}")
        except Exception as e:
            print(f"      formatter call failed: {type(e).__name__}: {e}")
            print(f"      style pool sample: {GENERAL_PROMPTS_V1[msg['style']][0]}")

# mixture weight preview for the v2 config (sqrt(size) * group rate)
print("\n===== v2 mixture preview =====")
sizes = {"docci": EXPECT["docci"], "textcaps": EXPECT["textcaps"], "dci": EXPECT["dci"],
         "cosyn_point": 68051, "tulu4_max_2304": 977072}
groups = {"caption": (0.4, ["docci", "textcaps", "dci"]),
          "pointing": (0.4, ["cosyn_point"]),
          "nlp": (0.2, ["tulu4_max_2304"])}
total = 0.0
shares = {}
for g, (rate, members) in groups.items():
    ws = {m: math.sqrt(sizes[m]) for m in members}
    s = sum(ws.values())
    for m, w in ws.items():
        shares[m] = rate * w / s
        total += shares[m]
for m, v in sorted(shares.items(), key=lambda x: -x[1]):
    print(f"  {m:20s} {v*100:5.2f}%  (sum={total:.2f})")

print("\nRESULT:", "ALL_OK" if all_ok else "COUNT_MISMATCH")
