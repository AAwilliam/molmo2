#!/usr/bin/env python
"""Convert self-downloaded caption datasets (DOCCI / TextCaps-FineVision / DCI)
into trainable parquet under the project's dataset/stage1/ directory.

Output schema (mirrors CoSyn-point):
  id: string, image: struct{bytes: binary, path: string}, caption: string
  (DCI additionally keeps short_caption)

Usage:
  /opt/molmo2_venv/bin/python scripts/convert_caption_datasets.py --dataset all --dry-run
  /opt/molmo2_venv/bin/python scripts/convert_caption_datasets.py --dataset all
"""
import argparse
import glob
import io
import json
import os
import statistics
import time

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = "/tmp/algorithm/molmo2"
REPORT_DIR = "/tmp/algorithm/molmo2/dataset/stage1/_conversion/reports"
DOCCI_SRC = "/tmp/data/docci/docci_descriptions.jsonlines"
DOCCI_IMG = "/tmp/data/docci/images"
TEXTCAPS_DIR = "/tmp/data/textcaps_finevision/textcaps"
DCI_ANN = "/tmp/data/dci/densely_captioned_images/annotations"
DCI_IMG = "/tmp/data/dci/images"
TEXTCAPS_MIN_QUALITY = 3

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("caption", pa.string()),
])
DCI_SCHEMA = pa.schema(list(SCHEMA) + [("short_caption", pa.string())])


def valid_image_bytes(b):
    try:
        img = Image.open(io.BytesIO(b))
        w, h = img.size
        img.verify()
        return w > 0 and h > 0
    except Exception:
        return False


class ShardWriter:
    def __init__(self, out_dir, schema, shard_size=2000, prefix="train", overwrite=False):
        if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "*.parquet")):
            if not overwrite:
                raise RuntimeError(f"output already exists: {out_dir} (use --overwrite)")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.schema = schema
        self.shard_size = shard_size
        self.prefix = prefix
        self.rows = []
        self.tmp_files = []
        self.n = 0

    def add(self, row):
        self.rows.append(row)
        self.n += 1
        if len(self.rows) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        tmp = os.path.join(self.out_dir, f"tmp-{len(self.tmp_files):05d}.parquet")
        pq.write_table(pa.Table.from_pylist(self.rows, schema=self.schema), tmp)
        self.tmp_files.append(tmp)
        self.rows = []

    def finalize(self):
        self._flush()
        total = len(self.tmp_files)
        final = []
        for i, tmp in enumerate(self.tmp_files):
            fin = os.path.join(self.out_dir, f"{self.prefix}-{i:05d}-of-{total:05d}.parquet")
            os.replace(tmp, fin)
            final.append(fin)
        return final


class Stats:
    def __init__(self, name):
        self.name = name
        self.input = 0
        self.kept = 0
        self.dropped = {}
        self.word_lens = []
        self.samples = []  # (id, caption, image_bytes)

    def drop(self, reason):
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def keep(self, row):
        self.kept += 1
        self.word_lens.append(len(row["caption"].split()))
        self.input += 0  # noop
        if len(self.samples) < 30 or self.kept % 997 == 0:
            if len(self.samples) < 30:
                self.samples.append((row["id"], row["caption"], row["image"]["bytes"]))

    def report(self):
        wl = self.word_lens
        return {
            "dataset": self.name,
            "input_rows": self.input,
            "kept": self.kept,
            "dropped": dict(sorted(self.dropped.items())),
            "caption_words": {
                "avg": round(statistics.mean(wl), 1) if wl else 0,
                "p50": statistics.median(wl) if wl else 0,
                "max": max(wl) if wl else 0,
            },
        }


def save_samples(name, samples):
    thumb_dir = os.path.join(REPORT_DIR, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    lines = []
    for sid, caption, img_bytes in samples:
        safe_id = sid.replace("/", "_")
        thumb = os.path.join(thumb_dir, f"{name}_{safe_id}.jpg")
        try:
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            im.thumbnail((360, 360))
            im.save(thumb, "JPEG", quality=80)
        except Exception:
            thumb = None
        lines.append(json.dumps({"id": sid, "caption": caption[:500], "thumb": thumb},
                                ensure_ascii=False))
    with open(os.path.join(REPORT_DIR, f"samples_{name}.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def convert_docci(dry):
    name = "DOCCI"
    st = Stats(name)
    out_train = os.path.join(ROOT, "dataset", "stage1", "DOCCI", "data")
    out_eval = os.path.join(ROOT, "dataset", "eval", "DOCCI", "data")
    w_train = None if dry else ShardWriter(out_train, SCHEMA, prefix="train")
    w_eval = None if dry else ShardWriter(out_eval, SCHEMA, prefix="test")
    t0 = time.time()
    with open(DOCCI_SRC, encoding="utf-8") as f:
        for line in f:
            st.input += 1
            d = json.loads(line)
            caption = (d.get("description") or "").strip()
            if not caption:
                st.drop("empty_caption")
                continue
            img_path = os.path.join(DOCCI_IMG, d.get("image_file", ""))
            if not os.path.isfile(img_path):
                st.drop("missing_image")
                continue
            row = {
                "id": f"docci_{d['example_id']}",
                "image": {"bytes": b"" if dry else open(img_path, "rb").read(),
                          "path": d["image_file"]},
                "caption": caption,
            }
            if not dry and not valid_image_bytes(row["image"]["bytes"]):
                st.drop("invalid_image")
                continue
            if d.get("split") == "train":
                if w_train:
                    w_train.add(row)
                st.keep(row)
            else:
                if w_eval:
                    w_eval.add(row)
                st.drop("to_eval_split")
            if st.input % 2000 == 0:
                print(f"[{name}] {st.input} rows, {st.kept} kept, "
                      f"{time.time()-t0:.0f}s", flush=True)
    files = w_train.finalize() if w_train else []
    if w_eval:
        w_eval.finalize()
    if not dry:
        save_samples("docci", st.samples)
    return st, files


def convert_textcaps(dry):
    name = "TextCaps-FineVision"
    st = Stats(name)
    out_dir = os.path.join(ROOT, "dataset", "stage1", "TextCaps-FineVision", "data")
    writer = None if dry else ShardWriter(out_dir, SCHEMA)
    keys = ["formatting_min", "image_correspondence_min", "visual_dependency_min", "relevance_min"]
    t0 = time.time()
    seq = 0
    for f in sorted(glob.glob(os.path.join(TEXTCAPS_DIR, "*.parquet"))):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=256, columns=["images", "texts"] + keys):
            for row in batch.to_pylist():
                st.input += 1
                q = min(row[k] for k in keys)
                if q < TEXTCAPS_MIN_QUALITY:
                    st.drop("quality_below_threshold")
                    continue
                texts = row.get("texts") or []
                images = row.get("images") or []
                if not texts or not images:
                    st.drop("missing_field")
                    continue
                caption = (texts[0].get("assistant") or "").strip()
                img = images[0]
                if not caption or not img.get("bytes"):
                    st.drop("empty_caption_or_image")
                    continue
                seq += 1
                row_out = {
                    "id": f"textcaps_{seq:06d}",
                    "image": {"bytes": img["bytes"], "path": img.get("path") or ""},
                    "caption": caption,
                }
                if not dry and not valid_image_bytes(row_out["image"]["bytes"]):
                    st.drop("invalid_image")
                    continue
                if writer:
                    writer.add(row_out)
                st.keep(row_out)
            if st.input % 2000 == 0:
                print(f"[{name}] {st.input} rows, {st.kept} kept, "
                      f"{time.time()-t0:.0f}s", flush=True)
    files = writer.finalize() if writer else []
    if not dry:
        save_samples("textcaps", st.samples)
    return st, files


def convert_dci(dry):
    name = "DCI"
    st = Stats(name)
    out_dir = os.path.join(ROOT, "dataset", "stage1", "DCI", "data")
    writer = None if dry else ShardWriter(out_dir, DCI_SCHEMA)
    t0 = time.time()
    for p in sorted(glob.glob(os.path.join(DCI_ANN, "*-data.json"))):
        st.input += 1
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        img_name = d.get("image", "")
        caption = (d.get("extra_caption") or "").strip()
        short = (d.get("short_caption") or "").strip()
        img_path = os.path.join(DCI_IMG, img_name)
        if not caption:
            st.drop("empty_caption")
            continue
        if not os.path.isfile(img_path):
            st.drop("missing_image")
            continue
        row = {
            "id": f"dci_{os.path.splitext(img_name)[0]}",
            "image": {"bytes": b"" if dry else open(img_path, "rb").read(), "path": img_name},
            "caption": caption,
            "short_caption": short,
        }
        if not dry and not valid_image_bytes(row["image"]["bytes"]):
            st.drop("invalid_image")
            continue
        if writer:
            writer.add(row)
        st.keep(row)
        if st.input % 2000 == 0:
            print(f"[{name}] {st.input} rows, {st.kept} kept, {time.time()-t0:.0f}s", flush=True)
    files = writer.finalize() if writer else []
    if not dry:
        save_samples("dci", st.samples)
    return st, files


def verify_output(name, files, expect_rows):
    total = 0
    schemas_ok = True
    bad = 0
    for f in files:
        pf = pq.ParquetFile(f)
        total += pf.metadata.num_rows
        if f == files[0]:
            cols = [x.name for x in pf.schema_arrow]
            expect = ["id", "image", "caption"] + (["short_caption"] if name == "DCI" else [])
            schemas_ok = cols == expect
    # random decode check
    import random
    random.seed(0)
    checks = 0
    for f in random.sample(files, min(3, len(files))):
        pf = pq.ParquetFile(f)
        rows = pf.read_row_group(0).slice(0, 10).to_pylist()
        for r in rows:
            checks += 1
            if not valid_image_bytes(r["image"]["bytes"]) or not r["caption"].strip():
                bad += 1
    return {"files": len(files), "rows": total, "rows_match": total == expect_rows,
            "schema_ok": schemas_ok, "sample_checked": checks, "sample_bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["docci", "textcaps", "dci", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    runners = {"docci": convert_docci, "textcaps": convert_textcaps, "dci": convert_dci}
    names = list(runners) if args.dataset == "all" else [args.dataset]

    summary = []
    for n in names:
        print(f"===== converting {n} (dry_run={args.dry_run}) =====", flush=True)
        st, files = runners[n](args.dry_run)
        rep = st.report()
        if files:
            rep["verify"] = verify_output(n, files, st.kept)
        summary.append(rep)
        print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)

    tag = "dryrun" if args.dry_run else "full"
    out = os.path.join(REPORT_DIR, f"convert_report_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"report saved: {out}", flush=True)


if __name__ == "__main__":
    main()
