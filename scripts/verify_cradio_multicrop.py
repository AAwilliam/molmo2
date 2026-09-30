"""Check Molmo2-style C-RADIO crops and render CoSyn grounding examples.

Run from the repository root after sourcing the PPU SDK environment.
"""

import argparse
import dataclasses
import io
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from olmo.data.video_loader import VideoFrames
from olmo.model_configs import LLMS, RADIO_VISION_BACKBONE
from olmo.nn.image_vit import RadioVisionTransformer
from olmo.nn.vision_backbone import MolmoVisionBackbone, MolmoVisionBackboneConfig
from olmo.preprocessing.image_preprocessor import ImagePreprocessor, select_tiling
from olmo.preprocessing.multicrop_preprocessor import MultiCropImagePreprocessor
from olmo.preprocessing.preprocessor_utils import batch_pixels_to_patches
from olmo.preprocessing.video_preprocessor import TokenIndexingVideoPreprocessor


SIZE = 512
PATCH = 16
MARGIN = 4
STRIDE = (SIZE // PATCH - 2 * MARGIN) * PATCH
COLORS = ["#ff304f", "#00b5d8", "#ffbc16", "#a258ff", "#35c273"]


def make_preprocessor():
    image = ImagePreprocessor(
        normalize="none", resize="siglip", image_patch_size=PATCH,
        base_image_input_size=(SIZE, SIZE), use_image_mask=False,
    )
    tokens = SimpleNamespace(
        image_patch_token_id=1, image_col_token_id=2,
        image_start_token_id=3, image_end_token_id=4,
        image_low_res_token_id=5, low_res_image_start_token_id=6,
        frame_start_token_id=7, frame_end_token_id=8,
    )
    multicrop = MultiCropImagePreprocessor(
        tokenizer=tokens, image_preprocessor=image,
        crop_mode="overlap-and-resize-c2", max_crops=8,
        overlap_margins=(MARGIN, MARGIN),
        single_crop_for_small_images=True,
        use_single_crop_col_tokens=False,
        use_single_crop_start_token=True,
    )
    return image, multicrop


def assert_point_maps_once(data, x_norm, y_norm):
    mapping = data.token_mapping
    px = min(max(math.floor(x_norm * mapping.shape[1]), 0), mapping.shape[1] - 1)
    py = min(max(math.floor(y_norm * mapping.shape[0]), 0), mapping.shape[0] - 1)
    patch_id = mapping[py, px]
    locations = np.argwhere(data.token_pooling == patch_id)
    assert len(locations) == 1, (x_norm, y_norm, patch_id, locations)


def synthetic_checks(image_pre, multicrop):
    config = OmegaConf.load("configs/pretrain/cradio_v4_so400m_stage1.yaml")
    assert config.model.vision_backbone.vit.resize_mode == "siglip"
    assert config.model.vision_backbone.vit.normalize == "none"
    assert config.model.mm_preprocessor.image.crop_mode == "overlap-and-resize-c2"
    assert config.model.mm_preprocessor.image.max_crops == 8
    assert list(config.model.mm_preprocessor.image.overlap_margins) == [4, 4]
    assert config.model.mm_preprocessor.image.single_crop_for_small_images is True
    # Both common image dtypes must reach RADIO on the same [0, 1] scale.
    for dtype in (np.uint8, np.float32):
        source = np.full((32, 48, 3), 255 if dtype == np.uint8 else 1.0, dtype=dtype)
        normalized, mask = image_pre.resize_image(source, (SIZE, SIZE), False, np.random)
        assert np.allclose(normalized, 1.0), (dtype, normalized.min(), normalized.max())
        assert mask.all()
    for h, w, n_crops, mapping_shape, n_tokens in [
        (20, 20, 1, (32, 32), 256),
        (256, 384, 1, (32, 32), 256),
        (512, 512, 1, (32, 32), 256),
        (513, 513, 5, (56, 56), 1040),
        (896, 896, 5, (56, 56), 1040),
        (640, 1280, None, None, None),
    ]:
        arr = np.full((h, w, 3), 255, dtype=np.uint8)
        data = multicrop(arr)
        assert data.images.shape[1:] == (1024, 768), data.images.shape
        assert data.images.dtype == np.float32, data.images.dtype
        assert np.allclose(data.images, 1.0), "Unexpected padding or pixel normalization"
        if n_crops is not None:
            assert data.images.shape[0] == n_crops, data.images.shape
            assert data.token_mapping.shape == mapping_shape, data.token_mapping.shape
            assert data.token_pooling.shape == (n_tokens, 4), data.token_pooling.shape
        valid = data.token_pooling[data.token_pooling >= 0]
        assert len(valid) > 0 and valid.max() < data.images.shape[0] * 1024
        for x, y in [(0, 0), (0.5, 0.5), (0.999, 0.999), (0.43, 0.57)]:
            assert_point_maps_once(data, x, y)
        # Exercise the existing C-RADIO pixel-layout adapter without loading its weights.
        import torch
        dummy = SimpleNamespace(config=SimpleNamespace(
            image_patch_size=PATCH, image_default_input_size=(SIZE, SIZE)))
        recovered = RadioVisionTransformer._unpatchify(dummy, torch.from_numpy(data.images))
        assert recovered.shape == (data.images.shape[0], 3, SIZE, SIZE)
        assert torch.allclose(recovered, torch.ones_like(recovered))
        print(f"PASS image {w}x{h}: crops={data.images.shape[0]}, pooled={data.token_pooling.shape[0]}")

    original_mode = dataclasses.replace(multicrop, single_crop_for_small_images=False)
    original_data = original_mode(np.full((512, 512, 3), 255, dtype=np.uint8))
    assert original_data.images.shape[0] == 2
    assert original_data.token_pooling.shape == (512, 4)
    print("PASS default Molmo2 mode: one local crop plus one global image")

    frames = np.full((2, 360, 640, 3), 255, dtype=np.uint8)
    video_crops, video_masks, _ = image_pre.build_single_crop(frames, False, np.random)
    assert video_crops.shape == (2, SIZE, SIZE, 3)
    assert video_masks is None
    assert np.allclose(video_crops, 1.0)
    assert batch_pixels_to_patches(video_crops, PATCH).shape == (2, 1024, 768)
    print("PASS video preprocessing: two frames, direct 512x512 resize, no padding")

    video = TokenIndexingVideoPreprocessor(
        tokenizer=multicrop.tokenizer, image_preprocessor=image_pre,
        video_loader=None, pooling=3, use_frame_special_tokens=True,
    )
    video_data = video.video_to_patches_and_tokens(
        VideoFrames(frames, np.array([0.0, 1.0]), target_fps=1.0),
        frame_prefixes=[[], []],
    )
    assert video_data.images.shape == (2, 1024, 768)
    assert video_data.token_pooling.shape == (242, 9)
    assert video_data.token_mapping.shape == (2, 32, 32)
    print("PASS video tokenization: 121 pooled visual positions per frame")

    class MockRadio(__import__("torch").nn.Module):
        def forward(self, images):
            assert images.shape == (2, 3, SIZE, SIZE)
            return SimpleNamespace(features=images.new_zeros((2, 1024, 1152)))

    import torch
    mock_tower = object.__new__(RadioVisionTransformer)
    torch.nn.Module.__init__(mock_tower)
    mock_tower.config = SimpleNamespace(
        image_patch_size=PATCH, image_default_input_size=(SIZE, SIZE), image_emb_dim=1152)
    mock_tower.radio = MockRadio()
    result = mock_tower(torch.from_numpy(video_data.images))[0]
    assert result.shape == (2, 1024, 1152)
    print("PASS C-RADIO adapter: unpatchify and spatial-output shape contract")


def decode_image(value):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError(f"Unsupported image payload: {type(value)}")


def first_points(row):
    names = row.get("names") or []
    answer_points = row.get("answer_points") or []
    for name, points in zip(names, answer_points):
        if isinstance(points, dict):
            xs, ys = points.get("x", []), points.get("y", [])
            xy = [(float(x) / 100.0, float(y) / 100.0) for x, y in zip(xs, ys)]
            xy = [(x, y) for x, y in xy if 0 <= x <= 1 and 0 <= y <= 1]
            if xy:
                return str(name), xy[:5]
    return None, []


def mark(draw, x, y, color, label):
    radius = 9
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="white", width=5)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)
    draw.text((x+radius+3, y-radius-3), label, fill=color, stroke_width=2, stroke_fill="white")


def render_sample(image, points, label, index, image_pre, multicrop, output_dir):
    source = np.asarray(image).copy()
    data = multicrop(source)
    h, w = source.shape[:2]
    margin_px = 2 * MARGIN * PATCH
    rows, cols = select_tiling(max(h-margin_px, 1), max(w-margin_px, 1), STRIDE, 8)
    target_h, target_w = int(rows * STRIDE + margin_px), int(cols * STRIDE + margin_px)
    resized, mask = image_pre.resize_image(source, (target_h, target_w), False, np.random)
    assert mask.all(), "Original Molmo2 resize should not introduce padding"
    expected_mapping = (target_h // PATCH, target_w // PATCH)
    assert data.token_mapping.shape == expected_mapping

    original = image.copy()
    grid = Image.fromarray(np.clip(resized * 255, 0, 255).astype(np.uint8))
    thumb = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    od, gd, td = ImageDraw.Draw(original), ImageDraw.Draw(grid), ImageDraw.Draw(thumb)
    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * STRIDE, row * STRIDE
            gd.rectangle((x0, y0, x0+SIZE-1, y0+SIZE-1), outline="#00ef7a", width=2)
            gd.text((x0+8, y0+8), f"crop {row},{col}", fill="black", stroke_width=2, stroke_fill="white")
    for point_ix, (x, y) in enumerate(points):
        assert_point_maps_once(data, x, y)
        color = COLORS[point_ix % len(COLORS)]
        tag = f"{point_ix+1}:({x:.2f},{y:.2f})"
        mark(od, x*w, y*h, color, tag)
        mark(gd, x*target_w, y*target_h, color, tag)
        mark(td, x*SIZE, y*SIZE, color, str(point_ix+1))

    max_panel = 850
    panels = []
    for panel in (original, grid, thumb):
        scale = min(1, max_panel / max(panel.size))
        panels.append(panel.resize((max(1, round(panel.width*scale)), max(1, round(panel.height*scale)))))
    gutter, header = 12, 60
    out_w = sum(panel.width for panel in panels) + 4*gutter
    out_h = max(panel.height for panel in panels) + header + 2*gutter
    canvas = Image.new("RGB", (out_w, out_h), "#f2f2f2")
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 8), f"CoSyn grounding #{index:02d}  {label[:80]}", fill="black")
    draw.text((gutter, 30), f"original {w}x{h} | grid {cols}x{rows}, {target_w}x{target_h} | crops {data.images.shape[0]} | points {len(points)}", fill="black")
    x_cursor = gutter
    for panel in panels:
        canvas.paste(panel, (x_cursor, header))
        x_cursor += panel.width + gutter
    path = output_dir / f"grounding_{index:02d}.png"
    canvas.save(path)
    print(f"VISUAL {path} original={w}x{h} grid={cols}x{rows} points={len(points)}")


def visualize_real_data(image_pre, multicrop, output_dir, count):
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("dataset/stage1/CoSyn-point/data")
    index = 0
    for parquet_path in sorted(data_dir.glob("train-*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=16, columns=["image", "answer_points", "names"]):
            for row in batch.to_pylist():
                label, points = first_points(row)
                if not points:
                    continue
                image = decode_image(row["image"])
                index += 1
                render_sample(image, points, label, index, image_pre, multicrop, output_dir)
                if index >= count:
                    return
    raise RuntimeError(f"Found only {index}/{count} CoSyn rows with valid points")


def real_cpu_forward():
    """Exercise the actual pretrained C-RADIO weights without touching the busy GPU."""
    import dataclasses
    import torch

    torch.set_num_threads(2)
    snapshot = Path("pretrained_models/vision/c-radio-v4-so400m-hf").resolve()
    config = dataclasses.replace(RADIO_VISION_BACKBONE, init_path=str(snapshot))
    tower = RadioVisionTransformer(config, device="cpu")
    tower.reset_with_pretrained_weights()
    tower.eval()
    image = np.full((1, SIZE, SIZE, 3), 128.0 / 255.0, dtype=np.float32)
    patches = torch.from_numpy(batch_pixels_to_patches(image, PATCH))
    with torch.inference_mode():
        features = tower(patches)[0]
    assert features.shape == (1, 1024, 1152), features.shape
    assert torch.isfinite(features).all()
    print(f"PASS actual C-RADIO CPU forward: {tuple(features.shape)}")


def real_connector_cpu_forward(multicrop):
    """Check real RADIO weights through Molmo2's pooling and projector."""
    import dataclasses
    import torch

    torch.set_num_threads(2)
    snapshot = Path("pretrained_models/vision/c-radio-v4-so400m-hf").resolve()
    vit = dataclasses.replace(RADIO_VISION_BACKBONE, init_path=str(snapshot), resize_mode="siglip")
    config = MolmoVisionBackboneConfig(
        vit=vit, vit_layers=(-1,), skip_unused_layers=False,
        pooling_attention_mask=True, compile_vit=None, compile_connector=None,
    )
    backbone = MolmoVisionBackbone(config, LLMS["qwen3_4b_instruct"], device="cpu")
    backbone.reset_with_pretrained_weights()
    # Freeze the expensive tower for this CPU-only gradient check; verify that
    # the real 2x2 crop path still trains the Molmo2 pooling/projector modules.
    backbone.image_vit.requires_grad_(False)
    backbone.train()
    data = multicrop(np.full((896, 896, 3), 128, dtype=np.uint8))
    assert data.images.shape[0] == 5
    assert data.token_pooling.shape[0] == 1040
    images = torch.from_numpy(data.images).unsqueeze(0)
    pooling = torch.from_numpy(data.token_pooling.astype(np.int64)).unsqueeze(0)
    features = backbone(images, None, pooling)
    assert features.shape[0] == data.token_pooling.shape[0], features.shape
    assert torch.isfinite(features).all()
    features.float().square().mean().backward()
    connector_grads = [p.grad for p in backbone.get_connector_parameters() if p.requires_grad]
    assert connector_grads and any(g is not None and torch.isfinite(g).all() for g in connector_grads)
    print(f"PASS Molmo2 connector CPU forward/backward: {tuple(features.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--real-forward", action="store_true")
    parser.add_argument("--real-connector", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output/cradio_multicrop_grounding"))
    args = parser.parse_args()
    image_pre, multicrop = make_preprocessor()
    synthetic_checks(image_pre, multicrop)
    if args.visualize:
        visualize_real_data(image_pre, multicrop, args.output_dir, args.count)
    if args.real_forward:
        real_cpu_forward()
    if args.real_connector:
        real_connector_cpu_forward(multicrop)


if __name__ == "__main__":
    main()
