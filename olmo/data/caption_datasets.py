"""Open-source caption datasets converted to local parquet under dataset/stage1/.

Covers DOCCI, TextCaps (FineVision release) and DCI. Every row supplies an
embedded image plus a single caption; the user-side prompts come from the
DataFormatter style pools ("long_caption" / "short_caption"), matching how
PixMoCap builds its caption messages.
"""
from pathlib import Path

import datasets

from olmo.data.dataset import Dataset

DATA_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "stage1"


class CaptionDataset(Dataset):
    PATH: str = ""
    NAME: str = ""
    STYLE: str = "long_caption"

    def __init__(self, split: str = "train"):
        self.split = split
        self.data = datasets.load_dataset(self.PATH, split=split, keep_in_memory=False)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        return dict(
            image=ex["image"],
            message_list=[dict(text=ex["caption"], style=self.STYLE)],
            metadata=dict(image_id=ex["id"], dataset=self.NAME),
        )


class Docci(CaptionDataset):
    PATH = str(DATA_ROOT / "DOCCI")
    NAME = "docci"
    STYLE = "long_caption"


class TextCaps(CaptionDataset):
    PATH = str(DATA_ROOT / "TextCaps-FineVision")
    NAME = "textcaps"
    STYLE = "short_caption"


class Dci(CaptionDataset):
    PATH = str(DATA_ROOT / "DCI")
    NAME = "dci"
    STYLE = "long_caption"
