import torch
from PIL import Image
from torch.utils.data import Dataset


class TreeDataset(Dataset):
    def __init__(self, records, img_root, class_to_idx, transform, use_metadata):
        """
        records: list of dicts with keys ['img_path','angle_deg', 'label_id', …]
        """
        self.records = records
        self.img_root = img_root
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_metadata = use_metadata

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img = Image.open(f"{self.img_root}/{rec['img_path']}").convert("RGB")
        img = self.transform(img)

        if self.use_metadata:
            θ = rec["angle_deg"] * torch.pi / 180.0
            meta = torch.tensor([torch.sin(θ), torch.cos(θ)], dtype=torch.float32)
        else:
            meta = torch.zeros(0)  # placeholder

        label = torch.tensor(self.class_to_idx[rec["label_id"]], dtype=torch.long)
        return img, meta, label
