from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Sequence, Optional

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

from pairedcl.data.datasets import get_base_dataset

# ---------- helper registry ----------
_transform_registry = {
    "permute": lambda p: lambda x: x.view(-1)[p].view_as(x),
    "rotate":  lambda deg: T.RandomRotation((deg, deg)),
    "gauss_noise": lambda sigma: lambda x: x + torch.randn_like(x) * sigma,
    # register more here …
}



# Transform specification for a particular transformation given by the above transform registry 
@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def build(self):
        fn = _transform_registry[self.name]
        return fn(**self.params) if callable(fn) else fn




# Task specification -- abstraction for a UED parameter selection -- 
# given a base dataset, sequentially apply a list of transforms o n this dataset by using the TransformedDataset wrapper 
@dataclass(frozen=True)
class TaskSpec:
    dataset_id: str                        # e.g. "mnist-train", "core50-s10"
    transforms: List[TransformSpec]        # order matters
    class_subset: Optional[Sequence[int]] = None
    seed: int = 0                          # reproducibility

    def make_dataset(self, *, train: bool = True) -> Dataset:
        base_ds = get_base_dataset(self.dataset_id, train=train, seed=self.seed)
        if self.class_subset is not None:                       # filter classes
            idx = [i for i, y in enumerate(base_ds.targets) if y in self.class_subset]
            base_ds = torch.utils.data.Subset(base_ds, idx)

        transform_pipeline = T.Compose([ts.build() for ts in self.transforms])
        return _TransformedDataset(base_ds, transform_pipeline)



# Wrapper on a dataset to apply a transform elementwise to data items 
class _TransformedDataset(Dataset):
    def __init__(self, root_ds: Dataset, transform):
        self.dataset, self.transform = root_ds, transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y