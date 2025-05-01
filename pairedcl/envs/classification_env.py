# classification_env.py  ─────────────────────────────────────────────
from __future__ import annotations
import itertools, gym, torch
from gym import spaces
from torch.utils.data import DataLoader, ConcatDataset
from typing import Sequence, Optional, Tuple, Dict, Any, List
import math 
import random 

from .task_spec import TaskSpec, TransformSpec

import math, random, bisect
from torch.utils.data import Dataset

class MixtureDataset(Dataset):
    """
    A fixed‐length dataset that draws IID (with replacement) from the union
    of multiple datasets, pre‐shuffling once at construction.

    Each sample is chosen by:
      1) picking a flat index in [0, total_len) uniformly at random
      2) mapping that flat index back to (dataset_i, sample_j)
    """

    def __init__(self, datasets, fixed_len, seed=None):
        super().__init__()
        self.datasets  = list(datasets)
        self.fixed_len = fixed_len

        # 1) cumulative lengths and total
        lengths = [len(d) for d in self.datasets]
        self.cums  = list(itertools.accumulate(lengths))
        total     = self.cums[-1]

        flat_idxs = random.choices(range(total), k=fixed_len)

        # 3) map each flat index → (dataset_i, sample_j)
        self.mapping = []
        for idx in flat_idxs:
            ds_i = bisect.bisect_right(self.cums, idx)
            prev = self.cums[ds_i-1] if ds_i > 0 else 0
            sample_j = idx - prev
            self.mapping.append((ds_i, sample_j))

    def __len__(self):
        return self.fixed_len

    def __getitem__(self, idx):
        ds_i, sample_j = self.mapping[idx]
        return self.datasets[ds_i][sample_j]



class ClassificationEnv(gym.Env):
    """Streams batches from one *or many* TaskSpecs for continual-learning."""

    metadata = {"render.modes": []}

    # ------------------------------------------------------------------
    # ctor
    # ------------------------------------------------------------------
    def __init__(
        self,
        task_spec: TaskSpec | Sequence[TaskSpec],
        *,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
        shuffle: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.device     = torch.device(device)
        self.shuffle    = shuffle
        self.task_specs = task_spec 
        #self.default_task_spec = TaskSpec("mnist-train", [TransformSpec("permute", {"p": torch.arange(28*28)})])
      

        # allow multiple task specs to be passed in as a single arg 
        self._init_from_specs(task_spec)

    


    def _init_from_specs(self, specs: TaskSpec | Sequence[TaskSpec]) -> None:
        """Accept a single TaskSpec or an iterable; build a mixed DataLoader."""
        # Normalise to list
        if isinstance(specs, TaskSpec):
            specs = [specs]
        elif not isinstance(specs, (list, tuple)):
            specs = list(specs)
        if not specs: 
            return self._init_from_specs(self.task_specs) # if nothing is given just propose all the tasks 
            #return self._init_from_specs(self.default_task_spec)
        self.task_specs: List[TaskSpec] = specs

        # Build datasets and concat
        datasets = [spec.make_dataset(train=True) for spec in specs]
        
        dataset  = datasets[0] if len(datasets) == 1 else MixtureDataset(datasets, len(datasets[0]))

        self.dataset_size= len(dataset) 


        # DataLoader & endless iterator
        self.loader    = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )
        self.iterator  = itertools.cycle(self.loader)

        # Observation / action spaces
        sample, _ = next(iter(self.loader))
        data_shape            = sample.shape[1:]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.batch_size, *data_shape),
            dtype="float32",
        )

        # Derive #classes from union of targets
        all_targets = []
        for ds in datasets:
            # torch Subset stores .dataset.targets
            targets = getattr(ds, "targets", None) or getattr(ds.dataset, "targets")
            all_targets.extend(targets)
        n_classes = len(set(int(t) for t in all_targets))

        self.action_space = spaces.MultiDiscrete([n_classes] * self.batch_size)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        task_spec: Optional[TaskSpec | Sequence[TaskSpec]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._init_from_specs(task_spec or self.task_specs) # holy another goated one liner 

        self.obs, self.labels = next(self.iterator)
        self.obs, self.labels = self.obs.to(self.device), self.labels.to(self.device)
        return self.obs, {}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)

        acc    = (action == self.labels).float().mean().item()
        reward = acc                                        # reward = accuracy

        self.obs, self.labels = next(self.iterator)
        self.obs, self.labels = self.obs.to(self.device), self.labels.to(self.device)

        return self.obs, reward, False, {"accuracy": acc, "labels": self.labels}

    # ------------------------------------------------------------------
    def render(self, mode="human"):
        pass
