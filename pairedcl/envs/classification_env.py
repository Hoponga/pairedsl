# classification_env.py  ─────────────────────────────────────────────
from __future__ import annotations
import itertools, gym, torch
from gym import spaces
from torch.utils.data import DataLoader, ConcatDataset
from typing import Sequence, Optional, Tuple, Dict, Any, List
import math 
import random 

from .task_spec import TaskSpec, TransformSpec



class MixtureDataset(ConcatDataset):
    """
    A ConcatDataset that *cycles* underlying datasets up to `fixed_len`,
    but presents them in a fixed, pre-shuffled order.
    """
    def __init__(self, datasets: Sequence[Dataset], fixed_len: int, seed: int = None):
        super().__init__(datasets)
        self.fixed_len = fixed_len

        # cumulative lengths of each sub-dataset
        self.cums = list(itertools.accumulate(len(d) for d in datasets))
        self.total_len = self.cums[-1]

        # build a repeated index array [0,1,2,…,total_len-1, 0,1,2,…] cut to fixed_len
        reps = math.ceil(fixed_len / self.total_len)
        all_idx = list(range(self.total_len)) * reps
        all_idx = all_idx[:fixed_len]

        # shuffle with optional seed
        # if seed is not None:
        #     random.seed(seed)
        random.shuffle(all_idx)

        self.mapping = all_idx

    def __len__(self):
        return self.fixed_len

    def __getitem__(self, idx):
        # look up the shuffled “base” index
        base_idx = self.mapping[idx]

        # now map that base_idx into (dataset_i, sample_j)
        # binary search on cums
        lo, hi = 0, len(self.cums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if base_idx < self.cums[mid]:
                hi = mid
            else:
                lo = mid + 1

        ds_idx = lo
        prev_cum = 0 if ds_idx == 0 else self.cums[ds_idx - 1]
        sample_idx = base_idx - prev_cum

        return self.datasets[ds_idx][sample_idx]



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
        self.default_task_spec = TaskSpec("mnist-train", [TransformSpec("permute", {"p": torch.arange(28*28)})])
      

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
            return self._init_from_specs(self.default_task_spec)
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
        """Optionally swap in new TaskSpec(s); return first observation batch."""
        if task_spec is not None:
            self._init_from_specs(task_spec)

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
