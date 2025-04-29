# classification_env.py  ─────────────────────────────────────────────
from __future__ import annotations
import itertools, gym, torch
from gym import spaces
from torch.utils.data import DataLoader, ConcatDataset
from typing import Sequence, Optional, Tuple, Dict, Any, List

from .task_spec import TaskSpec, TransformSpec



# Dataset from concatenated dataset with a fixed size 
class MixtureDataset(ConcatDataset):
    """
    A ConcatDataset that *cycles* underlying datasets so that __len__()
    returns exactly `fixed_len`.  The DataLoader will see repeated samples
    if `fixed_len` > total unique items.
    """
    def __init__(self, datasets: Sequence, fixed_len: int):
        super().__init__(datasets)
        self.fixed_len = fixed_len
        # pre-compute cumulative lengths for fast index remap
        self.cums = list(itertools.accumulate(len(d) for d in datasets))

    def __len__(self):
        return self.fixed_len

    def __getitem__(self, idx):
        idx = idx % self.cums[-1]          # wrap around full concat length
        # binary search to locate dataset
        lo, hi = 0, len(self.cums)-1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cums[mid]:
                hi = mid
            else:
                lo = mid + 1
        ds_idx = lo
        prev_cum = 0 if ds_idx == 0 else self.cums[ds_idx-1]
        sample_idx = idx - prev_cum
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
