from __future__ import annotations
import itertools, gym, torch
from gym import spaces
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any

from .task_spec import TaskSpec

class ClassificationEnv(gym.Env):
    """Dataset-agnostic streaming environment for continual-learning PAIRED."""
    metadata = {"render.modes": []}

    def __init__(
        self,
        task_spec: TaskSpec,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
        shuffle: bool = True,
    ):
        super().__init__()
        self.batch_size, self.device, self.shuffle = batch_size, torch.device(device), shuffle
        self._init_from_spec(task_spec)

    # ---------- internal ----------
    def _init_from_spec(self, spec: TaskSpec):
        self.task_spec = spec
        ds = spec.make_dataset(train=True)
        self.loader = DataLoader(ds, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)
        self.iterator = itertools.cycle(self.loader)

        # observation / action spaces
        sample, _ = next(iter(self.loader))
        C, H, W = sample.shape[1:]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.batch_size, C, H, W), dtype="float32")
        n_classes = len(set(ds.dataset.targets))  # works for torch Subset as well
        self.action_space = spaces.MultiDiscrete([n_classes] * self.batch_size)

    # ---------- gym API ----------
    def reset(self, *, task_spec: Optional[TaskSpec] = None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        if task_spec is not None:
            self._init_from_spec(task_spec)
        self.obs, self.labels = next(self.iterator)
        self.obs, self.labels = self.obs.to(self.device), self.labels.to(self.device)
        return self.obs, {}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)
        acc = (action == self.labels).float().mean().item()
        reward = acc
        self.obs, self.labels = next(self.iterator)
        self.obs, self.labels = self.obs.to(self.device), self.labels.to(self.device)
        return self.obs, reward, False, {"accuracy": acc}

    def render(self, mode="human"): pass