import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gym

from pairedcl.envs.task_spec import TaskSpec, TransformSpec
from typing import Union 


class TaskGenerator(nn.Module):
    """Generic policy/value network that outputs a continuous Box action of
    arbitrary dimension.  A user-supplied *param_defs* list maps each action
    dimension (range –1 … 1 after tanh‑squash) to a concrete transformation
    parameter and range, so the same module can drive any task manifold.

    Example – rotation+noise for MNIST
    ----------------------------------
    >>> param_defs = [
    ...     dict(tname="rotate",   pkey="deg",   low=0.0,   high=360.0),
    ...     dict(tname="gauss_noise", pkey="sigma", low=0.0, high=0.3),
    ... ]
    >>> tg = TaskGenerator(param_defs, base_dataset="mnist-train", device="cuda")
    >>> act  = tg.act(torch.zeros(1))           # sample 2‑D action
    >>> spec = tg.action_to_taskspec(act)
    """

    def __init__(
        self,
        param_defs: list,
        *,
        obs_dim: int = 1,
        hidden_dim: int = 64,
        base_dataset: str = "mnist-train",
        device: Union[str, torch.device]= "cpu",
    ):
        super().__init__()

        self.param_defs   = param_defs              # list of dicts
        self.base_dataset = base_dataset
        self.device       = torch.device(device)
        self.action_dim   = len(param_defs)

        # Small MLP policy/value
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head  = nn.Linear(hidden_dim, self.action_dim)
        self.log_std  = nn.Parameter(torch.zeros(self.action_dim))
        self.v_head   = nn.Linear(hidden_dim, 1)

        # Expose Gym spaces for buffers / PPO
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                           shape=(self.action_dim,), dtype=np.float32)
        self.obs_shape    = (obs_dim,)

    # ------------- internal helpers ----------------------------------
    def _forward_base(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs.float())

    def forward(self, obs: torch.Tensor):
        h = self._forward_base(obs.to(self.device))
        mu = self.mu_head(h)
        std = torch.exp(self.log_std).expand_as(mu)
        value = self.v_head(h)
        return mu, std, value

    # -----------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Return squashed action in (‑1,1)."""
        mu, std, _ = self.forward(obs)
        raw = mu if deterministic else Normal(mu, std).sample()
        action = torch.tanh(raw)
        return action.squeeze(0)

    def value_logp(self, obs: torch.Tensor, action: torch.Tensor):
        """Compute value estimate and log‑prob under current policy for *given*
        squashed action (tanh‑space)."""
        mu, std, value = self.forward(obs)
        # inverse tanh (atanh)
        atanh = 0.5 * (torch.log1p(action) - torch.log1p(-action))
        log_det_jacob = torch.log1p(-action.pow(2) + 1e-6).sum(-1, keepdim=True)
        logp = Normal(mu, std).log_prob(atanh).sum(-1, keepdim=True) - log_det_jacob
        return value, atanh, logp, None

    # -----------------------------------------------------------------
    def action_to_taskspec(self, action: torch.Tensor) -> TaskSpec:
        """Translate a (D,) action tensor into a **TaskSpec** using *param_defs*.

        Each entry in *param_defs* must be a dict with keys:
          • tname : str   ─ Transform name            (e.g. "rotate")
          • pkey  : str   ─ Parameter key in that transform (e.g. "deg")
          • low   : float ─ Minimum value in real range
          • high  : float ─ Maximum value in real range
        """
        action = torch.as_tensor(action, device=self.device).squeeze()
        assert action.numel() == self.action_dim, "Action dimension mismatch"

        transforms = []
        for a, pd in zip(action, self.param_defs):
            # scale from (-1,1) → (low, high)
            val = ((a.item() + 1) / 2) * (pd["high"] - pd["low"]) + pd["low"]
            transforms.append(
                TransformSpec(pd["tname"], {pd["pkey"]: val})
            )
        return TaskSpec(dataset_id=self.base_dataset, transforms=transforms)
