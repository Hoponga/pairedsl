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
        max_gen_tasks = 10, # how many tasks are we allowed to select for "one environment" 
    ):
        super().__init__()

        self.param_defs   = param_defs              # list of dicts
        self.base_dataset = base_dataset
        self.device       = torch.device(device)

        self.max_gen_tasks = max_gen_tasks 
        self.single_task_action_dim = len(param_defs) + 1 # include gate 
        self.action_dim = self.max_gen_tasks*self.single_task_action_dim
        
        
        # Small MLP policy/value
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head  = nn.Linear(hidden_dim,self.action_dim)
        self.log_std  = nn.Parameter(torch.zeros(self.action_dim))
        self.v_head   = nn.Linear(hidden_dim, 1)
        
        # Expose Gym spaces for buffers / PPO
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                           shape=(self.action_dim,), dtype=np.float32)
        self.obs_shape    = (obs_dim,)

    # ------------- internal helpers ----------------------------------
    def _forward_base(self, obs: torch.Tensor):
        #print(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)

    def forward(self, obs: torch.Tensor):
        h = self._forward_base(obs.to(self.device))
        mu = self.mu_head(h)
        std = torch.exp(self.log_std).expand_as(mu)
        value = self.v_head(h)
        return mu, std, value

    # -----------------------------------------------------------------


    # we already do squeeze here 
    # inside class TaskGenerator

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Parameters
        ----------
        obs : Tensor
            Observation tensor shaped (B, obs_dim) or (obs_dim,) for a single batch.
        deterministic : bool, default False
            If True, use the mean (µ) of the Gaussian; else sample.

        Returns
        -------
        value : Tensor
            Critic estimate V(s) with shape (B, 1).
        action : Tensor
            Squashed action in (-1, 1) with shape (B, action_dim).
        action_log_dist : Tensor
            Log-probability of *action* under the current policy, shape (B, 1).
        """
        obs = obs.to(self.device)
        mu, std, value = self.forward(obs)              # (B, D), (B, D), (B, 1)
        dist = Normal(mu, std)

        raw = mu if deterministic else dist.rsample()   # rsample() keeps grad flow
        action = torch.tanh(raw)

        # log-probability in **squashed** (tanh) space
        log_prob_raw = dist.log_prob(raw).sum(dim=-1, keepdim=True)  # (B, 1)
        log_det_jac  = torch.log1p(-action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        action_log_dist = log_prob_raw - log_det_jac

        return value, action, action_log_dist

    

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        rnn_hxs=None,          # kept for API symmetry – ignored
        masks=None             # kept for API symmetry – ignored
    ):
        """
        Parameters
        ----------
        obs : Tensor
            (B, obs_dim) or (obs_dim,) observation batch.
        action : Tensor
            (B, action_dim) tensor of *squashed* (tanh) actions in (-1, 1).

        Returns
        -------
        value : Tensor          shape (B, 1)
        action_log_probs : Tensor  shape (B, 1)
        dist_entropy : Tensor      scalar mean entropy of the policy
        rnn_hxs : None            (placeholder for RNN APIs)
        """
        obs = obs.to(self.device)
        action = action.to(self.device)

        mu, std, value = self.forward(obs)          # (B, D), (B, D), (B, 1)
        
        dist = Normal(mu, std)

        # -------- log-probability of the *given* squashed action -------------
        # Inverse tanh
        atanh_action = 0.5 * (torch.log1p(action) - torch.log1p(-action))
        # Log-prob under the Gaussian before the squash
        log_prob_raw = dist.log_prob(atanh_action).sum(dim=-1, keepdim=True)
        # |det J| of the tanh transformation
        log_det_jac  = torch.log1p(-action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        action_log_probs = log_prob_raw - log_det_jac                      # (B,1)

        # -------- entropy ----------------------------------------------------
        # Entropy of the *unsquashed* Gaussian (common practice for PPO)
        dist_entropy = dist.entropy().sum(dim=-1).mean()                   # scalar

        return value, action_log_probs, dist_entropy, None 

    # -----------------------------------------------------------------
    def action_to_taskspec(self, action: torch.Tensor) -> TaskSpec:
        """Translate a (D,) action tensor into a **TaskSpec** using *param_defs*.

        Each entry in *param_defs* must be a dict with keys:
          • tname : str   ─ Transform name            (e.g. "rotate")
          • pkey  : str   ─ Parameter key in that transform (e.g. "deg")
          • low   : float ─ Minimum value in real range
          • high  : float ─ Maximum value in real range
        """
        action = torch.as_tensor(action, device=self.device).flatten()
        #print(action)
        assert action.numel() == self.single_task_action_dim - 1, "Action dimension mismatch DID YOU MAYBE FORGET TO REMOVE THE GATE!!!!!"

        transforms = []
        for a, pd in zip(action, self.param_defs):
            # scale from (-1,1) → (low, high)
            val = ((a.item() + 1) / 2) * (pd["high"] - pd["low"]) + pd["low"]
            pkey = pd["pkey"] if pd["pkey"] != "sigma" else "p"
            transforms.append(
                TransformSpec(pd["tname"], {pkey: self._permute_spec(val)})
            )
        return TaskSpec(dataset_id=self.base_dataset, transforms=transforms)



    # allow self.max_gen_tasks to be converted to task_specs all at once 
    def actions_to_taskspecs(self, action: torch.Tensor) -> list:
        """
        Decode a (M*(D+1),) tensor into ≤M TaskSpecs.
        Layout per task: (gate, param_0 … param_{D-1})
        """
        action = action.flatten()
        tasks  = []
        block  = self.single_task_action_dim
        for i in range(self.max_gen_tasks):
            g   = action[i*block]
            if g < 0:                       # gate off  → skip task
                continue
            pvec = action[i*block + 1 : (i+1)*block]
            #print(pvec)
            tasks.append(self.action_to_taskspec(pvec))
        return tasks
