# pairedcl/algorithms/storage.py

from __future__ import annotations  # safe for <3.10 “|” operator

from typing import Union, Optional
import torch
import numpy as np



numpy_to_torch = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32:   torch.int32,
    np.int64:   torch.int64,
}
class RolloutStorage:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_space,          # only .shape & dtype needed
        gamma: float,
        gae_lambda: float,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.num_steps   = num_steps
        self.num_envs    = num_envs
        self.device      = torch.device(device)
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda

        # --- main buffers --------------------------------------------------
        # The obs shape is currently just a single integer, seems sketchy 

        self.obs      = torch.zeros((num_steps, num_envs, *obs_shape),
                                    device=self.device)
        self.actions  = torch.zeros((num_steps, num_envs, *action_space.shape),device=self.device)
        self.logprobs = torch.zeros((num_steps, num_envs, 1), device=self.device)
        self.values   = torch.zeros((num_steps, num_envs, 1), device=self.device)
        self.rewards  = torch.zeros((num_steps, num_envs, 1), device=self.device)
        self.masks    = torch.ones( (num_steps, num_envs, 1), device=self.device)

        # computed later
        self.returns     = torch.zeros_like(self.values)
        self.advantages  = torch.zeros_like(self.values)

        self.step = 0  # write-pointer

    def insert(
        self,
        obs:     torch.Tensor,
        action:  torch.Tensor,
        logp:    torch.Tensor,
        value:   torch.Tensor,
        reward:  torch.Tensor,
        mask:    torch.Tensor,
    ) -> None:
        """Store a single transition at the current write pointer."""
        if self.step >= self.num_steps:
            raise RuntimeError("RolloutCLStorage is full — call after_update() first.")

        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(action)
        self.logprobs[self.step].copy_(logp)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step].copy_(mask)

        self.step += 1

    # --------------------------------------------------------------------- #
    # Book-keeping
    # --------------------------------------------------------------------- #
    def size(self) -> int:
        return self.step

    def after_update(self) -> None:
        """Clear the buffer (cheaply) by resetting the write pointer."""
        self.step = 0

    # --------------------------------------------------------------------- #
    # Advantage / return computation (GAE or plain MC)
    # --------------------------------------------------------------------- #
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        use_gae: bool = True,
    ) -> None:
        """
        Populate self.returns and self.advantages in-place.

        last_value : V(s_T) for bootstrap (shape [num_envs, 1])
        """
        if use_gae:
            gae = torch.zeros_like(last_value)
            for t in reversed(range(self.step)):
                delta = (self.rewards[t] +
                         self.gamma * last_value * self.masks[t] -
                         self.values[t])
                gae = delta + self.gamma * self.gae_lambda * self.masks[t] * gae
                self.advantages[t] = gae
                self.returns[t]    = self.advantages[t] + self.values[t]
                last_value = self.values[t]
        else:  # plain discounted return
            running_return = last_value
            for t in reversed(range(self.step)):
                running_return = self.rewards[t] + self.gamma * running_return * self.masks[t]
                self.returns[t] = running_return
            self.advantages = self.returns - self.values