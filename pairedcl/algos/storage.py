# pairedcl/algorithms/storage.py

from __future__ import annotations  # safe for <3.10 “|” operator

from typing import Union, Optional
import torch
import numpy as np
from torch.utils.data.sampler import \
    BatchSampler, SubsetRandomSampler, SequentialSampler



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

        self.obs      = torch.zeros((num_steps + 1, num_envs, *obs_shape),
                                    device=self.device)
        self.actions  = torch.zeros((num_steps, num_envs, *action_space.shape),device=self.device)
        self.logprobs = torch.zeros((num_steps, num_envs, 1), device=self.device)
        self.values   = torch.zeros((num_steps + 1, num_envs, 1), device=self.device)
        self.rewards  = torch.zeros((num_steps, num_envs, 1), device=self.device)
        self.masks    = torch.ones( (num_steps + 1, num_envs, 1), device=self.device)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
            self.action_log_dist = torch.zeros(num_steps, num_envs, *action_space.shape)
            self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        elif action_space.__class__.__name__ == 'MultiDiscrete':
            action_shape = len(list(action_space.nvec))
            self.action_log_dist = torch.zeros(num_steps, num_envs, np.sum(list(action_space.nvec)))
            self.action_log_probs = torch.zeros(num_steps, num_envs, len(list(action_space.nvec)))
        else: # Hack it to just store action prob for sampled action if continuous
            action_shape = action_space.shape[0]
            self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
            self.action_log_dist = torch.zeros_like(self.action_log_probs)


        if action_space.__class__.__name__ in ['Discrete', 'MultiDiscrete']:
            self.actions = self.actions.long()


        self.is_dict_obs = False 

        # computed later
        self.returns     = torch.zeros_like(self.values)
        self.advantages  = torch.zeros_like(self.values)
        self.use_popart = False 
        self.step = 0  # write-pointer

    def insert(self, obs, actions, action_log_probs, action_log_dist,
               value_preds, rewards, mask):
        if len(rewards.shape) == 3: rewards = rewards.squeeze(2)

        if self.is_dict_obs:
            [self.obs[k][self.step + 1].copy_(obs[k]) for k in self.obs.keys()]
        else:
            self.obs[self.step + 1].copy_(obs)


        self.actions[self.step].copy_(actions) 
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.action_log_dist[self.step].copy_(action_log_dist)
        self.values[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    # --------------------------------------------------------------------- #
    # Book-keeping
    # --------------------------------------------------------------------- #
    def size(self) -> int:
        
        return self.step

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch


        #print(batch_size, mini_batch_size, self.obs.shape, self.actions.shape)

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False)
     
        for indices in sampler:
            if self.is_dict_obs:
                obs_batch = {k: self.obs[k][:-1].view(-1, *self.obs[k].size()[2:])[indices] for k in self.obs.keys()}
            else:
                obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]



            actions_batch = self.actions.view(-1,
                                            self.actions.size(-1))[indices]

            value_preds_batch = self.values[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]

            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

          

            yield obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


    def to(self, device): 
        self.device = device
        if self.is_dict_obs:
            for k in self.obs.keys():
                self.obs[k] = self.obs[k].to(device)
        else:
            self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_log_dist = self.action_log_dist.to(device)


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