import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 
import pdb

class PPO():
    """
    Vanilla PPO
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_value_loss=True,
                 log_grad_norm=False):

        self.actor_critic = actor_critic    

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_value_loss = clip_value_loss

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.log_grad_norm = log_grad_norm

    def _grad_norm(self):
        total_norm = 0
        for p in self.actor_critic.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def update(self, rollouts):
        if rollouts.use_popart:
            value_preds = rollouts.denorm_value_preds
        else:
            value_preds = rollouts.values

        advantages = rollouts.returns[:-1] - value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loc_entropy_epoch = 0
        obj_entropy_epoch = 0

        if self.log_grad_norm:
            grad_norms = []

        for e in range(self.ppo_epoch):
            #print(advantages.shape, self.num_mini_batch)
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch)
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if rollouts.use_popart:
                    self.actor_critic.popart.update(return_batch)
                    return_batch = self.actor_critic.popart.normalize(return_batch)

                if self.clip_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                else:
                    value_loss = F.smooth_l1_loss(values, return_batch)

                self.optimizer.zero_grad()
                if isinstance(dist_entropy, list):
                    loss = (value_loss * self.value_loss_coef + action_loss - (dist_entropy[0]+dist_entropy[1]) * self.entropy_coef)
                else:
                    loss = (value_loss*self.value_loss_coef + action_loss - dist_entropy*self.entropy_coef)

                loss.backward()

                if self.log_grad_norm:
                    grad_norms.append(self._grad_norm())

                #print(self._grad_norm())

                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                            self.max_grad_norm)
                    
                self.optimizer.step()
                                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                if isinstance(dist_entropy, list):
                    obj_entropy_epoch += dist_entropy[0].item()
                    loc_entropy_epoch += dist_entropy[1].item()
                else:
                    dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        if isinstance(dist_entropy, list):
            obj_entropy_epoch /= num_updates
            loc_entropy_epoch /= num_updates
            dist_entropy_epoch = {'obj': obj_entropy_epoch, 'loc': loc_entropy_epoch}
        else:
            dist_entropy_epoch /= num_updates

        info = {}
        if self.log_grad_norm:
            info = {'grad_norms': grad_norms}

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, info