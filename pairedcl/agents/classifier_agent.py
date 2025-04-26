# pairedcl/agents/classifier_agent.py

import torch
import torch.nn.functional as F

class ClassifierAgent:
    def __init__(self, model, optimizer, args, num_envs):
        """
        model     : nn.Module mapping obs → logits [B, C]
        optimizer : torch.optim.Optimizer (e.g. Adam)
        args      : argparse.Namespace with .device
        num_envs  : ignored for supervised training
        """
        self.device    = torch.device(args.device)
        self.model     = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_envs = num_envs 

    def act(self, obs, hidden=None, mask=None):
        """
        RL‐style signature, but here we just do
        a forward pass & greedy classification.

        Returns:
          value_est  = None
          action     = predicted class indices [B]
          log_dist   = log‐softmax logits [B, C]
          new_hidden = None
        """
        obs = obs.to(self.device)
        with torch.no_grad():
            logits = self.model(obs)                      # [B, C]
            action = torch.argmax(logits, dim=1)          # [B]
            log_dist = F.log_softmax(logits, dim=1)       # [B, C]
        return None, action.cpu(), log_dist.cpu(), None

    def process_action(self, action_cpu):
        # identity mapping: action is already class index
        return action_cpu

    def update(self, obs, labels):
        """
        Supervised step on one minibatch.

        Args:
          obs    : tensor [B, C, H, W] or [B, features...] 
          labels : tensor [B] of int class IDs

        Returns:
          loss.item() as float
        """
        obs    = obs.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        logits = self.model(obs)                 # [B, C]
        loss   = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()