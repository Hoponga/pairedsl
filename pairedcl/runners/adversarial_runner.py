# pairedcl/runners/adversarial_runner.py

import torch
from torch.utils.data import DataLoader

class AdversarialRunner:
    def __init__(self, *,
                 agent,              # your supervised ClassifierAgent
                 antagonist,         # frozen copy of agent or same API
                 adversary_env,      # TaskGenerator π_E
                 class_env,          # ClassificationEnv
                 ppo,                # PPO updater for π_E
                 storage,            # RolloutStorageCL for π_E
                 k_inner_updates: int,
                 rollout_length: int,
                 device: str = "cpu"):
        self.agent         = agent
        self.antagonist    = antagonist
        self.adversary_env = adversary_env
        self.class_env     = class_env
        self.ppo           = ppo
        self.storage       = storage
        self.k_inner       = k_inner_updates
        self.rollout_len   = rollout_length
        self.device        = torch.device(device)
        # a trivial “context” for π_E; could be richer later
        self.context_obs   = torch.zeros(1, device=self.device)

    def evaluate_accuracy(self, agent, loader: DataLoader, eval_batches: int = 5):
        agent.model.eval()
        correct = total = 0
        with torch.no_grad():
            for _ in range(eval_batches):
                imgs, labels = next(loader)
                _, preds, _, _ = agent.act(imgs)
                correct += (preds.to(labels.device) == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def run(self):
        # adversary picks a new task 
        with torch.no_grad():
            action_e = self.adversary_env.act(self.context_obs.unsqueeze(0)).squeeze(0)
        task_spec = self.adversary_env.action_to_taskspec(action_e)

        obs, _ = self.class_env.reset(task_spec=task_spec)
        # Create a fresh DataLoader for supervised updates
        loader = DataLoader(self.class_env.env.dataset,
                            batch_size=self.class_env.batch_size,
                            shuffle=self.class_env.shuffle,
                            drop_last=True)

        for _ in range(self.k_inner):
            imgs, labels = next(loader)      # CPU tensors
            self.agent.update(imgs, labels)

        # ==== 4) Evaluate both learners on this task ====
        acc_pro = self.evaluate_accuracy(self.agent,     loader)
        acc_ant = self.evaluate_accuracy(self.antagonist, loader)

        # ==== 5) Compute reward & store transition for π_E ====
        reward_e = max(acc_ant - acc_pro, 0.0)
        mask = 1.0   # always “not done” in infinite-horizon setting
        self.storage.insert(self.context_obs, action_e, reward_e, mask)

        # ==== 6) Possibly update π_E when rollout is ready ====
        if self.storage.size() >= self.rollout_len:
            self.storage.compute_returns_and_advantages(last_value=0, use_gae=False)
            self.ppo.update(self.storage)
            self.storage.after_update()

        return {
            "acc_pro": acc_pro,
            "acc_ant": acc_ant,
            "reward_e": reward_e
        }