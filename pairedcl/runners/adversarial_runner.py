import torch
import numpy as np
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

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def evaluate_accuracy(self, agent, loader: DataLoader, eval_batches: int = 5):
        """Quick accuracy evaluation on a DataLoader."""
        agent.model.eval()
        correct = total = 0
        with torch.no_grad():
            for _ in range(eval_batches):
                imgs, labels = next(loader)
                _, preds, _, _ = agent.act(imgs)
                correct += (preds.to(labels.device) == labels).sum().item()
                total += labels.size(0)
        return correct / total

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------
    def agent_rollout(self,
                      agent,
                      num_steps: int,
                      update: bool = False,
                      is_env: bool = False):
        """Perform either an environment‑adversary (π_E) rollout or a supervised
        classifier rollout, depending on *is_env*.

        When *is_env=True* we treat π_E as the *acting* policy and collect one
        PPO transition, optionally performing a PPO update (when *update=True*
        and enough transitions are stored).

        When *is_env=False* we fine‑tune *agent* on *num_steps* minibatches from
        the current ClassificationEnv task.
        """
        stats = {}

        # --------  Environment‑adversary rollout (reinforcement learning)  ----
        if is_env:
            # 1) π_E proposes a TaskSpec
            with torch.no_grad():
                action_e = self.adversary_env.act(self.context_obs.unsqueeze(0)).squeeze(0)
            task_spec = self.adversary_env.action_to_taskspec(action_e)

            # 2) Reset the classification env to that task
            obs, _ = self.class_env.reset(task_spec=task_spec)

            # 3) Inner‑loop supervised updates of the protagonist
            for _ in range(self.k_inner):
                imgs, labels = next(self.class_env.iterator)
                self.agent.update(imgs, labels)

            # 4) Evaluate protagonist & antagonist accuracies
            acc_pro = self._eval_accuracy(self.agent)
            acc_ant = self._eval_accuracy(self.antagonist)
            reward  = max(acc_ant - acc_pro, 0.0)

            # 5) Store transition for PPO
            mask = torch.ones(1, 1, device=self.device)  # never "done" in this setting
            value, _, logp, _ = self.adversary_env.value_logp(
                self.context_obs.unsqueeze(0), action_e.unsqueeze(0)
            )
            self.storage.insert(self.context_obs, action_e, logp, value, torch.tensor([[reward]]), mask)

            stats.update(dict(acc_pro=acc_pro, acc_ant=acc_ant, reward_env=reward))

            # 6) Optional PPO update
            if update and self.storage.size() >= self.rollout_len:
                with torch.no_grad():
                    last_val, _, _, _ = self.adversary_env.value_logp(
                        self.context_obs.unsqueeze(0), action_e.unsqueeze(0)
                    )
                self.storage.compute_returns_and_advantages(last_val, use_gae=True)
                v_loss, p_loss, entropy = self.ppo.update(self.storage)
                self.storage.after_update()
                stats.update(dict(policy_loss=p_loss, value_loss=v_loss, entropy=entropy))

            return stats

        # --------  Protagonist rollout (supervised learning)  ---------------
        # Reset env with its current task (no new TaskSpec here)
        obs, _ = self.class_env.reset()
        losses, accs = [], []

        for _ in range(num_steps):
            imgs, labels = next(self.class_env.iterator)
            loss = agent.update(imgs, labels)
            losses.append(loss)

            with torch.no_grad():
                _, pred, _, _ = agent.act(imgs)
                accs.append((pred.to(labels.device) == labels).float().mean().item())

        stats.update(dict(avg_loss=float(np.mean(losses)),
                          avg_acc=float(np.mean(accs))))
        return stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self):
        """Single interaction step: task proposal ➜ adaptation ➜ reward ➜ PPO update.

        The original implementation duplicated the logic now contained in
        *agent_rollout*. We delegate the whole process to that utility and
        simply post‑process the returned statistics to keep the public API
        unchanged.
        """
        rollout_stats = self.agent_rollout(
            agent=self.agent,
            num_steps=self.k_inner,
            update=True,         # perform PPO update when rollout buffer is full
            is_env=True          # run the environment‑adversary branch
        )

        # Maintain backward‑compatibility with the original return signature
        return {
            "acc_pro":  rollout_stats.get("acc_pro"),
            "acc_ant":  rollout_stats.get("acc_ant"),
            "reward_e": rollout_stats.get("reward_env")
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _eval_accuracy(self, agent):
        """Helper used by *agent_rollout* to quickly evaluate accuracy on the
        current ClassificationEnv task using the env‑configured DataLoader
        settings."""
        loader = DataLoader(
            self.class_env.env.dataset,
            batch_size=self.class_env.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return self.evaluate_accuracy(agent, loader, eval_batches=5)
