# train.py  ────────────────────────────────────────────────────────────
"""
Puts the pieces together to train PAIRED‐CL on Permuted MNIST.

• TaskGenerator outputs a 1-D action ∈ (–1,1) that we scale to a 32-bit seed.
• That seed deterministically generates a pixel permutation for MNIST.
• ClassificationEnv applies the permutation; classifier learns online.
• PPO trains π_E to find permutations the protagonist hasn’t mastered.
"""

import sys, pprint


import argparse, time, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
torch.autograd.set_detect_anomaly(True)
import wandb

# --- import your modules -----------------------------------------------------
from pairedcl.envs.task_spec          import TaskSpec, TransformSpec
from pairedcl.envs.classification_env import ClassificationEnv
from pairedcl.envs.task_generator     import TaskGenerator
from pairedcl.agents.classifier_agent import ClassifierAgent
from pairedcl.algos.storage      import RolloutStorage
from pairedcl.algos.ppo          import PPO       
from pairedcl.runners.adversarial_runner import AdversarialRunner    # with agent_rollout


# -----------------------------------------------------------------------------
# simple MLP for MNIST
class MLP(nn.Module):
    def __init__(self, in_dim=28*28, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, in_dim), # anti permutation layer??!?!!?!?!?
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

# -----------------------------------------------------------------------------
def seed_all(seed=0):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def main():
    from pairedcl.utils.arguments import get_args 
    # p = argparse.ArgumentParser()
    # p.add_argument("--epochs", type=int, default=100)
    # p.add_argument("--k_inner", type=int, default=3,
    #                help="batches to train classifier per task")
    # p.add_argument("--rollout", type=int, default=16,
    #                help="tasks per PPO update")
    # p.add_argument("--device", default="cpu")
    # args = p.parse_args()
    args = get_args()
    args.context_obs_shape += args.action_dim*args.max_gen_tasks
    seed_all(0)
    device = torch.device(args.device)
    run = wandb.init(                       # REQUIRED
        project=args.wandb_project,          # your project name
        name=f"seed{args.seed}",            # an optional run name
        config=vars(args),                  # saves all CLI flags
        save_code=True                      # logs the entire script
    )

    # a dummy identity TaskSpec to bootstrap env
    id_spec = TaskSpec("mnist-train", transforms=[])
    class_env = ClassificationEnv(id_spec, batch_size=64, device=device)

    # --- protagonists --------------------------------------------------------
    model = MLP().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    agent = ClassifierAgent(model, opt,
                            argparse.Namespace(device=device),
                            num_envs=1)

    
    antagonist_model = MLP().to(device)
    antagonist = ClassifierAgent(antagonist_model,
                                 optim.Adam(antagonist_model.parameters(), lr=1e-3),
                                 argparse.Namespace(device=device),
                                 num_envs=1)  # keep frozen / update occasionally

    # --- environment adversary ----------------------------------------------

    # # RL storage + PPO for π_E
    # storage = RolloutStorage(
    #     num_steps=args.rollout,
    #     num_envs=1,
    #     obs_shape=(1,),
    #     action_space=tg.action_space,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     device=device,
    # )
    # ppo = PPO(model=tg,
    #           lr=3e-4,
    #           eps=0.2,
    #           epochs=4,
    #           value_loss_coef=0.5,
    #           entropy_coef=0.01,
    #           max_grad_norm=0.5)

    # --- runner --------------------------------------------------------------
    # replace the “RL storage + PPO” block
    from pairedcl.utils.make_agent import make_agent
    from pairedcl.utils.evaluator import Evaluator

    evaluator = Evaluator(args.task_type, class_env, T=args.max_gen_tasks, batches_per_task=8, device=args.device)

    adversary_model, ppo, storage = make_agent(args, device, tasks = evaluator.task_specs)


    runner = AdversarialRunner(
        agent=agent,
        antagonist=antagonist,
        adversary_env=adversary_model,   # <- now the actor-critic module
        evaluator=evaluator,
        class_env=class_env,
        ppo=ppo,
        storage=storage,
        k_inner_updates=args.k_inner,
        rollout_length=args.rollout,
        device=device,
        antagonist_delta = args.antagonist_delta
    )

    # inside your training loop
   
    

    # --- training loop -------------------------------------------------------
    for epoch in range(args.epochs):
        info_env, info_cls = runner.run()
        log_dict = {"epoch": epoch}
        log_dict.update({f"env/{k}": v for k, v in info_env.items()})
        log_dict.update({f"cls/{k}": v for k, v in info_cls.items()})
        
        wandb.log(log_dict, step = epoch)
        
        if epoch % 10 == 0:
            eval_acc = evaluator.evaluate(agent)
            wandb.log({"eval/accuracy": eval_acc}, step=epoch)
            print(f"[{epoch:04d}] "
                  f"env_reward={info_env.get('reward_env',0):.3f}  "
                  f"cls_acc={info_cls['avg_acc']:.3f}   "
                  f"cls_acc_ant={info_cls['avg_acc_ant']:.3f}   "
                  f"eval_acc={eval_acc:.3f}   "
                  f"loss={info_cls['avg_loss']:.4f}")

    run.finish()

if __name__ == "__main__":
    main()