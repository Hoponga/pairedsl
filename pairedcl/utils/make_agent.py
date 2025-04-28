# pairedcl/algos/make_agent.py
from pairedcl.algos.storage import RolloutStorage
from pairedcl.algos.ppo     import PPO

import torch 
from pairedcl.envs.task_generator import TaskGenerator 
from pairedcl.envs.task_spec import TaskSpec, TransformSpec


def make_task_generator(device):
    # 1-D action → permutation seed in [0, 2^20)
    param_defs = [
        dict(tname="permute", pkey="seed", low=0.0, high=2**20 - 1)
    ]
    tg = TaskGenerator(param_defs,
                       base_dataset="mnist-train",
                       obs_dim=1,
                       hidden_dim=64,
                       device=device)
    # Monkey-patch: interpret 'seed' into permutation tensor
    def _permute_spec(pd_seed):
        g = torch.Generator().manual_seed(int(pd_seed))
        return torch.randperm(28*28, generator=g)
    tg._permute_spec = _permute_spec
    # Override action_to_taskspec for permuted MNIST
    def _a2spec(self, action):
        seed_val = ((action[0].item() + 1)/2) * (2**20 -1)
        perm = self._permute_spec(seed_val)
        ts = TaskSpec("mnist-train", [TransformSpec("permute", {"p": perm})])
        return ts
    tg.action_to_taskspec = _a2spec.__get__(tg, TaskGenerator)
    return tg

def make_agent(args, device="cpu"):
    actor_critic = make_task_generator(device)
    


    

    storage = RolloutStorage(
        num_steps=args.num_steps,
        num_envs=1,
        obs_shape=(1,),
        action_space=actor_critic.mu_head.weight.new_zeros(1),  # (1,)
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )

    algo = PPO(
            actor_critic=actor_critic,
            clip_param=args.clip_param,
            ppo_epoch=args.adv_ppo_epoch,
            num_mini_batch=args.adv_num_mini_batch,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            lr=args.lr,
            eps=args.clip_eps,
            max_grad_norm=args.max_grad_norm,
            clip_value_loss=args.clip_value_loss,
            log_grad_norm=args.log_grad_norm
        )

    # algo = PPO(
    #     actor_critic=actor_critic,
    #     lr=3e-4,
    #     eps=0.2,
    #     value_loss_coef=0.5,
    #     entropy_coef=0.01,
    #     max_grad_norm=0.5,
    #     clip_param=args.clip_param,
    #     ppo_epoch=ppo_epoch,
    #     num_mini_batch=num_mini_batch,
    # )
    return actor_critic, algo, storage