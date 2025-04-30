# pairedcl/algos/make_agent.py
from pairedcl.algos.storage import RolloutStorage
from pairedcl.algos.ppo     import PPO

import torch 
from pairedcl.envs.task_generator import TaskGenerator, SubsetTaskGenerator
from pairedcl.envs.task_spec import TaskSpec, TransformSpec
from pairedcl.utils.permute import noise_sorted_perm
import numpy as np 


def make_task_generator(device, args, seed = 0, tasks = None):
    # # our action is the noise factor for our noise_sorted_perm 
    # param_defs = [
    #     dict(tname="permute", pkey="sigma", low=0.0, high=1.0)
    # ]
    # assert args.action_dim == len(param_defs) + 1
    # tg = TaskGenerator(param_defs,
    #                    base_dataset="mnist-train",
    #                    obs_dim=args.context_obs_shape,
    #                    hidden_dim=64,
    #                    device=device, 
    #                    max_gen_tasks=args.max_gen_tasks)


    # tg._permute_spec = lambda p : noise_sorted_perm(28*28, p, np.random.RandomState(seed))
    # Override action_to_taskspec for permuted MNIST
    # def _a2spec(self, action):
    #     seed_val = ((action[0].item() + 1)/2) * (2**20 -1)
    #     perm = self._permute_spec(seed_val)
    #     ts = TaskSpec("mnist-train", [TransformSpec("permute", {"p": perm})])
    #     return ts
    # tg.action_to_taskspec = _a2spec.__get__(tg, TaskGenerator)

    tg = SubsetTaskGenerator(args.context_obs_shape, tasks, device=device)
    return tg

def make_agent(args, device="cpu", tasks = None):
    actor_critic = make_task_generator(device, args, tasks = tasks)
    


    
    #print(actor_critic.mu_head)
    storage = RolloutStorage(
        num_steps=args.num_steps,
        num_envs=1,
        obs_shape=(args.context_obs_shape,), # what info does the adversary environment take in? 
        action_space=actor_critic.logit_head.weight.new_zeros(actor_critic.logit_head.out_features ),  # (1,)
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