# pairedcl/algos/make_agent.py
from pairedcl.algos.storage import RolloutCLStorage
from pairedcl.algos.ppo     import PPO
from pairedcl.envs.simple_actor_critic import TGActorCritic

def make_agent(args, device="cpu"):
    actor_critic = TGActorCritic().to(device)

    storage = RolloutCLStorage(
        num_steps=args.rollout,
        num_envs=1,
        obs_shape=(1,),
        action_space=actor_critic.mu_head.weight.new_zeros(1).shape,  # (1,)
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )

    algo = PPO(
        actor_critic=actor_critic,
        lr=3e-4,
        eps=0.2,
        epochs=4,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    return actor_critic, algo, storage