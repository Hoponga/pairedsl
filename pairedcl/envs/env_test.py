from pairedcl.envs.task_spec import TaskSpec, TransformSpec
from pairedcl.envs.classification_env import ClassificationEnv 
from pairedcl.agents.classifier_agent import ClassifierAgent
import torch 
import matplotlib.pyplot as plt 
import time
# create a random permutation 
perm = torch.randperm(28*28)

#Define a task specification as the permutation transformation on the base data 
tspec = TaskSpec(
        dataset_id="mnist-train",
        transforms=[TransformSpec("permute", {"p": perm})])


# who knows wtf this does 
tspec2 = TaskSpec(
        dataset_id="core50-s1",
        transforms=[
            TransformSpec("gauss_noise", {"sigma": 0.05})],
        class_subset=None)

env = ClassificationEnv(tspec2, batch_size=64, device="cuda")

agent = ClassifierAgent(model=ResNet18(), optimizer='adam', args=args, num_envs=args.num_envs)
env.reset()
for _ in range(15): 
    # step the environment 
    obs, reward, done, info = env.step(torch.zeros(64).to("cuda"))
    print(f"reward: {reward}")
    img = obs[0].squeeze().cpu().numpy()  

    # plot
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.title(f"Batch 0, reward={reward:.2f}")
    plt.savefig(f"test{time.time()}.png", dpi=300)