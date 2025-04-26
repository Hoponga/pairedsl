from task_spec import TaskSpec 
from classification_env import ClassificationEnv 


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

env = ClassificationEnv(tspec, batch_size=64, device="cuda")