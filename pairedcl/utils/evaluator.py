import torch
from typing import List


from pairedcl.envs.task_spec import TaskSpec, TransformSpec 



class Evaluator:
    """
    Benchmark an agent on a fixed panel of T permuted-MNIST tasks.

    Parameters
    ----------
    task_gen : TaskGenerator
        The adversary / task generator that can map actions → TaskSpec.
    class_env : ClassificationEnv
        The environment used for data loading and label retrieval.
    T : int, default 20
        Number of distinct tasks kept for the lifetime of the evaluator.
    batches_per_task : int, default 5
        Minibatches evaluated for each TaskSpec when calling `evaluate`.
    device : str | torch.device
    """

    def __init__(self,
                 task_type,
                 class_env,
                 T: int = 10,
                 batches_per_task: int = 5,
                 device="cpu"):
        self.device = torch.device(device)
        self.class_env = class_env
        self.batches_per_task = batches_per_task

        # ---- create and store T TaskSpecs ----------------------------------
        self.T = T 
        self.task_specs: List[TaskSpec] = []
        assert task_type == "permute" 

        for _ in range(T):
            self.task_specs.append(self._create_permutation_task())


            
    def _create_permutation_task(self): 
        random_perm = torch.randperm(28*28) 
        return TaskSpec("mnist-train", [TransformSpec("permute", {"p": random_perm})])
    # -----------------------------------------------------------------------

    # "REDUCE THE PROBLEM" @JIANTAO JIAO 
    @torch.no_grad()
    def evaluate(self, agent) -> float:
    
        total_correct, total_seen = 0, 0

        for spec in self.task_specs:
            # reset env to the task
            _, _ = self.class_env.reset(task_spec=spec)
            loader_iter = iter(self.class_env.loader)

            for _ in range(self.batches_per_task):
                try:
                    imgs, labels = next(loader_iter)
                except StopIteration:               # handle small loaders
                    loader_iter = iter(self.class_env.loader)
                    imgs, labels = next(loader_iter)

                imgs, labels = imgs.to(self.device), labels.to(self.device)
                _, preds, _, _ = agent.act(imgs)
                total_correct += (preds == labels).sum().item()
                total_seen   += labels.numel()

        return total_correct / total_seen   # mean accuracy