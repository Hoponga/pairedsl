import torch.nn as nn 

class TaskGenerator(nn.Module):
    """
    Environment-generator (teacher) policy π_E.
    Observation  : one-hot of which classes picked, budget left, etc.
    Action space : MultiDiscrete => choose next class / augmentation / n_shots
    Episode ends : after K decisions → TaskSpec is finalised.
    """
    def reset(self, batch_size: int) -> torch.Tensor: ...
    def act(self, obs, hidden, mask) -> Tuple[val, action, log_prob, hidden]: ...
    def step_adversary(self, action):
        """
        identical signature to gym:
        returns (obs, 0., done, {})   # reward always 0
        """