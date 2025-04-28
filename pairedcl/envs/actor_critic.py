# pairedcl/envs/simple_actor_critic.py
import torch, torch.nn as nn
from torch.distributions import Normal

class TGActorCritic(nn.Module):
    """Actor-critic for the TaskGenerator in Permuted-MNIST."""
    def __init__(self, obs_dim=1, hidden=64, action_dim=1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh())
        self.mu_head   = nn.Linear(hidden, action_dim)
        self.logstd    = nn.Parameter(torch.zeros(action_dim))
        self.value_net = nn.Linear(hidden, 1)

    # policy only ---------------------------------------------------------
    def _dist(self, obs):
        h  = self.shared(obs)
        mu = torch.tanh(self.mu_head(h))          # keep in (-1,1)
        std = self.logstd.exp().expand_as(mu)
        return Normal(mu, std)

    def act(self, obs):
        dist   = self._dist(obs)
        action = dist.sample()
        logp   = dist.log_prob(action).sum(-1, keepdim=True)
        value  = self.value_net(self.shared(obs))
        return action, logp, value

    # actor–critic interface used by runner ------------------------------
    def value_logp(self, obs, action):
        dist  = self._dist(obs)
        value = self.value_net(self.shared(obs))
        logp  = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return value, dist, logp, entropy