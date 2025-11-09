import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    Stable Actor-Critic for continuous actions.
    - forward(x) -> mu, sigma, value
    - evaluate(state, action) -> log_prob, value, entropy
    - act(state, deterministic=False) -> action, log_prob
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, init_log_std=-3.0):
        super().__init__()
        # Feature extractor
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # actor
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_param = nn.Parameter(torch.ones(action_dim) * init_log_std)

        # critic
        self.value_head = nn.Linear(hidden_dim, 1)

        # weight init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: tensor shape (..., state_dim)
        returns: mu, sigma, value
        """
        x = self.net(x)
        mu = torch.tanh(self.mu_head(x))       # in (-1,1)
        log_std = self.log_std_param.expand_as(mu)
        sigma = torch.exp(log_std).clamp(min=1e-6, max=1.0)
        value = self.value_head(x)
        return mu, sigma, value

    def act(self, state, deterministic=False):
        """
        state: tensor (batch or single)
        returns action (same shape as mu), log_prob (scalar per sample)
        """
        mu, sigma, _ = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()   # reparameterized for low variance
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        """
        For training: returns log_prob, entropy, value
        """
        mu, sigma, value = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze(-1)
