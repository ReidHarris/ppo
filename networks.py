import torch
import torch.nn as nn
import torch.nn.functional as F

import abc

class Policy(nn.Module):
    def __init__(self, input_size, output_size, layer_size=64):
        super(Policy, self).__init__()
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, output_size)
        )

    @abc.abstractmethod
    def select_action(self, state):
        pass

class PolicyContinuous(Policy):
    def __init__(self, input_size, output_size, layer_size=64):
        super(PolicyContinuous, self).__init__(input_size, output_size, layer_size)
        self.std = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        x = self.model(x)
        return x

    def select_action(self, state, test=False):
        mean = self.forward(state)
        if test :
            return mean
        std = torch.exp(self.std)
        m = torch.distributions.Normal(mean, std)
        action = m.sample()
        return action, m.log_prob(action)

    def log_probs(self, states, actions):
        mean = self.forward(states)
        std = torch.exp(self.std)
        dist = torch.distributions.Normal(mean, std)
        log_probabilities = dist.log_prob(actions)
        entropy = dist.entropy()
        entropy = torch.sum(entropy, dim=-1)
        return log_probabilities, entropy

class PolicyDiscrete(Policy):
    def __init__(self, input_size, output_size, layer_size=64):
        super(PolicyDiscrete, self).__init__(input_size, output_size, layer_size)

    def forward(self, x):
        x = self.model(x)
        return x

    def select_action(self, state):
        probs = self.forward(state)
        probs = torch.nn.functional.softmax(probs)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)

    def log_probs(self, states, actions):
        probs = self.forward(states)
        probs = torch.nn.functional.softmax(probs)
        log_probabilities = torch.distributions.Categorical(probs).log_prob(actions).squeeze(0)
        return log_probabilities

class Critic(nn.Module):
    def __init__(self, input_size, critic_output_size, layer_size=64):
        super(Critic, self).__init__()
        self.critic_output_size = critic_output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, critic_output_size)
        )

    def forward(self, x):
        return self.model(x)