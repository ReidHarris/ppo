from collections import namedtuple
import torch
import numpy as np
from collections import deque

class Memory:
    def __init__(self, action_space_size, state_space_size, size):
        self.actions = torch.zeros(size, action_space_size).to(device)
        self.states = torch.zeros(size, state_space_size).to(device)
        self.log_probabilities = torch.zeros(size, action_space_size).to(device)
        self.rewards = torch.zeros(size).to(device)
        self.dones = torch.zeros(size).to(device)
        self.next_states = torch.zeros(size, state_space_size).to(device)
        self.advantages = torch.zeros(size+1).to(device)
        self.gt = torch.zeros(size+1).to(device)

        self.index = 0
        self.max_size = size


    def add_memory(self, action, state, reward, log_prob, next_state, done) :
        self.actions[self.index] = torch.Tensor(action)
        self.states[self.index] = torch.Tensor(state)
        self.rewards[self.index] = torch.Tensor([reward]).squeeze(-1)
        self.log_probabilities[self.index] = torch.Tensor(log_prob).detach()
        self.next_states[self.index] = torch.Tensor(next_state)
        self.dones[self.index] = torch.Tensor([int(done)]).squeeze(-1)

        #print(self.actions)
        #print(self.states)
        #print(self.values)
        #print(self.rewards)
        #print(self.log_probabilities)
        #print(self.next_states)
        #print(self.dones)
        self.index += 1


    def delete_memory(self) :
        self.index = 0

    def get_actions(self) :
        return self.actions

    def get_states(self) :
        return self.states

    def get_rewards(self) :
        return self.rewards

    def get_probabilities(self) :
        return self.log_probabilities

    def get_advantages(self):
        return self.advantages

    def get_gt(self):
        return self.gt

    def calculate_advantages(self, next_values, values, gamma=0.99, alpha=0.9):
        self.gt[-1] = next_values[-1]
        for i in reversed(range(self.max_size)):
            value = values[i]
            reward = self.rewards[i]
            next_value = next_values[i]
            done = self.dones[i]

            delta = reward + gamma * next_value * (1 - done) - value
            self.advantages[i] = delta + alpha * gamma * self.advantages[i+1] * (1 - done)
            self.gt[i] = reward + gamma * self.gt[i+1] * (1 - done)

    def __len__(self) :
        return self.index