import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LinearLR

import gymnasium as gym

from networks import *
from memory import Memory

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

class Config :
    def __init__(self, time_steps=2000000) :
        self.policy_clip = 0.2
        self.value_clip = 0.2
        self.gradient_clip = 0.5
        self.entropy_coefficient = 0.0001

        self.policy_clip_start = self.policy_clip
        self.value_clip_start = self.value_clip
        self.gradient_clip_start = self.gradient_clip
        self.entropy_coefficient_start = self.entropy_coefficient

        self.policy_clip_end = 0.05
        self.value_clip_end = 0.05
        self.gradient_clip_end = 0.5
        self.entropy_coefficient_end = 0.0

        self.total_steps = time_steps
        self.fraction = 1/self.total_steps

        self.gamma = 0.99
        self.alpha = 0.9

    def step(self):
        self.policy_clip -= (self.policy_clip_start - self.policy_clip_end) / self.total_steps
        self.value_clip -= (self.value_clip_start - self.value_clip_end) / self.total_steps
        self.gradient_clip -= (self.gradient_clip_start - self.gradient_clip_end) / self.total_steps
        self.entropy_coefficient -= (self.entropy_coefficient_start - self.entropy_coefficient_end) / self.total_steps

class PPO():
    def __init__(self,
                 env,
                 time_steps,
                 memory_length,
                 policy_learning_rate=8e-5,
                 critic_learning_rate=8e-5,
                 device=device
                 ):
        self.env = env

        self.critic_input_size = env.observation_space.shape[0]
        self.critic_output_size = 1
        self.memory_length = memory_length

        self.policy_input_size = env.observation_space.shape[0]
        self.policy_output_size = env.action_space.shape[0]
        self.policy = PolicyContinuous(self.policy_input_size, self.policy_output_size, layer_size=64).to(device)
        self.memory = Memory(self.policy_output_size, self.critic_input_size, memory_length)

        self.critic = Critic(self.critic_input_size, 1, layer_size=64).to(device)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-5)

        self.lrs_a = LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.00001, total_iters=time_steps)
        self.lrs_c = LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.00001, total_iters=time_steps)

        self.timestep_rewards = []
        self.episode_rewards = []


        self.config = self.Config(time_steps)
        self.t_step = 0
        self.episode_count = 1

    def get_trajectory(self, state):
        while (len(self.memory) < self.memory_length):
            action, prob = self.policy.select_action(state)
            old_state = state
            state, reward, done, truncated, info = self.env.step(action.detach().cpu().numpy())
            state = torch.from_numpy(state).float().to(device)
            self.memory.add_memory(action, old_state, reward, prob, state, done)
            self.t_step += 1

            if done:
                state, _ = self.env.reset()
                state = torch.from_numpy(state).float().to(device)
                print("Episode {} Reward : {:.4f}".format(self.episode_count, info['episode']['r']))
                self.episode_rewards.append(info['episode'])
                self.episode_count += 1
        return state

    def update_critic(self, states, gt, old_values):
        self.critic_optimizer.zero_grad()
        values = self.critic(states).squeeze(-1)

        value_loss_surrogate_1 = torch.square(gt - values)
        clamped_values = torch.clamp(values, old_values-self.config.value_clip, old_values+self.config.value_clip)
        value_loss_surrogate_2 = torch.square(gt - clamped_values)
        value_clipped_fraction = ((values - old_values).abs() > self.config.value_clip).float().sum()
        value_loss = torch.maximum(value_loss_surrogate_1, value_loss_surrogate_2)
        value_loss = torch.mean(value_loss)

        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()
        return value_loss.item(), value_clipped_fraction

    def update_policy(self, states, actions, old_log_probs, advantages):
        self.policy_optimizer.zero_grad()
        log_probs, entropy = self.policy.log_probs(states, actions)
        log_probs = torch.sum(log_probs, dim=-1)
        ratios = torch.exp(log_probs - old_log_probs.detach())
        advantage_normed = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss_surrogate_1 = ratios * advantage_normed
        policy_loss_surrogate_2 = torch.clamp(ratios, 1-self.config.policy_clip, 1+self.config.policy_clip) * advantage_normed
        policy_clipped_fraction = ((torch.abs(ratios-1) > self.config.policy_clip).float().sum())

        policy_loss = -torch.minimum(policy_loss_surrogate_1, policy_loss_surrogate_2)
        policy_loss = torch.mean(policy_loss) - self.config.entropy_coefficient * torch.mean(entropy)
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
        self.policy_optimizer.step()
        return policy_loss.item(), policy_clipped_fraction

    def train(self, n_training_iterations, batch_size):
        state, _ = self.env.reset()
        state = torch.from_numpy(state).float().to(device)
        while (self.t_step < self.config.total_steps) :
            state = self.get_trajectory(state)

            actions = self.memory.get_actions()
            states = self.memory.get_states()
            old_values = self.critic(states).squeeze(-1).detach()
            old_log_probs = torch.sum(self.memory.get_probabilities(), dim=-1)

            next_states = self.memory.next_states
            next_values = self.critic(next_states).squeeze(-1).detach()
            self.memory.calculate_advantages(next_values, old_values, self.config.gamma, alpha=self.config.alpha)
            advantages = self.memory.get_advantages()
            gt = self.memory.get_gt()

            value_losses = []
            policy_losses = []

            total_iterations = n_training_iterations * self.memory_length
            for _ in range(n_training_iterations) :
                indices = np.arange(self.memory_length)
                np.random.shuffle(indices)

                for k in range(0, self.memory_length, batch_size):
                    ind = indices[k: k+batch_size]

                    value_loss, value_clipped_fraction = self.update_critic(states[ind], gt[ind], old_values[ind])
                    value_losses.append(value_loss)

                    policy_loss, policy_clipped_fraction = self.update_policy(states[ind], actions[ind], old_log_probs[ind], advantages[ind])
                    policy_losses.append(policy_loss)

            value_loss_mean = sum(value_losses)/len(value_losses)
            policy_loss_mean = sum(policy_losses)/len(policy_losses)
            print("Time Step {:<8} \tMean Value Loss : {:.4f}\tMean Policy Loss : {:.4f}".format(self.t_step, value_loss_mean, policy_loss_mean))
            print("\t\t\t% Values Clipped : {:.4f}\t% Policy Clipped : {:.4f}".format(value_clipped_fraction/total_iterations, policy_clipped_fraction/total_iterations))

            value_losses.clear()
            policy_losses.clear()

            self.memory.delete_memory()
            self.lrs_a.step()
            self.lrs_c.step()
            self.config.step()

    def test(self) :
        episode_reward = 0
        time_steps = 0
        done = False
        state, _ = self.env.reset()
        state = torch.from_numpy(state).float().to(device)
        while(not done) :
            action = self.policy.select_action(state, True)
            state, reward, done, _, _ = self.env.step(action.detach().cpu().numpy())
            state = torch.from_numpy(state).float().to(device)

            episode_reward += reward
            time_steps += 1
        print("Final Episode Reward : {:.6f}".format(episode_reward))

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(
            checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])
        print(f"Loaded agent from {path}.")