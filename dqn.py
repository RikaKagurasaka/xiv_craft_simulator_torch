import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from craft_simulator import SKI_length, ActionsCount, Simulator, CraftResultEnum


class MLP(torch.nn.Module):
    def __init__(self,
                 input_dims, hidden_dims, output_dims, layer_count,
                 activation=nn.LeakyReLU,
                 output_activation=nn.Identity
                 ):
        super().__init__()
        self.activation = activation()
        self.output_activation = output_activation()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dims, hidden_dims))
        for i in range(layer_count - 1):
            self.layers.append(nn.Linear(hidden_dims, hidden_dims))
        self.layers.append(nn.Linear(hidden_dims, output_dims))

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_activation(x)
        return x


class DQN(torch.nn.Module):

    def __init__(self,
                 input_shape=SKI_length,
                 num_actions=ActionsCount,
                 gamma=0.99,
                 hidden_dims=256,
                 layers_cnt=5
                 ):
        super().__init__()

        self.mlp = MLP(input_shape, hidden_dims, num_actions, layers_cnt)
        self.gamma = gamma
        self.buffer = torch.zeros((0, SKI_length + 1 + 1 + SKI_length + 1), device=device)
        self.loss_fn = nn.MSELoss()
        self.eps = 1.0
        self.eps_min = 0.1
        self.eps_decay = 0.9999
        self.batch_size = 50000

    def forward(self, x):
        return self.mlp(x)

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer = torch.cat([self.buffer, torch.cat([state, action.unsqueeze(-1), reward.unsqueeze(-1), next_state, done.unsqueeze(-1)], dim=1)], dim=0)
        if len(self.buffer) > 1000000:
            self.buffer = self.buffer[-(1000000 + 1):]

    def sample_from_buffer(self):
        batch_idx = random.sample(range(len(self.buffer)), self.batch_size)
        return self.buffer[batch_idx]

    def calc_loss(self, batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch[:, :SKI_length], batch[:, SKI_length], batch[:, SKI_length + 1], batch[:, -SKI_length - 1:-1], batch[:, -1]

        q_vals = self(state_batch)
        q_vals = q_vals.gather(1, action_batch.unsqueeze(-1).to(torch.int64)).squeeze(-1)

        next_q_vals = self(next_state_batch)
        next_q_vals = next_q_vals.max(1)[0]
        next_q_vals[done_batch > 0] = 0.0

        target_q_vals = reward_batch + self.gamma * next_q_vals

        loss = self.loss_fn(q_vals, target_q_vals.detach())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.002)
        return optimizer

    def sample_action(self, state_matrix):
        q_vals = self(state_matrix)
        action = torch.argmax(q_vals, dim=1)
        random_mask = torch.rand(*action.shape, device=action.device) < self.eps
        action[random_mask] = torch.randint(0, ActionsCount, action[random_mask].shape, device=random_mask.device)
        return action

    def update_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DQN()
    model.to(device)
    optim = model.configure_optimizers()

    for epoch in range(1000000):
        episode_reward = 0
        env = Simulator.reset(torch.tensor([6600, 14040, 70, 702, 265, 262], dtype=torch.float32, device=device), count=10000)
        while env.matrix.size(0):
            state = env.matrix[:, :]
            action = model.sample_action(env.matrix)
            score = env.score()
            env.run_action(action)
            next_state = env.matrix[:, :]
            next_score = env.score()
            reward = next_score - score
            done = env.craft_result > CraftResultEnum.PENDING
            episode_reward += reward.mean()
            model.add_to_buffer(state, action, reward, next_state, done)
            env.drop_finished()
            if len(model.buffer) > model.batch_size:
                batch = model.sample_from_buffer()
                model.update_eps()
                optim.zero_grad()
                loss = model.calc_loss(batch)
                loss.backward()
        print(f'Epoch {epoch + 1}, Reward: {episode_reward}')
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'model.pth')
