from replay_buffer import ReplayBufferNumpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from collections import deque
import json
import os

# ---------- 工具 ----------
def huber_loss(y_pred, y_true, delta=1.0):
    """PyTorch Huber loss"""
    error = y_pred - y_true
    is_small = error.abs() < delta
    squared = 0.5 * error.pow(2)
    linear = delta * (error.abs() - 0.5 * delta)
    return torch.where(is_small, squared, linear).mean()

# ---------- 基类 ----------
class Agent:
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version=''):
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (frames, board_size, board_size)  # (C,H,W)
        self._version = version
        self.reset_buffer()
        self._board_grid = np.arange(board_size ** 2).reshape(board_size, -1)

    def reset_buffer(self, buffer_size=None):
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size,
                                         self._n_frames, self._n_actions)

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        iteration = iteration or 0
        with open(f"{file_path}/buffer_{iteration:04d}.pkl", 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        iteration = iteration or 0
        with open(f"{file_path}/buffer_{iteration:04d}.pkl", 'rb') as f:
            self._buffer = pickle.load(f)

# ---------- DQN ----------
class DeepQLearningAgent(Agent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version=''):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_models()

    def reset_models(self):
        self._model = self._agent_model().to(self.device)
        if self._use_target_net:
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net()
        # 关键修复：统一命名 _optimizer
        self._optimizer = optim.RMSprop(self._model.parameters(), lr=5e-4)

    def _agent_model(self):
        class DQN(nn.Module):
            def __init__(self, in_ch, n_actions, h, w):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 6), nn.ReLU()
                )
                # 计算 conv 输出尺寸（假设 board_size=10）
                with torch.no_grad():
                    dummy = torch.zeros(1, in_ch, h, w)
                    out_ch = self.conv(dummy).shape[1:]
                flat_dim = np.prod(out_ch)
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_dim, 64), nn.ReLU(),
                    nn.Linear(64, n_actions)
                )

            def forward(self, x):
                return self.fc(self.conv(x))
        return DQN(self._n_frames, self._n_actions,
                   self._board_size, self._board_size)

    def _prepare_input(self, board):
        if board.ndim == 3:                          # 单帧
            board = np.expand_dims(board, 0)
        board = board.astype(np.float32) / 4.0
        # (B,H,W,C) -> (B,C,H,W)
        board = torch.from_numpy(board).permute(0, 3, 1, 2)
        return board.to(self.device)

    def move(self, board, legal_moves, value=None):
        board = self._prepare_input(board)
        with torch.no_grad():
            q = self._model(board).cpu().numpy()
        return np.argmax(np.where(legal_moves == 1, q, -np.inf), axis=1)

    def train_agent(self, batch_size=32, reward_clip=False):
        if self.get_buffer_size() < batch_size:
            return 0.0
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:
            r = np.sign(r)

        s = self._prepare_input(s)
        next_s = self._prepare_input(next_s)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # target Q
        with torch.no_grad():
            next_q = self._target_net(next_s) if self._use_target_net else self._model(next_s)
            next_q = torch.where(torch.FloatTensor(legal_moves).to(self.device) == 1,
                                 next_q, -1e8)
            max_next_q = next_q.max(1)[0]
            target = r + self._gamma * max_next_q * (1 - done)

        # 当前 Q
        current_q = self._model(s).gather(1, a.argmax(1).unsqueeze(1)).squeeze(1)

        loss = huber_loss(current_q, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def save_model(self, file_path='', iteration=None):
        iteration = iteration or 0
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pth")

    def load_model(self, file_path='', iteration=None):
        iteration = iteration or 0
        self._model.load_state_dict(
            torch.load(f"{file_path}/model_{iteration:04d}.pth", map_location=self.device))
        if self._use_target_net:
            self._target_net.load_state_dict(
                torch.load(f"{file_path}/model_{iteration:04d}_target.pth", map_location=self.device))
