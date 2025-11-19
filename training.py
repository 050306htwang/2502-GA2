#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for training the agent for snake using various methods (PyTorch version)
"""
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import json
import torch

from utils import play_game2
from game_environment import SnakeNumpy
from agent import DeepQLearningAgent  # 已换成 PyTorch 实现

# --------------- 基础配置 ---------------
torch.manual_seed(42)
version = 'v17.1'

with open(f'model_config/{version}.json', 'r') as f:
    m = json.load(f)

board_size      = m['board_size']
frames          = m['frames']          # >=2
max_time_limit  = m['max_time_limit']
supervised      = bool(m['supervised'])
n_actions       = m['n_actions']
obstacles       = bool(m['obstacles'])
buffer_size     = m['buffer_size']

episodes        = 2 * 10**5
log_frequency   = 500
games_eval      = 8

# --------------- 构造 Agent ---------------
agent = DeepQLearningAgent(
    board_size=board_size,
    frames=frames,
    n_actions=n_actions,
    buffer_size=buffer_size,
    version=version
)
agent_type = 'DeepQLearningAgent'
print(f'Agent is {agent_type}')

# --------------- 训练超参 ---------------
epsilon, epsilon_end = 1.0, 0.01
decay                = 0.97
reward_type          = 'current'
sample_actions       = False
n_games_training     = 8 * 16          # 并行游戏数

if supervised:
    epsilon = 0.01
    agent.load_model(file_path=f'models/{version}')
    try:
        agent.load_buffer(file_path=f'models/{version}', iteration=1)
    except FileNotFoundError:
        pass
else:
    # 预填充回放缓冲区
    games = 512
    env = SnakeNumpy(
        board_size=board_size,
        frames=frames,
        max_time_limit=max_time_limit,
        games=games,
        frame_mode=True,
        obstacles=obstacles,
        version=version
    )
    t0 = time.time()
    _ = play_game2(
        env, agent, n_actions,
        n_games=games, record=True,
        epsilon=epsilon, verbose=True,
        reset_seed=False, frame_mode=True,
        total_frames=games * 64
    )
    print(f'Playing {games * 64} frames took {time.time() - t0:.2f}s')

# --------------- 训练 & 评估环境 ---------------
env = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=n_games_training,
    frame_mode=True,
    obstacles=obstacles,
    version=version
)
env2 = SnakeNumpy(
    board_size=board_size,
    frames=frames,
    max_time_limit=max_time_limit,
    games=games_eval,
    frame_mode=True,
    obstacles=obstacles,
    version=version
)

# --------------- 主训练循环 ---------------
model_logs = {'iteration': [], 'reward_mean': [],
              'length_mean': [], 'games': [], 'loss': []}

for idx in tqdm(range(1, episodes + 1), desc='Training'):
    # ---- 采集数据 ----
    play_game2(
        env, agent, n_actions,
        epsilon=epsilon,
        n_games=n_games_training,
        record=True,
        sample_actions=sample_actions,
        reward_type=reward_type,
        frame_mode=True,
        total_frames=n_games_training,
        stateful=True
    )
    # ---- 训练一步 ----
    loss = agent.train_agent(batch_size=64, reward_clip=True)

    # ---- 评估 & 日志 ----
    if idx % log_frequency == 0:
        current_rewards, current_lengths, current_games = play_game2(
            env2, agent, n_actions,
            n_games=games_eval,
            epsilon=-1,
            record=False,
            sample_actions=False,
            frame_mode=True,
            total_frames=-1,
            total_games=games_eval
        )
        model_logs['iteration'].append(idx)
        model_logs['reward_mean'].append(float(current_rewards) / current_games)
        model_logs['length_mean'].append(float(current_lengths) / current_games)
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)

        pd.DataFrame(model_logs)[['iteration', 'reward_mean',
                                  'length_mean', 'games', 'loss']] \
          .to_csv(f'model_logs/{version}.csv', index=False)

        # ---- 更新 target 网络 & 存盘 ----
        agent.update_target_net()
        agent.save_model(file_path=f'models/{version}', iteration=idx)
        epsilon = max(epsilon * decay, epsilon_end)
