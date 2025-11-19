"""
Script for training the agent for snake using various methods
"""
import os
import sys
import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game2
from game_environment import SnakeNumpy
from agent import BreadthFirstSearchAgent, SupervisedLearningAgent
import json
# Global variables
version = 'v17.1'
# Load training configurations
try:
    with open(f'model_config/{version}.json', 'r') as f:
        m = json.loads(f.read())
        board_size = m['board_size']
        frames = m['frames']  # Keep frames >= 2
        max_time_limit = m['max_time_limit']
        supervised = bool(m['supervised'])
        n_actions = m['n_actions']
        obstacles = bool(m['obstacles'])
except FileNotFoundError:
    print(f"Configuration file 'model_config/{version}.json' not found.")
    sys.exit(1)
# Overwrite max_time_limit for testing purposes
max_time_limit = 28  # Default: 998
generate_training_data = False
do_training = True
n_games_training = 100
# Setup the environment
env = SnakeNumpy(board_size=board_size, frames=frames, games=n_games_training,
                 max_time_limit=max_time_limit, obstacles=obstacles, version=version)
n_actions = env.get_num_actions()
# Directory for saving models and buffers
model_dir = f'models/{version}'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# Generate training data
if generate_training_data:
    print("Generating training data...")
    agent = BreadthFirstSearchAgent(board_size=board_size, frames=frames,
                                    n_actions=n_actions, buffer_size=60000,
                                    version=version)
    for index in tqdm(range(1)):
        curr_time = time.time()
        _, _, _ = play_game2(env, agent, n_actions, epsilon=-1,
                             n_games=n_games_training, record=True,
                             reward_type='current', frame_mode=True,
                             total_frames=60000, stateful=True)
        print(f"Buffer size {agent.get_buffer_size()} filled in {time.time() - curr_time:.2f}s")
        agent.save_buffer(file_path=model_dir, iteration=(index + 1))
        agent.reset_buffer()
# Train the agent
if do_training:
    print("Starting training...")
    agent = SupervisedLearningAgent(board_size=board_size, frames=frames,
                                    n_actions=n_actions, buffer_size=1,
                                    version=version)
    total_files = 1  # Number of buffer files to load
    for index in tqdm(range(1 * total_files)):
        # Load the saved training data
        try:
            agent.load_buffer(file_path=model_dir, iteration=((index % total_files) + 1))
            print(f"Loaded buffer {((index % total_files) + 1)}. Current buffer size: {agent.get_buffer_size()}")
        except FileNotFoundError:
            print(f"Buffer file not found: {model_dir}/buffer_{((index % total_files) + 1):04d}.pkl. Skipping...")
            continue
        # Train the agent
        loss = agent.train_agent(epochs=20)
        print(f"Loss at buffer {((index % total_files) + 1)} is: {loss:.5f}")
    # Normalize output layer weights to prevent explosion in outputs
    print("Normalizing output layer weights...")
    agent.normalize_layers(agent.get_max_output())
    agent.update_target_net()
    # Save the trained model
    print("Saving trained model...")
    agent.save_model(file_path=model_dir)
    print(f"Model saved to {model_dir}")

