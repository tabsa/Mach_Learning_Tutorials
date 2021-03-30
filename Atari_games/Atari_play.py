## Atari game played by the trained DQN model
# Here, the newest trained DQN plays the Pong game
# The dat.file PongNoFramskip-v4-best contains the latest DQN model

#%% Import packages
import gym
import time
import numpy as np
import collections
from pathlib import Path
# Pytorch lib
import torch
import torch.nn as nn
import torch.optim as optim

# Import other classes
import dqn_model
import wrappers

#%% Hyperparameters of the example
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FRAME_PER_SEC = 25 # Frame/sec that is shown
VISUALIZE_FRAME = True # Frames to visualize in the render
RECORD_VIDEO = True # Option to record the render as video (it generates but the video does NOT work)

#%% Main script
if __name__ == "__main__":
    wok_dir = Path.cwd()
    dqn_trained = 'PongNoFrameskip-v4-best.dat'
    # Call the env
    env_render = True  # GUI visualization (True) or No visual (False)
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    if RECORD_VIDEO:
        env = gym.wrappers.Monitor(env, wok_dir)
    # Create the DQN model
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n) # We will run in the CPU, since is less demanding than training
    file_path = wok_dir / dqn_trained
    state = torch.load(file_path) # Load the trained dqn...In case of training with GPU, please include: map_location=lambda stg,_: stg
    net.load_state_dict(state)
    print(net) # Print out the NN structure

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True: # Loop for the playing episode
        start_ts = time.time() # Starting time of the game
        if env_render:
            env.render()
        # Almost the same code as for the class Agent - def play_step without the exploration phase
        state_a = np.array([state], copy=False)  # Convert state_t into np.array
        state_v = torch.tensor(state_a)  # Convert into tensor
        q_vals_v = net(state_v)  # Get Q-value (state,[action_1,....action_N])
        _, act_v = torch.max(q_vals_v, dim=1)  # Get action: max(Q_value)
        action = int(act_v.item())  # Action value NOT index of action
        c[action] += 1
        # Test in the environment
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
        if is_done: # Stopping condition
            break
        if VISUALIZE_FRAME:
            delta = 1/FRAME_PER_SEC - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    # Final message
    print(f'Total reward: {total_reward:.2f}')
    print(f'Action counts: {c}')
    if env_render: # Close the GUI for visualizing the game
        env.close()
    if RECORD_VIDEO:
        env.env.close()