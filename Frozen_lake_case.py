## Frozen Lake case - control problem using RL applications
# Here we apply the cross-entropy RL method, which is a baseline method and robust for simple control problems
# We use the gym lib to model the environment
# It is a grid world env and we design an agent to find the best route between A and B
# Grid with 4x4 movement, so the agent can move: up, down, left, right
# Point A is on the top-left of the grid, and the point B (goal) is on the bottom-right corner
# There are holes in the grid and if the episode ends if the agent goes to these holes with reward = 0 (failure)
# Only getting to point B without holes returns reward = 1 (success)
# Since is a 'Frozen Lake' everytime the agents moves it gets slippery, conditions:
#  - 33 % chance to slip right or left
#  - Moving left then 33 % moving left, 33 % chance moving up, 33 % chance moving down
#  - This creates uncertainty of movements, where the agent should exactly to move

#%% Import packages
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from RL_agent_cross_entropy import *

#%% Hyperparameters of the example
hidden_layer_size = 128
batch_size = 16
reward_top_per = 70 # q-th percentile of 'elite' episodes = 1 - reward_top

#%% Main script
if __name__ == '__main__':
    # Basic example of Frozen Lake environment
    # env = gym.make('FrozenLake-v0')
    # print(f'Observation space: {env.observation_space}, meaning = 4x4 grid')
    # print(f'Action space: {env.action_space}, meaning = up, down, left, right')
    # env.reset()
    # env.render()

    # Call the Frozen Lake env with the One-hot encoding class
    env = One_hot_encoding_transform(gym.make('FrozenLake-v0'))
    obs_size = env.observation_space.shape[0] # State size
    action_size = env.action_space.n # Action size

    # Define the cross-entropy agent
    cre_agent = net(obs_size, action_size, hidden_layer_size)  # Class the net
    train_obj = nn.CrossEntropyLoss()  # Used for multi-class classification-type problems, to deal with probability distribution
    # nn.CrossEntropyLoss: receives values before softmax fun, so before the probability distribution
    train_opt = optim.Adam(params=cre_agent.parameters(), lr=0.01)  # cre_agent has the parameters since is also defined by nn.Module
    # lr parameter is the learning rate

    writer = SummaryWriter(comment='-cartpole')  # TensorBoard result

    for iter, episode in enumerate(iterate_batch(env, cre_agent, batch_size)):  # Enumerate gives the iter and item.value (batch)
        obs_tr, act_tr, reward_b, reward_m = elite_batch(episode, reward_top_per)
        train_opt.zero_grad()  # Level to zero our NN gradients
        act_scores_tr = cre_agent(obs_tr)  # NN prediction to the elite episode
        loss = train_obj(act_scores_tr, act_tr)  # Newly loss to reinforce the NN for the elite episode
        loss.backward()  # Calculate the gradients of the weights
        train_opt.step()  # Update the NN weights with the optimizer
        # Print the progress per training iteration
        print(f'{iter}: loss={loss.item():.3f}, reward={reward_m:.1f}, reward_bound={reward_b:.1f}')
        # Save the progress in the tensorBoard
        writer.add_scalar('loss', loss.item(), iter)
        writer.add_scalar('reward_mean', reward_m, iter)
        writer.add_scalar('reward_bound', reward_m, iter)
        # Condition for stopping the NN training
        if reward_m > 199:
            print('Successful training of the RL agent with Cross-entropy method')
            break
        writer.close()