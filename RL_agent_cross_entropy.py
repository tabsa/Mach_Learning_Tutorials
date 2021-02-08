## RL agent with cross-entropy method - Baseline method to others (DQN, actor-critic)
# Applied for any type of control problem

#%% Import packages
import gym
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

#%% Hyperparameters
# Tuples to keep information over episodes
Episode = namedtuple('Episode', field_names=['reward', 'steps']) # Stores the total_reward and episode_steps
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) # Stores the state (env_obs) and action per step

#%% Class section
class net(nn.Module): # Class for the network using nn.Module to get nn object
    def __init__(self, state_size, action_size, hidden_layer):
        super(net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, action_size)
        )
        # Softmax func is not include here to increase the numerical stability of the training process
        # Require to use the nn.Softmax externally

    def forward(self, x):
        return  self.net(x) # Output is a number for every action (action_size)
        # Predict the probability distribution per action
        # Remember that Softmax is to calculate the probability of logit numbers
        # NN predicts numbers per action and softmax converts to probability following:
        # softmax = exp(x)/sum(exp(x))

class One_hot_encoding_transform(gym.ObservationWrapper): # One-hot encoding class
    # Implementation of One-hot encoding of Discrete inputs to float array with 16 numbers
    # One-hot encoding converts categorical data into numerical one, like the Frozen Lake case with 16 Discrete positions and 4 Discrete actions
    # gym.ObservationWrapper: Class of gym that allows to modify the observations returned by the environment
    def __init__(self, env):
        super(One_hot_encoding_transform, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete) # Make sure to only be used in Discrete envs
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)
        # Defines the previous 16 Discrete (env.observation_space.n) into a box of 16 numbers with range [0, 1]

    def observation(self, observation): # Function to modify the observations
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0 # Changing Discrete to a Float-array
        return res

#%% Function section
def iterate_batch(env, net, size): # Generated of batch per episode
    batch = [] # batch is episode for this example
    epi_reward = 0.0
    epi_steps = []
    obs = env.reset()
    sm_prob = nn.Softmax(dim=1) # Logistic probability, to convert the net(x) into probability
    while True:
        obs_tr = torch.FloatTensor([obs]) # Transform into a PyTorch.tensor (tr)
        action_prob_tr = sm_prob(net(obs_tr)) # NN predict the action distribution. Remember NN is predicting the probability distribution per action
        action_prob = action_prob_tr.data.numpy()[0] # Converting into a np.array because PyTorch.tensor are objects Remember!
        # We use numpy()[0] because is tensor with 2 dimensions (1 x X) so we want the one in axis 0. So converting 2D to 1D array
        action = np.random.choice(len(action_prob), p=action_prob) # Select action based on the NN probability distribution
        # Receive the env response to the action
        next_obs, reward, is_done, _ = env.step(action)
        epi_reward += reward # Update total reward
        epi_steps.append(EpisodeStep(observation=obs, action=action)) # Store the state and action at step t
        # We do NOT store the next_obs because we want to know the correlation between obs and action
        if is_done: # Episode got to the end
            batch.append(Episode(reward=epi_reward, steps=epi_steps)) # Store Episode info - Total reward and Steps (state, action)
            # Reset parameters for the new episode
            epi_reward = 0.0
            epi_steps = []
            next_obs = env.reset()
            if len(batch) == size: # Last episode of our example
                yield batch # Returns the batch once is done
                batch = [] # After we accumulate enough batch, we will train the NN for the next episodes
        # Step t+1 in case is_done == False (Episode Not finished)
        obs = next_obs # state t+1

def elite_batch(batch, percentile): # Selects the top batches (episodes) for training the NN for next episodes
    # Core function of the Cross-entropy RL method
    # Selects the top episodes to train the NN

    # Reward parameters
    rewards = list(map(lambda s: s.reward, batch)) # Converts the batch_tuple.reward into a list
    reward_bound = np.percentile(rewards, percentile) # Returns the reward value for the q-th percentile, Every reward 'better' than this boundary is selected
    reward_mean = float(np.mean(rewards))
    # Training parameters
    train_input = [] # Input data
    train_output = [] # Output data - Target
    for episode in batch:
        if episode.reward >= reward_bound: # Elite episode
            train_input.extend(map(lambda step: step.observation, episode.steps)) # List of states per episode - Using EpisodeSteps tuple
            train_output.extend(map(lambda step: step.action, episode.steps)) # List of actions per episode

    # Convert into PyTorch.tensor
    train_input_tr = torch.FloatTensor(train_input)
    train_output_tr = torch.LongTensor(train_output)
    return train_input_tr, train_output_tr, reward_bound, reward_mean
