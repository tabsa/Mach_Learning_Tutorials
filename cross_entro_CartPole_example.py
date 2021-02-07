## Cross-entropy RL method - Baseline method to others (DQN, actor-critic)
# Applied for the CartPole problem

#%% Import packages
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

#%% Hyperparameters of the example
hidden_layer_size = 128
batch_size = 16
reward_top_per = 70 # q-th percentile of 'elite' episodes = 1 - reward_top

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

#%% Function section
# Tuples to keep information over episodes
Episode = namedtuple('Episode', field_names=['reward', 'steps']) # Stores the total_reward and episode_steps
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) # Stores the state (env_obs) and action per step

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

#%% Main Script
if __name__ == '__main__':
    # Define Environment and parameters
    env = gym.make("CartPole-v0") # Env creation using gym lib
    #env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0] # state size
    action_size = env.action_space.n # Action size

    # Define the cross-entropy agent
    cre_agent = net(obs_size, action_size, hidden_layer_size) # Class the net
    train_obj = nn.CrossEntropyLoss() # Used for multi-class classification-type problems, to deal with probability distribution
    # nn.CrossEntropyLoss: receives values before softmax fun, so before the probability distribution
    train_opt = optim.Adam(params=cre_agent.parameters(), lr=0.01) # cre_agent has the parameters since is also defined by nn.Module
    # lr parameter is the learning rate

    writer = SummaryWriter(comment='-cartpole') # TensorBoard result

    for iter, episode in enumerate(iterate_batch(env, cre_agent, batch_size)): # Enumerate gives the iter and item.value (batch)
        obs_tr, act_tr, reward_b, reward_m = elite_batch(episode, reward_top_per)
        train_opt.zero_grad() # Level to zero our NN gradients
        act_scores_tr = cre_agent(obs_tr) # NN prediction to the elite episode
        loss = train_obj(act_scores_tr, act_tr) # Newly loss to reinforce the NN for the elite episode
        loss.backward() # Calculate the gradients of the weights
        train_opt.step() # Update the NN weights with the optimizer
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