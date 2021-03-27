## Cross-entropy RL method - Baseline method to others (DQN, actor-critic)
# Applied for the CartPole problem

#%% Import packages
import argparse
import time
import numpy as np
import collections
# Pytorch lib
import torch
import torch.nn as nn
import torch.optim as optim
# Dashboard in the tensorflow
from tensorboardX import SummaryWriter

# Import other classes
import dqn_model
import wrappers

#%% Hyperparameters of the example
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5 # Reward every 100 epi
GAMMA = 0.99 # Discount factor for the Bellman equation
BATCH_SIZE = 32 # No of epi for the training
REPLAY_SIZE = 10000 # Max capacity of exp replay buffer
REPLAY_START_SIZE = 10000 # Threshold after the training starts
LEARNING_RATE = 1e-4 # Parameter for the Adam optimizer
SYNC_TARGET_FRAMES = 1000 # No iteration to copy the net for the target_net - Synchronize effect

# E-greedy method
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
# Create the collection of epi for the exp replay buffer
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

#%% Class section
class ExperienceBuffer: # Exp replay buffer
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # Deque list

    def __len__(self):
        return len(self.buffer)

    def append(self, experience): # Allocate new episode to the deque
        self.buffer.append(experience)

    def sample(self, batch_size): # Select randomly the episodes for the batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False) # Generate the indexes
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices]) # Get info out of each epi
        # Return the state, action, reward, done, next_state as arrays
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

# Creating the RL-agent
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self): # Initialize in state_t0
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon: # Exploration phase
            action = env.action_space.sample() # Random action
        else: # Exploitation phase
            state_a = np.array([self.state], copy=False) # Convert into np.array
            state_v = torch.tensor(state_a).to(device) # Convert into tensor
            q_vals_v = net(state_v) # Get Q-value (state,[action_1,....action_N])
            _, act_v = torch.max(q_vals_v, dim=1) # Get action: max(Q_value)
            action = int(act_v.item()) # Action value NOT index of action
        # Test in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        # Organize the step into episode --> exp replay buffer
        exp = Experience(self.state, action, reward, is_done, new_state) # List of 4 elements (each can be a tensor)
        self.exp_buffer.append(exp)
        self.state = new_state # Next state_t+1
        if is_done: # Stopping condition
            done_reward = self.total_reward # Return the final reward if we finish the episode!!
            self._reset() # Reset the env to state_t0
        return done_reward # None means we have not finish the episode in the current state_t

# Loss function of the DQN model
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

#%% Main script
if __name__ == "__main__":
    get_cuda = False # GPU option (True) or CPU option (False)
    device = torch.device("cuda" if get_cuda else "cpu")
    # Create the env
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    # Create the DQN model
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device) # Target network - Used for training
    #writer = SummaryWriter(comment="-" + args.env)
    print(net) # Print out the NN structure

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True: # Episode loop, it will run until mean_reward_{i \in 100} > max_mean_reward
        frame_idx += 1 # next state (frame)
        # Update the epsilon-greedy (exploration vs exploitation)
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        # Get environment response to state_i[frame_idx]
        reward_frame_idx = agent.play_step(net, epsilon, device=device) # Reward per state_i
        # Check if reward_i is non-empty (not None), meaning frame_i finished the game
        if reward_frame_idx is not None: # If valid, we have a frame_i that finish the game
            total_rewards.append(reward_frame_idx) # rd_i non-positive is game over, otherwise successful game
            speed = (frame_idx - ts_frame) / (time.time() - ts) # no frames / seconds
            ts_frame = frame_idx # Previous frame for the next cycle
            ts = time.time() # Previous time
            mean_reward = np.mean(total_rewards[-100:]) # mean_rd for the last 100 (successful) frames
            # Print message with the latest update
            print(f'{frame_idx} frames: done {len(total_rewards)} games, mean reward {mean_reward:.2f}, '
                  f'epsilon {epsilon:.2f}, speed {speed:.2f} frame/sec')
            # If-condition for the mean_rd performance: latest mean_rd > best mean_rd found so far. We are improving!!!
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat") # Save the NN model
                if best_mean_reward is not None: # Update message that mean_rd got improved
                    print(f'Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}, model saved')
                best_mean_reward = mean_reward # Update the best mean_rd for next evaluation
            # If-condition for the mean_rd boundary condtion: mean_rd > max bound. We guarantee a high percentage of successful games
            if mean_reward > MEAN_REWARD_BOUND: # Max_bound is now 19.5 --> 90% of games are winning cases
                # Final message
                print(f'Solved in {frame_idx} frames!')
                break # Break the endless episode
        # Check if the buffer size reached the bound of 10k frames (REPLAY_START_SIZE)
        if len(buffer) < REPLAY_START_SIZE:
            continue # Returns to the beginning of the loop, rejecting all the code below

        ## This part is only executed when: len(buffer) > Replay_size (10k frames) ##
        # Condition to create the copy target_net from net
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        # Training loop
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()