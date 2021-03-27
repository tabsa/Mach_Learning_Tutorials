## DQN structure
# This file creates the DQN model that would be used in the Atari games
# This example follows the successfull implementation from DeepMind of 2013 and 2015
# It follows the Nature paper of 2015 "Human-level control through Deep Reinforcement Learing"

#%% Import packages
import torch
import torch.nn as nn
import numpy as np

#%% Class for the DQN
# DQN is composed by convolution NN (3 layers) followed by feed-forward NN (2 layers)
# All layers use the ReLU (rectified linear fun) as activation function
class DQN(nn.Module): # Inherents the object class nn.Module
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # Convolution NN with 3 layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Feed-forward NN with 2 layers
        conv_out_size = self._get_conv_out(input_shape) # Get the size of output as 1D tensor
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions) # Output - Q_val(state, [act_1,...,act_N])
            # Returns an array of Q_value per action_i
        )

    def _get_conv_out(self, shape): # Function to transform the output of convNN
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size())) # Returns the output with the size for building the feed-foward NN

    def forward(self, x): # Function to call nn.Module.forward - to predict with x as input
        # x is a 4D tensor - [batch, Color, Height, Width]
        # self.conv returns a 4D tensor for the prediction
        # We have to transform for the feed-forward NN - only accepts 1D tensor per batch
        conv_out = self.conv(x).view(x.size()[0], -1) # Transform the 3D tensor into 1D tensor (flatten the image shape[CHW] )
        # It returns a 2D tensor with shape [batch, Color x Height x Width]
        return self.fc(conv_out) # Returns the output as 1D tensor
