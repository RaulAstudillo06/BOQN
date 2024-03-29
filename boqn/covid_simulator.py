import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + '/group_testing/src')

import torch
from dynamic_protocol_design import simple_simulation


class CovidSimulator:

    def __init__(self, n_periods, seed=1):
        self.n_periods = n_periods
        self.seed =  seed
        
    def evaluate(self, X):
        X_scaled = 199. * X + 1. 
        input_shape = X_scaled.shape
        output = torch.zeros(input_shape[:-1] + torch.Size([3 * self.n_periods]))
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                states, losses = simple_simulation(list(X_scaled[i, j, :]), seed=self.seed)
                states = torch.tensor(states)
                losses = torch.tensor(losses)
                for t in range(self.n_periods):
                    output[i, j, 3 * t] = states[t, 0]
                    output[i, j, 3 * t + 1] = states[t, 1]
                    output[i, j, 3 * t + 2] = losses[t]
        #print(X)
        #print(output)
        return output
                