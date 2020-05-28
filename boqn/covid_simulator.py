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
        #X[0, 0, :] = torch.tensor([.82945, .87082])
        #X[0, 1, :] = torch.tensor([.87945, .84082])
        X_scaled = 199. * X + 1. 
        input_shape = X_scaled.shape
        output = torch.zeros(input_shape[:-1] + torch.Size([3 * self.n_periods]))
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                states, losses = simple_simulation(list(X_scaled[i, j, :]), seed=self.seed)
                states = torch.tensor(states)
                losses = torch.tensor(losses)
                for t in range(self.n_periods):
                    aux = (1 - states[t, 0] - states[t, 1])
                    output[i, j, 3 * t] = torch.log(states[t, 0]/aux) 
                    output[i, j, 3 * t + 1] = torch.log(states[t, 1]/aux)
                    output[i, j, 3 * t + 2] = losses[t]
        return output
                