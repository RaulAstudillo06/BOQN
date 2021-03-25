import os
import sys
import numpy as np
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

from covid_si_simulator import simulate
from dag import DAG
from experiment_manager import experiment_manager

# Initial state, 1% of each population is infected
state0 = [0.01, 0.01]

# Function network
def function_network(X: Tensor) -> Tensor:
    output = torch.zeros((X.shape[0], 6))
    for i in range(X.shape[0]):
        beta = X[i, :12].view((3, 2, 2))
        gamma = X[i, 12]
        output[i, :] = 100 * torch.tensor(simulate(state0, beta, gamma)).view((6,))
    return output
        
# Observed history
# beta
beta0 = np.asarray([
	[[0.30,0.05], [0.10, 0.7]], 
	[[0.60,0.05], [0.10, 0.8]], 
	[[0.20,0.05], [0.10, 0.2]], 
])
# gamma
gamma0 = 0.5
# True underlying parameters
x0 = np.zeros(13)
x0[:12] = beta0.flatten()
x0[12] = gamma0
x0 = torch.tensor(x0).unsqueeze(dim=0)
# Observed values
y_true = function_network(x0)

# Name and input dimension
problem = 'covid_calibration'    
input_dim = 13

# Underlying DAG
n_nodes = 6
dag_as_list = []
dag_as_list.append([])
dag_as_list.append([])
dag_as_list.append([0, 1])
dag_as_list.append([0, 1])
dag_as_list.append([2, 3])
dag_as_list.append([2, 3])
dag= DAG(dag_as_list)

# Active input indices
active_input_indices = []
for k in range(3):
    active_input_indices.append([4 * k + j for j in range(4)] + [12])
    active_input_indices.append([4 * k + j for j in range(4)] + [12])
    

# Function that maps the network output to the objective value
network_to_objective_transform = lambda Y: -((Y - y_true)**2).sum(dim=-1)
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)


# Run experiment
#algos = ["EIFN", "EICF", "EI", "KG", "Random"]
algos = ["Random"]

n_bo_iter = 3

if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

experiment_manager(
    first_trial=first_trial,
    last_trial=last_trial,
    problem=problem,
    function_network=function_network,
    network_to_objective_transform=network_to_objective_transform,
    input_dim=input_dim,
    dag=dag,
    active_input_indices=active_input_indices,
    algos=algos,
    n_init_evals=2*(input_dim + 1),
    n_bo_iter=n_bo_iter,
    restart=False,
)
