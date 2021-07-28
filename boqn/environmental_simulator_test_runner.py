import os
import sys
import numpy as np
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

from environmental_simulator import EnvironmentalSimulator
from dag import DAG
from experiment_manager import experiment_manager

# Grid
grid_s = torch.tensor([0.0, 1.0, 2.5])
grid_t = torch.tensor([15.0, 30.0, 45.0, 60.0])


# Function network
environmental_model = EnvironmentalSimulator(grid=(grid_s,grid_t))
def function_network(X: Tensor) -> Tensor:
    bounds = environmental_model.bounds
    X_unscaled = X.clone()
    X_unscaled[..., 0] = (bounds[0][1] - bounds[0][0]) * X[..., 0] + bounds[0][0]
    X_unscaled[..., 1] = (bounds[1][1] - bounds[1][0]) * X[..., 1] + bounds[1][0]
    X_unscaled[..., 2] = (bounds[2][1] - bounds[2][0]) * X[..., 2] + bounds[2][0]
    X_unscaled[..., 3] = (bounds[3][1] - bounds[3][0]) * X[..., 3] + bounds[3][0]
    print(environmental_model.evaluate(X_unscaled))
    return environmental_model.evaluate(X_unscaled)
        
# Target vector
target_vector = environmental_model.target_values

# Name and input dimension
problem = 'environmental_simulator'    
input_dim = 4

# Underlying DAG
n_nodes = 12
dag_as_list = []
for k in range(n_nodes):
    dag_as_list.append([])

dag= DAG(dag_as_list)

# Active input indices
active_input_indices = []
for k in range(n_nodes):
    active_input_indices.append([j for j in range(input_dim)])
    
# Function that maps the network output to the objective value
network_to_objective_transform = lambda Y: -((Y - target_vector)**2).sum(dim=-1)
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)


# Run experiment
#algos = ["EIFN", "EICF", "EI", "KG", "Random"]
algos = ["EICF"]

n_bo_iter = 50

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
