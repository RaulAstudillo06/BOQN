import os
import sys
import numpy as np
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
torch.set_default_dtype(torch.float64)
debug._set_state(True)

from dag import DAG
from experiment_manager import experiment_manager
from alpine2 import Alpine2

# Function network
n_nodes = 4
input_dim = n_nodes
problem = "alpine2_" + str(n_nodes)
alpine2 = Alpine2(n_nodes=n_nodes)

def function_network(X: Tensor):
    return alpine2.evaluate(X=X)

# Underlying DAG
parent_nodes = []
parent_nodes.append([])
for k in range(n_nodes - 1):
    parent_nodes.append([k])

dag= DAG(parent_nodes=parent_nodes)

# Active input indices
active_input_indices = []
for k in range(n_nodes):
    active_input_indices.append([k])  


# Function that maps the network output to the objective value
network_to_objective_transform = lambda Y: Y[..., -1]
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)

# Run experiment
algos = ["EIFN"]

n_bo_iter = 100

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
    restart=True,
)