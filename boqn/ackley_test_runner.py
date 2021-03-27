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
from ackley import Ackley

# Function network
n_nodes = 3
input_dim = 6
problem = "ackley"

ackley = Ackley(input_dim=input_dim)

def function_network(X: Tensor):
    return ackley.evaluate(X=X)

# Underlying DAG
parent_nodes = []
for k in range(n_nodes - 1):
    parent_nodes = [].append([])

parent_nodes.append([k for k in range(n_nodes - 1)])
dag = DAG(parent_nodes=parent_nodes)

# Active input indices
active_input_indices = []
for k in range(n_nodes - 1):
    active_input_indices.append([i for i in range(input_dim)])

active_input_indices.append([])

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