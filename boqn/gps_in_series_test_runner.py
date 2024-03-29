import os
import sys
import time
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import botorch
from botorch.settings import debug
debug._set_state(True)

# Get script directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-5]
results_folder = project_path + '/experiments_results/'

# Simulator setup
from gps_in_series import GPs_in_series
from dag import DAG
simulator_seed = 1
n_nodes = 4
test_problem = 'gps_in_series_' + str(n_nodes)
    # Define network structure
dag_as_list = []
dag_as_list.append([])
for k in range(n_nodes - 1):
    dag_as_list.append([k])
dag= DAG(dag_as_list)

active_input_indices = []
for k in range(n_nodes):
    active_input_indices.append([k])  

main_input_indices = []
for k in range(n_nodes):
    main_input_indices.append([k]) 

# EI-QN especifics
from botorch.acquisition.objective import GenericMCObjective
from network_gp import NetworkGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

g_mapping = lambda Y: Y[..., -1]
g = GenericMCObjective(g_mapping)

def output_for_EIQN(simulator_output):
    return simulator_output

MC_SAMPLES = 512
BATCH_SIZE = 1

# EI especifics
from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize

def output_for_EI(simulator_output):
    return simulator_output[...,[-1]]


def initialize_model(X, Y, Yvar=None, state_dict=None):
    # define model
    #model = HeteroskedasticSingleTaskGP(X, Y, Yvar)
    model = FixedNoiseGP(X, Y, torch.ones(Y.shape) * 1e-4, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

# Random especifics
def update_random_observations(best_Random):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    x = torch.rand([1, 1, n_nodes])
    simulator_output = simulator.evaluate(x)
    fx = output_for_EI(simulator_output)
    next_Random_best = fx.max().item()
    best_Random.append(max(best_Random[-1], next_Random_best))       
    return best_Random

# Acquisition function optimization
from botorch.optim import optimize_acqf

bounds = torch.tensor([[0. for i in range(n_nodes)], [1. for i in range(n_nodes)]])

def optimize_acqf_and_get_suggested_point(acq_func):
    """Optimizes the acquisition function, and returns a new candidate."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*n_nodes,
        raw_samples=100*n_nodes,
        #options={'iprint': 101},
    )
    # suggested point(s)
    new_x = candidates.detach()
    new_x =  new_x.view([1, BATCH_SIZE, n_nodes])
    return new_x
    
# Function to generate initial data
def generate_initial_X(n, seed=None):
    # generate training data
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        X = torch.rand([1, n, n_nodes])
        torch.random.set_rng_state(old_state)
    else:
        X = torch.rand([1, n, n_nodes])
    return X

# Run BO loop times
N_BATCH = 5
simulator = GPs_in_series(n_nodes=n_nodes, seed=simulator_seed)

from objective_function import ObjectiveFunction

def obj_func_caller(X):
    print(g_mapping(simulator.evaluate(X))[..., 0].shape)
    return g_mapping(simulator.evaluate(X))[..., 0]

obj_func = ObjectiveFunction(obj_func_caller)

if False:
    x_opt, val_opt = optimize_acqf(
        acq_function=obj_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*n_nodes,
        raw_samples=100*n_nodes,
    )

if not os.path.exists(results_folder) :
            os.makedirs(results_folder)
if len(sys.argv) > 1:
    trial = int(sys.argv[1])

    best_observed_EI, best_observed_EIQN, best_observed_Random = [], [], []
    
    # call helper functions to generate initial training data and initialize model
    X = generate_initial_X(n=2*(n_nodes+1), seed=trial)
    simulator_output_at_X = simulator.evaluate(X)
    print(X)
    print(simulator_output_at_X)
    
    X_EIQN = X.clone()
    X_EI = X
    
    fX_EIQN = output_for_EIQN(simulator_output_at_X)
    fX_EI = output_for_EI(simulator_output_at_X)
    
    best_value_EIQN = g_mapping(fX_EI).max().item()
    best_value_EI = fX_EI.max().item()
    
    mll_EI, model_EI = initialize_model(X_EI, fX_EI)
    
    
    
    best_observed_EIQN.append(best_value_EIQN)
    best_observed_EI.append(best_value_EI)
    best_observed_Random.append(np.copy(best_value_EI))
    
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):    
                
        # optimize and get new observation
        t0 = time.time()
        model_EIQN = NetworkGP(dag, X_EIQN, fX_EIQN, active_input_indices=active_input_indices, main_input_indices=main_input_indices)
        t1 = time.time()
        print('Training the model took: ' + str(t1 - t0))
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        EIQN = qExpectedImprovement(
            model=model_EIQN, 
            best_f=best_value_EIQN,
            sampler=qmc_sampler,
            objective=g,

        )
        t0 = time.time()
        new_x_EIQN = optimize_acqf_and_get_suggested_point(EIQN)
        t1 = time.time()
        print('Optimizing the acquisition function took: ' + str(t1 - t0))
        new_fx_EIQN = output_for_EIQN(simulator.evaluate(new_x_EIQN))
        
        fit_gpytorch_model(mll_EI)
        EI = ExpectedImprovement(model=model_EI, best_f=best_value_EI)
        
        new_x_EI = optimize_acqf_and_get_suggested_point(EI)
        new_fx_EI = output_for_EI(simulator.evaluate(new_x_EI))
                
        # update training data
        X_EIQN = torch.cat([X_EIQN, new_x_EIQN], 1)
        fX_EIQN = torch.cat([fX_EIQN, new_fx_EIQN], 1)
        
        X_EI = torch.cat([X_EI, new_x_EI], 1)
        fX_EI = torch.cat([fX_EI, new_fx_EI], 1)

        # update progress
        best_value_EIQN = g_mapping(fX_EIQN).max().item()
        best_value_EI = fX_EI.max().item()
        
        best_observed_EIQN.append(best_value_EIQN)
        best_observed_EI.append(best_value_EI)
        best_observed_Random = update_random_observations(best_observed_Random)

        # rEInitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        model_EIQN = NetworkGP(dag, X_EIQN, fX_EIQN, active_input_indices=active_input_indices, main_input_indices=main_input_indices)
        
        mll_EI, model_EI = initialize_model(
            X_EI, 
            fX_EI, 
            model_EI.state_dict(),
        )
        
        np.savetxt(results_folder + test_problem + '_EIQN_' + str(trial) + '.txt', np.atleast_1d(best_observed_EIQN))
        np.savetxt(results_folder + test_problem + '_EI_' + str(trial) + '.txt', np.atleast_1d(best_observed_EI))
        np.savetxt(results_folder + test_problem + '_Random_' + str(trial) + '.txt', np.atleast_1d(best_observed_Random))
        
        print(
            f"\nBatch {iteration:>2}: best_value (Random, EI, EI-QN) = "
            f"({max(best_observed_Random):>4.4f}, {best_value_EI:>4.4f}, {best_value_EIQN:>4.4f}), ", end=""
        )
