import os
import sys
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
from queues_in_series import queues_in_series
from dag import DAG
simulator_seed = 1
nqueues = 4
test_problem = 'queues_in_series_' + str(nqueues)
    # Define network structure
dag_as_list = []
dag_as_list.append([])
for k in range(nqueues - 1):
    dag_as_list.append([k])
dag= DAG(dag_as_list)

active_input_indices = []
for k in range(nqueues):
    active_input_indices.append([j for j in range(k + 1)])  

main_input_indices = []
for k in range(nqueues):
    main_input_indices.append([k]) 

# EI-QN especifics
from botorch.acquisition.objective import GenericMCObjective
from network_gp import NetworkGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

g_mapping = lambda Y: Y[..., -1]
g = GenericMCObjective(g_mapping)

def output_for_EIQN(simulator_output):
    return simulator_output[..., 0]

MC_SAMPLES = 512
BATCH_SIZE = 1

# EI especifics
from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize

def output_for_EI(simulator_output):
    return simulator_output[...,[-1], 0]


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
    x = np.random.dirichlet(np.ones((nqueues, )), 1)
    x = torch.from_numpy(x)
    x = x.view([1, 1, nqueues])
    simulator_output = simulator.evaluate(x)
    fx = output_for_EI(simulator_output)
    next_Random_best = fx.max().item()
    best_Random.append(max(best_Random[-1], next_Random_best))       
    return best_Random

# Acquisition function optimization
from botorch.optim import optimize_acqf

bounds = torch.tensor([[0.] * nqueues, [1.] * nqueues])

def optimize_acqf_and_get_suggested_point(acq_func):
    """Optimizes the acquisition function, and returns a new candidate."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*nqueues,
        raw_samples=100*nqueues,
        equality_constraints=[(torch.tensor([i for i in range(nqueues)]), torch.tensor([1. for i in range(nqueues)]), 1.)],
        #options={'disp': True},
    )
    # suggested point(s)
    new_x = candidates.detach()
    new_x =  new_x.view([1, BATCH_SIZE, nqueues])
    return new_x

# Function to generate initial data
def generate_initial_X(n, seed=None):
    # generate training data
    if seed is not None:
        random_state = np.random.RandomState(seed)
        X = random_state.dirichlet(np.ones((nqueues, )), n)
    else:
        X = np.random.dirichlet(np.ones((nqueues, )), n)
    X = torch.from_numpy(X)
    X = X.view((1, n, nqueues))
    return X

# Run BO loop times
N_BATCH = 2
simulator = queues_in_series(nqueues=nqueues, arrival_rate=1., seed=simulator_seed)
if not os.path.exists(results_folder) :
            os.makedirs(results_folder)
if len(sys.argv) > 1:
    trial = int(sys.argv[1])

    best_observed_EI, best_observed_EIQN, best_observed_Random = [], [], []
    
    # call helper functions to generate initial training data and initialize model
    X = generate_initial_X(n=2 * nqueues, seed=trial)
    simulator_output_at_X = simulator.evaluate(X)
    
    X_EIQN = X.clone()
    X_EI = X
    
    fX_EIQN = output_for_EIQN(simulator_output_at_X)
    fX_EI = output_for_EI(simulator_output_at_X)
    
    best_value_EIQN = g_mapping(fX_EI).max().item()
    best_value_EI = fX_EI.max().item()
    
    model_EIQN = NetworkGP(dag, X_EIQN, fX_EIQN, active_input_indices=active_input_indices, main_input_indices=main_input_indices)
    mll_EI, model_EI = initialize_model(X_EI, fX_EI)
    
    
    
    best_observed_EIQN.append(best_value_EIQN)
    best_observed_EI.append(best_value_EI)
    best_observed_Random.append(np.copy(best_value_EI))
    
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):    
        
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        EIQN = qExpectedImprovement(
            model=model_EIQN, 
            best_f=best_value_EIQN,
            sampler=qmc_sampler,
            objective=g,

        )
        
        fit_gpytorch_model(mll_EI)
        EI = ExpectedImprovement(model=model_EI, best_f=best_value_EI)
        
        # optimize and get new observation
        new_x_EIQN = optimize_acqf_and_get_suggested_point(EIQN)
        new_fx_EIQN = output_for_EIQN(simulator.evaluate(new_x_EIQN))
        
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
            f"({max(best_observed_Random):>4.2f}, {best_value_EI:>4.2f}, {best_value_EIQN:>4.2f}), ", end=""
        )
