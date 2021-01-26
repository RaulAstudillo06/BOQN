import os
import sys
import time

from copy import copy

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import botorch
from botorch.settings import debug
debug._set_state(True)

# Get script directory
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-5]

# Problem setup
from dropwave import Dropwave
dropwave = Dropwave()
input_dim = 2
test_problem = 'dropwave'
results_folder = project_path + '/experiments_results/' + test_problem + '/'

# Define network structure
from dag import DAG

n_nodes = 2
dag_as_list = [[]]
dag_as_list.append([0])
dag= DAG(dag_as_list)

active_input_indices = [[0, 1], []]
main_input_indices = copy(active_input_indices)
    
# EI-QN especifics
from botorch.acquisition.objective import GenericMCObjective
from network_gp import NetworkGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from posterior_mean import PosteriorMean

g_mapping = lambda Y: Y[..., -1]
g = GenericMCObjective(g_mapping)

def output_for_EIQN(simulator_output):
    return simulator_output

MC_SAMPLES = 512
BATCH_SIZE = 1

# EI especifics
from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize

def output_for_EI(simulator_output):
    return simulator_output[...,[-1]]

def initialize_model(X, Y, Yvar=None):
    # define model
    model = FixedNoiseGP(X, Y, torch.ones(Y.shape) * 1e-6, outcome_transform=Standardize(m=Y.shape[-1], batch_shape=torch.Size([])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# Random especifics
def update_random_observations(best_Random):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    x = torch.rand([1, input_dim])
    simulator_output = dropwave.evaluate(x)
    fx = output_for_EI(simulator_output)
    next_Random_best = fx.max().item()
    best_Random.append(max(best_Random[-1], next_Random_best))       
    return best_Random

# Function to generate initial data
def generate_initial_X(n, seed=None):
    # generate training data
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        X = torch.rand([n, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        X = torch.rand([n, input_dim])
    return X

# Acquisition function optimization
from botorch.optim import optimize_acqf
from custom_optimizer import custom_optimize_acqf

bounds = torch.tensor([[0. for i in range(input_dim)], [1. for i in range(input_dim)]])

def optimize_acqf_and_get_suggested_point(acq_func, posterior_mean):
    """Optimizes the acquisition function, and returns a new candidate."""
    baseline_candidate, _ = optimize_acqf(
        acq_function=posterior_mean,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*input_dim,
        raw_samples=100*input_dim,
    )

    baseline_candidate = baseline_candidate.detach().view(torch.Size([1, BATCH_SIZE, input_dim]))
    
    candidate, acq_value = custom_optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*input_dim,
        raw_samples=100*input_dim,
        baseline_initial_conditions=baseline_candidate,
        #options={'disp': True, 'iprint': 101},
    )
    
    baseline_acq_value = acq_func.forward(baseline_candidate)[0].detach()
    print('Test begins')
    print(acq_value)
    print(baseline_acq_value)
    print('Test ends')
    if baseline_acq_value > acq_value:
        print('Baseline candidate was best found.')
        new_x = baseline_candidate
    elif baseline_acq_value == acq_value:
        p = np.random.rand(1)
        if p > 0.5:
            new_x = baseline_candidate
        else:
            new_x = candidate
    else:
        new_x = candidate
        
    new_x = new_x.detach().view([BATCH_SIZE, input_dim])
    return new_x

# Run BO loop times
N_BATCH = 100

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if not os.path.exists(results_folder + 'X/'):
    os.makedirs(results_folder + 'X/')
if not os.path.exists(results_folder + 'Y/'):
    os.makedirs(results_folder + 'Y/')
if not os.path.exists(results_folder + 'running_times/'):
    os.makedirs(results_folder + 'running_times/')

    
run_EIQN = True
run_EI = False
run_Random = False

if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

if len(sys.argv) > 1:
    for trial in range(first_trial, last_trial + 1):
    
        # call helper functions to generate initial training data and initialize model
        X = generate_initial_X(n=2*(input_dim+1), seed=trial)
        simulator_output_at_X = dropwave.evaluate(X)
        
        if run_EIQN:
            best_observed_EIQN = []
            running_times_EIQN = []
            X_EIQN = X.clone()
            fX_EIQN = simulator_output_at_X
            best_value_EIQN = g_mapping(fX_EIQN).max().item()
            best_observed_EIQN.append(best_value_EIQN)
        if run_EI:
            best_observed_EI = []
            running_times_EI = []
            X_EI = X.clone()
            fX_EI = output_for_EI(simulator_output_at_X)
            mll_EI, model_EI = initialize_model(X_EI, fX_EI)
            best_value_EI = fX_EI.max().item()
            best_observed_EI.append(best_value_EI)  
        if run_Random:
            best_observed_Random = []
            running_times_Random = []
            best_observed_Random.append(output_for_EI(simulator_output_at_X).max().item())
        
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):
            print('Experiment: ' + test_problem)
            print('Replication id: ' + str(trial))
            print('Iteration: ' + str(iteration))
            if run_EIQN:
                t0 = time.time()
                
                model_EIQN = NetworkGP(dag, X_EIQN, fX_EIQN, active_input_indices=active_input_indices, main_input_indices=main_input_indices)
                
                qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
                EIQN = qExpectedImprovement(
                    model=model_EIQN, 
                    best_f=best_value_EIQN,
                    sampler=qmc_sampler,
                    objective=g,
        
                )
                posterior_mean_EIQN = PosteriorMean(
                    model=model_EIQN, 
                    sampler=qmc_sampler,
                    objective=g,
                )
                
                new_x_EIQN = optimize_acqf_and_get_suggested_point(EIQN, posterior_mean_EIQN)
                print('Candidate suggested by the EIQN policy: ' + str(new_x_EIQN))
                
                t1 = time.time()
                running_times_EIQN.append(t1 - t0)
                
                new_fx_EIQN = dropwave.evaluate(new_x_EIQN)

                X_EIQN = torch.cat([X_EIQN, new_x_EIQN], 0)
                fX_EIQN = torch.cat([fX_EIQN, new_fx_EIQN], 0)
                
                best_value_EIQN = g_mapping(fX_EIQN).max().item()
                best_observed_EIQN.append(best_value_EIQN)
                print('Best value so far found the EIQN policy: ' + str(best_value_EIQN) )
                np.savetxt(results_folder + test_problem + '_EIQN_' + str(trial) + '.txt', np.atleast_1d(best_observed_EIQN))
                np.savetxt(results_folder + 'running_times/' + test_problem + '_rt_EIQN_' + str(trial) + '.txt', np.atleast_1d(running_times_EIQN))
                np.savetxt(results_folder + 'X/' + test_problem + '_X_EIQN_' + str(trial) + '.txt', X_EIQN.numpy())
                np.savetxt(results_folder + 'Y/' + test_problem + '_Y_EIQN_' + str(trial) + '.txt', fX_EIQN.numpy())
                
            if run_EI:
                t0 = time.time()
                fit_gpytorch_model(mll_EI)
                
                EI = ExpectedImprovement(model=model_EI, best_f=best_value_EI)
                posterior_mean_EI = GPPosteriorMean(model=model_EI)
                
                new_x_EI = optimize_acqf_and_get_suggested_point(EI, posterior_mean_EI)
                
                mll_EI, model_EI = initialize_model(X_EI, fX_EI)
                
                t1 = time.time()
                running_times_EI.append(t1 - t0)
                
                new_fx_EI = output_for_EI(dropwave.evaluate(new_x_EI))
                
                X_EI = torch.cat([X_EI, new_x_EI], 0)
                fX_EI = torch.cat([fX_EI, new_fx_EI], 0)
                
                best_value_EI = fX_EI.max().item()
                best_observed_EI.append(best_value_EI)
                
                print('Best value so far found the EI policy: ' + str(best_value_EI) )
                np.savetxt(results_folder + test_problem + '_EI_' + str(trial) + '.txt', np.atleast_1d(best_observed_EI))
                np.savetxt(results_folder + 'running_times/' + test_problem + '_rt_EI_' + str(trial) + '.txt', np.atleast_1d(running_times_EI))
                np.savetxt(results_folder + 'X/' + test_problem + '_X_EI_' + str(trial) + '.txt', X_EI.numpy())
                np.savetxt(results_folder + 'Y/' + test_problem + '_Y_EI_' + str(trial) + '.txt', fX_EI.numpy())
                
            if run_Random:
                best_observed_Random = update_random_observations(best_observed_Random)
                print('Best value so far found the Random policy: ' + str(best_observed_Random[-1]))
                np.savetxt(results_folder + test_problem + '_Random_' + str(trial) + '.txt', np.atleast_1d(best_observed_Random))
            print('')
