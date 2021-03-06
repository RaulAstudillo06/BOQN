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


# Simulator setup
from alpine2 import Alpine2
from dag import DAG
n_nodes = 4
input_dim = n_nodes
test_problem = 'alpine2_' + str(n_nodes)
results_folder = project_path + '/experiments_results/' + test_problem + '/'
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
from posterior_mean import PosteriorMean

g_mapping = lambda Y: Y[..., -1]
g = GenericMCObjective(g_mapping)

def output_for_EIQN(simulator_output):
    return simulator_output

MC_SAMPLES = 512
BATCH_SIZE = 1

# EI especifics
from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize

def output_for_EI(simulator_output):
    return simulator_output[...,[-1]]


def initialize_model(X, Y, Yvar=None, state_dict=None):
    # define model
    model = FixedNoiseGP(X, Y, torch.ones(Y.shape) * 1e-6, outcome_transform=Standardize(m=1, batch_shape=torch.Size([])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

# MVES especifics
from botorch.acquisition import qMaxValueEntropy
from botorch.utils import draw_sobol_samples

# KG especifics
from botorch.acquisition import qKnowledgeGradient

def optimize_KG_and_get_suggested_point(acq_func):
    """Optimizes the KG acquisition function, and returns a new candidate."""
    
    candidate, _ = custom_optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*input_dim,
        raw_samples=100*input_dim,
        #options={'disp': True, 'iprint': 101},
    )
    
    new_x = candidate.detach()
    return new_x

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

# Random especifics
def update_random_observations(best_Random):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    x = torch.rand([1, input_dim])
    simulator_output = simulator.evaluate(x)
    fx = output_for_EI(simulator_output)
    next_Random_best = fx.max().item()
    best_Random.append(max(best_Random[-1], next_Random_best))       
    return best_Random

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

    baseline_candidate = baseline_candidate.detach().view([1, BATCH_SIZE, input_dim])
    
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
    if baseline_acq_value >= acq_value:
        print('Baseline candidate was best found.')
        candidate = baseline_candidate
    new_x = candidate.detach().view([BATCH_SIZE, input_dim])
    return new_x

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

# Run BO loop times
N_BATCH = 100
simulator = Alpine2(n_nodes=n_nodes)
def my_objective(X):
    print(g_mapping(simulator.evaluate(X))[..., 0].shape)
    return g_mapping(simulator.evaluate(X))[..., 0]

if False:
    x_opt, val_opt = optimize_acqf(
        acq_function=my_objective,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*input_dim,
        raw_samples=100*input_dim,
        #options={'iprint': 101},
    )
if not os.path.exists(results_folder) :
    os.makedirs(results_folder)

run_EIQN = True
run_EI = False
run_MVES = False
run_KGQN = False
run_KG = False
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
        simulator_output_at_X = simulator.evaluate(X)
        
        if run_EIQN:
            best_observed_EIQN = []
            X_EIQN = X.clone()
            fX_EIQN = output_for_EIQN(simulator_output_at_X)
            best_value_EIQN = g_mapping(fX_EIQN).max().item()
            best_observed_EIQN.append(best_value_EIQN)
        if run_EI:
            best_observed_EI = []
            X_EI = X.clone()
            fX_EI = output_for_EI(simulator_output_at_X)
            mll_EI, model_EI = initialize_model(X_EI, fX_EI)
            best_value_EI = fX_EI.max().item()
            best_observed_EI.append(best_value_EI)  
        if run_MVES:
            best_observed_MVES = []
            X_MVES = X.clone()
            fX_MVES = output_for_EI(simulator_output_at_X)
            mll_MVES, model_MVES = initialize_model_KG(X_MVES, fX_MVES)
            best_value_MVES = fX_MVES.max().item()
            best_observed_MVES.append(best_value_MVES)
        if run_KGQN:
            best_observed_KGQN = []
            X_KGQN = X.clone()
            fX_KGQN = output_for_EIQN(simulator_output_at_X)
            best_value_KGQN = fX_KGQN.max().item()
            best_observed_KGQN.append(best_value_KGQN)
        if run_KG:
            best_observed_KG = []
            X_KG = X.clone()
            fX_KG = output_for_EI(simulator_output_at_X)
            mll_KG, model_KG = initialize_model(X_KG, fX_KG)
            best_value_KG = fX_KG.max().item()
            best_observed_KG.append(best_value_KG)
        if run_Random:
            best_observed_Random = []
            best_observed_Random.append(output_for_EI(simulator_output_at_X).max().item())
        
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):
            print('Experiment: ' + test_problem)
            print('Replication id: ' + str(trial))
            print('Iteration: ' + str(iteration))
            if run_EIQN:
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
                posterior_mean_EIQN = PosteriorMean(
                    model=model_EIQN, 
                    sampler=qmc_sampler,
                    objective=g,
                )
                t0 = time.time()
                new_x_EIQN = optimize_acqf_and_get_suggested_point(EIQN, posterior_mean_EIQN)
                t1 = time.time()
                print('Optimizing the acquisition function took: ' + str(t1 - t0))
                new_fx_EIQN = output_for_EIQN(simulator.evaluate(new_x_EIQN))

                X_EIQN = torch.cat([X_EIQN, new_x_EIQN], 0)
                fX_EIQN = torch.cat([fX_EIQN, new_fx_EIQN], 0)
                
                best_value_EIQN = g_mapping(fX_EIQN).max().item()
                best_observed_EIQN.append(best_value_EIQN)
                print('Best value so far found the EIQN policy: ' + str(best_value_EIQN) )
                np.savetxt(results_folder + test_problem + '_EIQN_' + str(trial) + '.txt', np.atleast_1d(best_observed_EIQN))
                
            if run_EI:
                fit_gpytorch_model(mll_EI)
                EI = ExpectedImprovement(model=model_EI, best_f=best_value_EI)
                posterior_mean_EI = GPPosteriorMean(model=model_EI)
                
                new_x_EI = optimize_acqf_and_get_suggested_point(EI, posterior_mean_EI)
                new_fx_EI = output_for_EI(simulator.evaluate(new_x_EI))
                
                X_EI = torch.cat([X_EI, new_x_EI], 0)
                fX_EI = torch.cat([fX_EI, new_fx_EI], 0)
                
                mll_EI, model_EI = initialize_model(X_EI, fX_EI, model_EI.state_dict())   
                
                best_value_EI = fX_EI.max().item()
                best_observed_EI.append(best_value_EI)
                
                print('Best value so far found the EI policy: ' + str(best_value_EI) )
                np.savetxt(results_folder + test_problem + '_EI_' + str(trial) + '.txt', np.atleast_1d(best_observed_EI))
                
            if run_MVES:
                fit_gpytorch_model(mll_MVES)
                candidate_set = torch.squeeze(draw_sobol_samples(bounds, 200*input_dim, BATCH_SIZE))
                MVES = qMaxValueEntropy(model_MVES, candidate_set)
                posterior_mean_MVES = GPPosteriorMean(model=model_MVES)
                
                new_x_MVES = optimize_acqf_and_get_suggested_point(MVES, posterior_mean_MVES)
                new_fx_MVES = output_for_EI(simulator.evaluate(new_x_MVES))
                
                X_MVES = torch.cat([X_MVES, new_x_MVES], 0)
                fX_MVES = torch.cat([fX_MVES, new_fx_MVES], 0)
                
                mll_MVES, model_MVES = initialize_model_KG(X_MVES, fX_MVES, model_MVES.state_dict())   
                
                best_value_MVES = fX_MVES.max().item()
                best_observed_MVES.append(best_value_MVES)
                
                print('Best value so far found the MVES policy: ' + str(best_value_MVES) )
                np.savetxt(results_folder + test_problem + '_MVES_' + str(trial) + '.txt', np.atleast_1d(best_observed_MVES))
                
            if run_KGQN:
                t0 = time.time()
                model_KGQN = NetworkGP(dag, X_KGQN, fX_KGQN, active_input_indices=active_input_indices, main_input_indices=main_input_indices)
                t1 = time.time()
                print('Training the model took: ' + str(t1 - t0))
                
                KGQN = qKnowledgeGradient(model=model_KGQN, num_fantasies=8, objective=g)
                
                t0 = time.time()
                new_x_KGQN = optimize_KG_and_get_suggested_point(KGQN)
                t1 = time.time()
                print('Optimizing the acquisition function took: ' + str(t1 - t0))
                new_fx_KGQN = output_for_EIQN(simulator.evaluate(new_x_KGQN))
                
                X_KGQN = torch.cat([X_KGQN, new_x_KGQN], 0)
                fX_KGQN = torch.cat([fX_KGQN, new_fx_KGQN], 0)
                
                best_value_KGQN = g_mapping(fX_KGQN).max().item()
                best_observed_KGQN.append(best_value_KGQN)
                print('Best value so far found the KGQN policy: ' + str(best_value_KGQN) )
                np.savetxt(results_folder + test_problem + '_KGQN_' + str(trial) + '.txt', np.atleast_1d(best_observed_KGQN))
                
            if run_KG:
                fit_gpytorch_model(mll_KG)
                KG = qKnowledgeGradient(model=model_KG, num_fantasies=8)
                
                new_x_KG = optimize_KG_and_get_suggested_point(KG)
                new_fx_KG = output_for_EI(simulator.evaluate(new_x_KG))
                
                X_KG = torch.cat([X_KG, new_x_KG], 0)
                fX_KG = torch.cat([fX_KG, new_fx_KG], 0)
                
                mll_KG, model_KG = initialize_model(X_KG, fX_KG, model_KG.state_dict())   
                
                best_value_KG = fX_KG.max().item()
                best_observed_KG.append(best_value_KG)
                
                print('Best value so far found the KG policy: ' + str(best_value_KG) )
                np.savetxt(results_folder + test_problem + '_KG_' + str(trial) + '.txt', np.atleast_1d(best_observed_KG))
                
            if run_Random:
                best_observed_Random = update_random_observations(best_observed_Random)
                print('Best value so far found the Random policy: ' + str(best_observed_Random[-1]))
                np.savetxt(results_folder + test_problem + '_Random_' + str(trial) + '.txt', np.atleast_1d(best_observed_Random))
            print('')

