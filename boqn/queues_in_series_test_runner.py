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
from queues_in_series import queues_in_series
from dag import DAG
simulator_seed = 1
n_queues = 5
n_nodes = n_queues
input_dim = n_queues
test_problem = 'queues_in_series_' + str(n_queues)
results_folder = project_path + '/experiments_results/' + test_problem + '/'
    # Define network structure
dag_as_list = []
dag_as_list.append([])
for k in range(n_queues - 1):
    dag_as_list.append([k])
dag= DAG(dag_as_list)

active_input_indices = []
for k in range(input_dim):
    active_input_indices.append([j for j in range(k + 1)])  

main_input_indices = []
for k in range(input_dim):
    main_input_indices.append([j for j in range(k + 1)])

# EI-QN especifics
from botorch.acquisition.objective import GenericMCObjective
from network_gp import NetworkGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from posterior_mean import PosteriorMean

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
from botorch.acquisition import PosteriorMean as GPPosteriorMean
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
    x = np.random.dirichlet(np.ones((input_dim, )), 1)
    x = torch.from_numpy(x)
    x = x.view([1, 1, input_dim])
    simulator_output = simulator.evaluate(x)
    fx = output_for_EI(simulator_output)
    next_Random_best = fx.max().item()
    best_Random.append(max(best_Random[-1], next_Random_best))       
    return best_Random

# Acquisition function optimization
from botorch.optim import optimize_acqf
from custom_optimizer import custom_optimize_acqf

bounds = torch.tensor([[0.] * input_dim, [1.] * input_dim])

def optimize_acqf_and_get_suggested_point(acq_func, posterior_mean):
    """Optimizes the acquisition function, and returns a new candidate."""
    baseline_candidate, _ = optimize_acqf(
        acq_function=posterior_mean,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*input_dim,
        raw_samples=100*input_dim,
        equality_constraints=[(torch.tensor([i for i in range(input_dim)]), torch.tensor([1. for i in range(input_dim)]), 1.)],
    )

    baseline_candidate = baseline_candidate.view([1, BATCH_SIZE, input_dim])
    
    candidate, acq_value = custom_optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10*input_dim,
        raw_samples=100*input_dim,
        baseline_initial_conditions=baseline_candidate,
        equality_constraints=[(torch.tensor([i for i in range(input_dim)]), torch.tensor([1. for i in range(input_dim)]), 1.)],
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
    new_x = candidate.detach()
    new_x =  new_x.view([1, BATCH_SIZE, input_dim])
    return new_x

# Function to generate initial data
def generate_initial_X(n, seed=None):
    # generate training data
    if seed is not None:
        random_state = np.random.RandomState(seed)
        X = random_state.dirichlet(np.ones((input_dim, )), n)
    else:
        X = np.random.dirichlet(np.ones((input_dim, )), n)
    X = torch.from_numpy(X)
    X = X.view((1, n, input_dim))
    return X

# Run BO loop times
N_BATCH = 25 * input_dim
simulator = queues_in_series(nqueues=n_queues, arrival_rate=1., seed=simulator_seed)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
run_Random = True
run_EI = True
run_EIQN = True
if len(sys.argv) > 1:
    trial = int(sys.argv[1])
    
    # call helper functions to generate initial training data and initialize model
    X = generate_initial_X(n=2*(input_dim+1), seed=trial)
    simulator_output_at_X = simulator.evaluate(X)
    #print(X)
    #print(simulator_output_at_X)
    
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
            print(fX_EIQN.shape)
            
            X_EIQN = torch.cat([X_EIQN, new_x_EIQN], 1)
            fX_EIQN = torch.cat([fX_EIQN, new_fx_EIQN], 1)
            
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
            print(fX_EI.shape)
            
            X_EI = torch.cat([X_EI, new_x_EI], 1)
            fX_EI = torch.cat([fX_EI, new_fx_EI], 1)
            
            mll_EI, model_EI = initialize_model(X_EI, fX_EI, model_EI.state_dict())   
            
            best_value_EI = fX_EI.max().item()
            best_observed_EI.append(best_value_EI)
            
            print('Best value so far found the EI policy: ' + str(best_value_EI) )
            np.savetxt(results_folder + test_problem + '_EI_' + str(trial) + '.txt', np.atleast_1d(best_observed_EI))
            
        if run_Random:
            best_observed_Random = update_random_observations(best_observed_Random)
            print('Best value so far found the Random policy: ' + str(best_observed_Random[-1]))
            np.savetxt(results_folder + test_problem + '_Random_' + str(trial) + '.txt', np.atleast_1d(best_observed_Random))
        print('')
