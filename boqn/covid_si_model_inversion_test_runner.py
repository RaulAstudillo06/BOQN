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
from covid_si_simulator import simulate

# Initial state, 1% of each population is infected
state0 = [0.01, 0.01]

def covid_si_simulator(X):
    output = torch.zeros((X.shape[0], 6))
    for i in range(X.shape[0]):
        beta = X[i, :12].view((3, 2, 2))
        gamma = X[i, 12]
        output[i, :] = 100 * torch.tensor(simulate(state0, beta, gamma)).view((6,))
    return output
        
# Correct value for the history
# beta
beta0 = np.asarray([
	[[0.30,0.05], [0.10, 0.7]], 
	[[0.60,0.05], [0.10, 0.8]], 
	[[0.20,0.05], [0.10, 0.2]], 
])
# gamma
gamma0 = 0.5

x0 = np.zeros(13)
x0[:12] = beta0.flatten()
x0[12] = gamma0

x0 = torch.tensor(x0).unsqueeze(dim=0)
y_true = covid_si_simulator(x0)
    
input_dim = 13
test_problem = 'covid_si_inversion'
results_folder = project_path + '/experiments_results/' + test_problem + '/'
# Define network structure
from dag import DAG
n_nodes = 6
dag_as_list = []
dag_as_list.append([])
dag_as_list.append([])
dag_as_list.append([0, 1])
dag_as_list.append([0, 1])
dag_as_list.append([2, 3])
dag_as_list.append([2, 3])
dag= DAG(dag_as_list)

active_input_indices = []
for k in range(3):
    active_input_indices.append([4 * k + j for j in range(4)] + [12])
    active_input_indices.append([4 * k + j for j in range(4)] + [12])

main_input_indices = []
for k in range(3):
    main_input_indices.append([4 * k + j for j in range(4)] + [12])
    main_input_indices.append([4 * k + j for j in range(4)] + [12])
    

# EI-QN especifics
from botorch.acquisition.objective import GenericMCObjective
from network_gp import NetworkGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from posterior_mean import PosteriorMean

g_mapping = lambda Y: -((Y - y_true)**2).sum(dim=-1)
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
    return -((simulator_output - y_true)**2).sum(dim=-1, keepdim=True)

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

# Random especifics
def update_random_observations(best_Random):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    x = torch.rand([1, input_dim])
    simulator_output = covid_si_simulator(x)
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
    if baseline_acq_value >= acq_value:
        print('Baseline candidate was best found.')
        candidate = baseline_candidate
    new_x = candidate.detach().view([BATCH_SIZE, input_dim])

    return new_x

# GP model training
def initialize_model(X, Y, Yvar=None):
    # define model
    model = FixedNoiseGP(X, Y, torch.ones(Y.shape) * 1e-6, outcome_transform=Standardize(m=Y.shape[-1], batch_shape=torch.Size([])))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# Initial data generation
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
N_BATCH = 260
init_batch_id = 1

if not os.path.exists(results_folder) :
    os.makedirs(results_folder)
if not os.path.exists(results_folder + 'X/'):
    os.makedirs(results_folder + 'X/')
if not os.path.exists(results_folder + 'Y/'):
    os.makedirs(results_folder + 'Y/')
if not os.path.exists(results_folder + 'running_times/'):
    os.makedirs(results_folder + 'running_times/')

run_EIQN = False
run_EICF = False
run_EI = False
run_KG = True
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
        simulator_output_at_X = covid_si_simulator(X)
        
        if run_EIQN:
            best_observed_EIQN = []
            X_EIQN = X.clone()
            fX_EIQN = output_for_EIQN(simulator_output_at_X)
            best_value_EIQN = g_mapping(fX_EIQN).max().item()
            best_observed_EIQN.append(best_value_EIQN)
        if run_EICF:
            best_observed_EICF = []
            X_EICF = X.clone()
            fX_EICF = output_for_EIQN(simulator_output_at_X)
            mll_EICF, model_EICF = initialize_model(X_EICF, fX_EICF)
            best_value_EICF = g_mapping(fX_EICF).max().item()
            best_observed_EICF.append(best_value_EICF)
        if run_EI:
            best_observed_EI = []
            X_EI = X.clone()
            fX_EI = output_for_EI(simulator_output_at_X)
            mll_EI, model_EI = initialize_model(X_EI, fX_EI)
            best_value_EI = fX_EI.max().item()
            best_observed_EI.append(best_value_EI)
        if run_KG:
            try:
                best_observed_KG = list(np.loadtxt(results_folder + test_problem + '_KG_' + str(trial) + '.txt'))
                running_times_KG = list(np.loadtxt(results_folder + 'running_times/' + test_problem + '_rt_KG_' + str(trial) + '.txt'))
                best_value_KG = torch.tensor(best_observed_KG[-1])
                X_KG = torch.tensor(np.loadtxt(results_folder + 'X/' + test_problem + '_X_KG_' + str(trial) + '.txt'))
                fX_KG = torch.tensor(np.loadtxt(results_folder + 'Y/' + test_problem + '_Y_KG_' + str(trial) + '.txt')).unsqueeze(dim=-1)
                print(fX_KG.shape)
                init_batch_id = len(best_observed_KG)
                print("Restarting experiment from available data.")
            except:
                best_observed_KG = []
                running_times_KG = []
                X_KG = X.clone()
                fX_KG = output_for_EI(simulator_output_at_X)
                best_value_KG = fX_KG.max().item()
                best_observed_KG.append(best_value_KG)
            mll_KG, model_KG = initialize_model(X_KG, fX_KG)
        if run_Random:
            best_observed_Random = []
            best_observed_Random.append(output_for_EI(simulator_output_at_X).max().item())
        
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(init_batch_id, N_BATCH + 1):
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
                new_fx_EIQN = output_for_EIQN(covid_si_simulator(new_x_EIQN))

                X_EIQN = torch.cat([X_EIQN, new_x_EIQN], 0)
                fX_EIQN = torch.cat([fX_EIQN, new_fx_EIQN], 0)
                
                best_value_EIQN = g_mapping(fX_EIQN).max().item()
                best_observed_EIQN.append(best_value_EIQN)
                print('Best value so far found the EIQN policy: ' + str(best_value_EIQN) )
                np.savetxt(results_folder + test_problem + '_EIQN_' + str(trial) + '.txt', np.atleast_1d(best_observed_EIQN))
                
            if run_EICF:
                fit_gpytorch_model(mll_EICF)
                qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
                EICF = qExpectedImprovement(
                    model=model_EICF, 
                    best_f=best_value_EICF,
                    sampler=qmc_sampler,
                    objective=g,
        
                )
                posterior_mean_EICF = PosteriorMean(
                    model=model_EICF, 
                    sampler=qmc_sampler,
                    objective=g,
                )

                new_x_EICF = optimize_acqf_and_get_suggested_point(EICF, posterior_mean_EICF)
                new_fx_EICF = output_for_EIQN(covid_si_simulator(new_x_EICF))

                X_EICF = torch.cat([X_EICF, new_x_EICF], 0)
                fX_EICF = torch.cat([fX_EICF, new_fx_EICF], 0)
                
                mll_EICF, model_EICF = initialize_model(X_EICF, fX_EICF)
                
                best_value_EICF = g_mapping(fX_EICF).max().item()
                best_observed_EICF.append(best_value_EICF)
                
                print('Best value so far found the EICF policy: ' + str(best_value_EICF) )
                np.savetxt(results_folder + test_problem + '_EICF_' + str(trial) + '.txt', np.atleast_1d(best_observed_EICF))
                np.savetxt(results_folder + 'X/' + test_problem + '_X_EICF_' + str(trial) + '.txt', X_EICF.numpy())
                np.savetxt(results_folder + 'Y/' + test_problem + '_Y_EICF_' + str(trial) + '.txt', fX_EICF.numpy())
                
            if run_EI:
                fit_gpytorch_model(mll_EI)
                EI = ExpectedImprovement(model=model_EI, best_f=best_value_EI)
                posterior_mean_EI = GPPosteriorMean(model=model_EI)
                
                new_x_EI = optimize_acqf_and_get_suggested_point(EI, posterior_mean_EI)
                new_fx_EI = output_for_EI(covid_si_simulator(new_x_EI))
                
                X_EI = torch.cat([X_EI, new_x_EI], 0)
                fX_EI = torch.cat([fX_EI, new_fx_EI], 0)
                
                mll_EI, model_EI = initialize_model(X_EI, fX_EI)   
                
                best_value_EI = fX_EI.max().item()
                best_observed_EI.append(best_value_EI)
                
                print('Best value so far found the EI policy: ' + str(best_value_EI) )
                np.savetxt(results_folder + test_problem + '_EI_' + str(trial) + '.txt', np.atleast_1d(best_observed_EI))
                
            if run_KG:
                t0 = time.time()
                fit_gpytorch_model(mll_KG)
                
                KG = qKnowledgeGradient(model=model_KG, num_fantasies=8)
                
                new_x_KG = optimize_KG_and_get_suggested_point(KG)
                
                mll_KG, model_KG = initialize_model(X_KG, fX_KG)
                
                t1 = time.time()
                running_times_KG.append(t1 - t0)
                
                new_fx_KG = output_for_EI(covid_si_simulator(new_x_KG))
                
                X_KG = torch.cat([X_KG, new_x_KG], 0)
                fX_KG = torch.cat([fX_KG, new_fx_KG], 0)
                
                best_value_KG = fX_KG.max().item()
                best_observed_KG.append(best_value_KG)
                
                print('Best value so far found the KG policy: ' + str(best_value_KG) )
                np.savetxt(results_folder + test_problem + '_KG_' + str(trial) + '.txt', np.atleast_1d(best_observed_KG))
                np.savetxt(results_folder + 'running_times/' + test_problem + '_rt_KG_' + str(trial) + '.txt', np.atleast_1d(running_times_KG))
                np.savetxt(results_folder + 'X/' + test_problem + '_X_KG_' + str(trial) + '.txt', X_KG.numpy())
                np.savetxt(results_folder + 'Y/' + test_problem + '_Y_KG_' + str(trial) + '.txt', fX_KG.numpy())
                
            if run_Random:
                best_observed_Random = update_random_observations(best_observed_Random)
                print('Best value so far found the Random policy: ' + str(best_observed_Random[-1]))
                np.savetxt(results_folder + test_problem + '_Random_' + str(trial) + '.txt', np.atleast_1d(best_observed_Random))
            print('')
