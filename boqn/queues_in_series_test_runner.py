import torch
import numpy as np
from botorch.exceptions import BadInitialCandidatesWarning
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Simulator setup
from queues_in_series import queues_in_series
simulator_seed = 1
nqueues = 3
simulator = queues_in_series(nqueues=nqueues, arrival_rate=1., seed=simulator_seed)
from dag import DAG
dag= DAG([[], [0], [1]])
indices_X = [[0], [1], [2]]

# EI-QN especifics
from botorch.acquisition.objective import GenericMCObjective
from network_gp import NetworkGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

g_mapping = lambda Y: Y[..., -1]
g = GenericMCObjective(g_mapping)

def output_for_eiqn(simulator_output):
    return simulator_output[..., 0]

MC_SAMPLES = 128
BATCH_SIZE = 1

# EI especifics
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch import fit_gpytorch_model

def output_for_ei(simulator_output):
    return simulator_output[...,[-1], 0]


def initialize_model(train_X, train_Y, train_Yvar=None, state_dict=None):
    # define model
    #model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

# Random especifics
def update_random_observations(best_random):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = 1.2 * np.random.dirichlet(np.ones((nqueues, )), 1)
    rand_x = torch.from_numpy(rand_x)
    rand_x = rand_x.view([1, 1, nqueues])
    simulator_output = simulator.evaluate(rand_x)
    rand_fx = output_for_ei(simulator_output)
    next_random_best = rand_fx.max().item()
    best_random.append(max(best_random[-1], next_random_best))       
    return best_random

# Acquisition function optimization
from botorch.optim import optimize_acqf

def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=5*nqueues,
        raw_samples=100*nqueues,  # used for intialization heuristic
        equality_constraints=[(torch.tensor([i for i in range(nqueues)]), torch.tensor([1. for i in range(nqueues)]), 1.2)],
    )
    # observe new values 
    new_x = candidates.detach()
    new_x =  new_x.view([1, BATCH_SIZE, nqueues])
    return new_x

# Problem setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
bounds = torch.tensor([[0.0] * nqueues, [1.2] * nqueues], device=device, dtype=dtype)
N_TRIALS = 10
N_BATCH = 40

def generate_initial_X(n, seed=None):
    # generate training data
    if seed is not None:
        random_state = np.random.RandomState(seed)
        train_X = random_state.dirichlet(np.ones((nqueues, )), n)
    else:
        train_X = np.random.dirichlet(np.ones((nqueues, )), n)
    train_X *= 1.2
    train_X = torch.from_numpy(train_X)
    train_X = train_X.view((1, n, nqueues))
    return train_X

# Run BO loop N_TRIALS times
verbose = True

best_observed_all_random, best_observed_all_ei, best_observed_all_eiqn = [], [], []

for trial in range(1, N_TRIALS + 1):
    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    best_observed_ei, best_observed_eiqn, best_observed_random = [], [], []
    
    # call helper functions to generate initial training data and initialize model
    train_x = generate_initial_X(n=2 * nqueues, seed=trial)
    simulator_output_at_train_x = simulator.evaluate(train_x)
    
    train_x_eiqn = train_x.clone()
    train_x_ei = train_x
    
    train_fx_eiqn = output_for_eiqn(simulator_output_at_train_x)
    train_fx_ei = output_for_ei(simulator_output_at_train_x)
    
    best_value_eiqn = g_mapping(train_fx_ei).max().item()
    best_value_ei = train_fx_ei.max().item()
    
    model_eiqn = NetworkGP(dag, train_x_eiqn, train_fx_eiqn, train_Yvar=None, indices_X=indices_X)
    mll_ei, model_ei = initialize_model(train_x_ei, train_fx_ei)
    
    
    
    best_observed_eiqn.append(best_value_eiqn)
    best_observed_ei.append(best_value_ei)
    best_observed_random.append(np.copy(best_value_ei))
    
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):    
        
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        EIQN = qExpectedImprovement(
            model=model_eiqn, 
            best_f=best_value_eiqn,
            sampler=qmc_sampler,
            objective=g,

        )
        
        fit_gpytorch_model(mll_ei)
        EI = ExpectedImprovement(model=model_ei, best_f=best_value_ei)
        
        # optimize and get new observation
        new_x_eiqn = optimize_acqf_and_get_observation(EIQN)
        new_fx_eiqn = output_for_eiqn(simulator.evaluate(new_x_eiqn))
        
        new_x_ei = optimize_acqf_and_get_observation(EI)
        new_fx_ei = output_for_ei(simulator.evaluate(new_x_ei))
                
        # update training data
        train_x_eiqn = torch.cat([train_x_eiqn, new_x_eiqn], 1)
        train_fx_eiqn = torch.cat([train_fx_eiqn, new_fx_eiqn], 1)
        
        train_x_ei = torch.cat([train_x_ei, new_x_ei], 1)
        train_fx_ei = torch.cat([train_fx_ei, new_fx_ei], 1)

        # update progress
        best_value_eiqn = g_mapping(train_fx_eiqn).max().item()
        best_value_ei = train_fx_ei.max().item()
        
        best_observed_eiqn.append(best_value_eiqn)
        best_observed_ei.append(best_value_ei)
        best_observed_random = update_random_observations(best_observed_random)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        model_eiqn = NetworkGP(dag, train_x_eiqn, train_fx_eiqn, indices_X=indices_X)
        
        mll_ei, model_ei = initialize_model(
            train_x_ei, 
            train_fx_ei, 
            model_ei.state_dict(),
        )
              
        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, EI, EI-QN) = "
                f"({max(best_observed_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_eiqn:>4.2f}) ", end=""
            )
        else:
            print(".", end="")
   
    best_observed_all_eiqn.append(best_observed_eiqn)
    best_observed_all_ei.append(best_observed_ei)
    best_observed_all_random.append(best_observed_random)

# Plot results   
from matplotlib import pyplot as plt

def ci(y):
    return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

iters = np.arange(N_BATCH + 1) * BATCH_SIZE
y_eiqn = np.asarray(best_observed_all_eiqn)
y_ei = np.asarray(best_observed_all_ei)
y_rnd = np.asarray(best_observed_all_random)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(iters, y_rnd.mean(axis=0), yerr=ci(y_rnd), label="random", linewidth=1.5)
ax.errorbar(iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="EI", linewidth=1.5)
ax.errorbar(iters, y_eiqn.mean(axis=0), yerr=ci(y_eiqn), label="EI-QN", linewidth=1.5)
#ax.set_ylim(bottom=0.5)
ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
ax.legend(loc="lower right")
plt.show()
