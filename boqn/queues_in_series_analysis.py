import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import os
import sys
from botorch.exceptions import BadInitialCandidatesWarning
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
project_path = script_dir[:-5]
results_folder = project_path + '/experiments_results/'

# Simulator setup
from queues_in_series import queues_in_series
seed = 1
nqueues = 2
test_problem = 'queues_in_series_' + str(nqueues)
nservers = 1.#1.2 * nqueues
simulator = queues_in_series(nqueues=nqueues, arrival_rate=1., seed=seed)
from dag import DAG

dag_as_list = []
dag_as_list.append([])
for k in range(nqueues - 1):
    dag_as_list.append([k])
dag= DAG(dag_as_list)

indices_X = []
for k in range(nqueues):
    indices_X.append([k])  


# EI-QN especifics
from botorch.acquisition.objective import GenericMCObjective
from network_gp import NetworkGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

g_mapping = lambda Y: Y[..., -1]
g = GenericMCObjective(g_mapping)

def output_for_eiqn(simulator_output):
    return simulator_output[..., 0]

MC_SAMPLES = 256
BATCH_SIZE = 1

# EI especifics
from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize

def output_for_ei(simulator_output):
    return simulator_output[...,[-1], 0]


def initialize_model(train_X, train_Y, train_Yvar=None, state_dict=None):
    # define model
    #model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
    model = FixedNoiseGP(train_X, train_Y, torch.ones(train_Y.shape) * 1e-6, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
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
    rand_x = nservers * np.random.dirichlet(np.ones((nqueues, )), 1)
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
        num_restarts=10*nqueues,
        raw_samples=100*nqueues,  # used for intialization heuristic
        equality_constraints=[(torch.tensor([i for i in range(nqueues)]), torch.tensor([1. for i in range(nqueues)]), nservers)],
    )
    # observe new values 
    new_x = candidates.detach()
    new_x =  new_x.view([1, BATCH_SIZE, nqueues])
    return new_x

# Problem setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
bounds = torch.tensor([[0.0] * nqueues, [nservers] * nqueues], device=device, dtype=dtype)

def generate_initial_X(n, seed=None):
    # generate training data
    if seed is not None:
        random_state = np.random.RandomState(seed)
        train_X = random_state.dirichlet(np.ones((nqueues, )), n)
    else:
        train_X = np.random.dirichlet(np.ones((nqueues, )), n)
    train_X *= nservers
    train_X = torch.from_numpy(train_X)
    train_X = train_X.view((1, n, nqueues))
    return train_X

# Make plot

#generate initial training data
train_x = torch.tensor([0.2, 0.7,  1.0])
train_x = torch.cat([train_x.unsqueeze(-1), nservers - train_x.unsqueeze(-1)], 1).view(torch.Size([1, 3, 2]))
discretization_size = 4
X1 = torch.linspace(0., nservers, discretization_size)
X = torch.cat([X1.unsqueeze(-1), nservers - X1.unsqueeze(-1)], 1).view(torch.Size([1, discretization_size, 2]))
train_x = X

simulator_output_at_train_x = simulator.evaluate(train_x)

train_x_eiqn = train_x.clone()
train_x_ei = train_x

train_fx_eiqn = output_for_eiqn(simulator_output_at_train_x)
train_fx_ei = output_for_ei(simulator_output_at_train_x)

best_value_eiqn = g_mapping(train_fx_eiqn).max().item()
best_value_ei = train_fx_ei.max().item()

model_eiqn = NetworkGP(dag, train_x_eiqn, train_fx_eiqn, train_Yvar=None, indices_X=indices_X)
mll_ei, model_ei = initialize_model(train_x_ei, train_fx_ei)
   
    
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
new_x_ei = optimize_acqf_and_get_observation(EI)
                
#
discretization_size = 49
X1 = torch.linspace(0., nservers, discretization_size)
X = torch.cat([X1.unsqueeze(-1), nservers - X1.unsqueeze(-1)], 1).view(torch.Size([discretization_size, 1, 2]))
ei_X = EI.forward(X)
eiqn_X =  EIQN.forward(X)
objective_X = torch.flatten(output_for_ei(simulator.evaluate(X)))
posterior_eiqn_X = model_eiqn.posterior(X)
posterior_ei_X =  model_ei.posterior(X)
samples_eiqn_X =  posterior_eiqn_X.rsample(torch.Size([10000]))
mean_eiqn = torch.mean(samples_eiqn_X, 0)
mean_eiqn = torch.flatten(mean_eiqn[..., 1]).detach().numpy()
samples_eiqn_X = samples_eiqn_X.sort(0)[0]
lower_eiqn =  samples_eiqn_X[500, ...]
lower_eiqn = torch.flatten(lower_eiqn[..., 1]).detach().numpy()
upper_eiqn = samples_eiqn_X[9500, ...]
upper_eiqn = torch.flatten(upper_eiqn[..., 1]).detach().numpy()
mean_ei = posterior_ei_X.mean
mean_ei = torch.flatten(mean_ei).detach().numpy()
std_ei = torch.sqrt(posterior_ei_X.variance)
std_ei = torch.flatten(std_ei).detach().numpy()



  
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(12, 8))
# EIQN model
ax = fig.add_subplot(2, 2, 1)
ax.plot(X1, objective_X, 'k-', label="objective(x)", linewidth=1.5)
ax.plot(torch.flatten(train_x[..., 0]), torch.flatten(output_for_ei(simulator.evaluate(train_x))), 'go', label="Evaluated points")
ax.plot(X1, mean_eiqn, 'b-', label="mean_{EIQN}(x)", linewidth=1.5)
ax.plot(X1, lower_eiqn, 'b--', linewidth=1.)
ax.plot(X1, upper_eiqn, 'b--', linewidth=1.)
ax.axvline(x=torch.flatten(new_x_eiqn[..., 0]), linewidth=1.5, color='blue') 
ax.set_ylim(bottom=-5., top=15.)
ax.set(xlabel='x1', ylabel='y')
ax.legend(loc="lower right")

ax = fig.add_subplot(2, 2, 2)
ax.plot(X1, objective_X, 'k-', label="objective(x)", linewidth=1.5)
ax.plot(torch.flatten(train_x[..., 0]), torch.flatten(output_for_ei(simulator.evaluate(train_x))), 'go', label="Evaluated points")
ax.plot(X1, mean_ei, 'r-', label="mean_{EI}(x)", linewidth=1.5)
ax.plot(X1, mean_ei - 1.96*std_ei, 'r--', linewidth=1.)
ax.plot(X1, mean_ei + 1.96*std_ei, 'r--', linewidth=1.)
ax.axvline(x=torch.flatten(new_x_ei[..., 0]), linewidth=1.5, color='red') 
ax.set_ylim(bottom=-5., top=15.)
ax.set(xlabel='x1', ylabel='y')
ax.legend(loc="lower right")



ax = fig.add_subplot(2, 2, 3)
ax.plot(X1, eiqn_X.detach().numpy(), 'b-', label="EIQN(x)", linewidth=1.5) 
ax.set_ylim(bottom=0., top=1.2)
ax.set(xlabel='x1', ylabel='y')
ax.legend(loc="lower right")

ax = fig.add_subplot(2, 2, 4)
ax.plot(X1, ei_X.detach().numpy(), 'r-', label="EI(x)", linewidth=1.5) 
ax.set_ylim(bottom=0., top=1.2)
ax.set(xlabel='x1', ylabel='y')
ax.legend(loc="lower right")

plt.show()