import torch
from matplotlib import pyplot as plt
simulator_seed = 1
n_periods = 3
from covid_simulator import CovidSimulator
simulator = CovidSimulator(n_periods=n_periods, seed=simulator_seed)

def output_for_EI(simulator_output):
    output =  -100 * torch.sum(simulator_output[..., [3*t + 2 for t in range(n_periods)]], dim=-1, keepdim=True)
    return output

discretization_size = 29
X = torch.ones([1, discretization_size, 3])
X[..., 1] = torch.linspace(0., 1., discretization_size)

if False:
    fX = output_for_EI(simulator.evaluate(X))
    fX = torch.flatten(fX)
    fig = plt.figure(figsize=(12, 8))
    # EIQN model
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X[0, :, 1], fX, 'k-', label="objective(x)", linewidth=1.5)
    #ax.set_ylim(bottom=-5., top=15.)
    ax.set(xlabel='x2', ylabel='y')
    ax.legend(loc="lower right")
    plt.show()

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir + '/group_testing/src')
from dynamic_protocol_design import test_properties

    
if True:
    prevalence=0.1
    group_sizes =  97.*X[0, :, 1] + 3.
    QFNR = []
    QFPR = []
    tests_per_person = []
    for group_size in group_sizes:
        print(group_size)
        out1, out2, out3 = test_properties(prevalence, group_size, n_households=10000, nreps = 50)
        QFNR.append(out1)
        QFPR.append(out2)
        tests_per_person.append(out3)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(group_sizes, QFNR, 'k-', label="QFNR", linewidth=1.5)
    ax.set(xlabel='x2', ylabel='y')
    ax.legend(loc="lower right")
    
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(group_sizes, QFPR, 'k-', label="QFPR", linewidth=1.5)
    ax.set(xlabel='x2', ylabel='y')
    ax.legend(loc="lower right")
    
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(group_sizes, tests_per_person, 'k-', label="Tests per person", linewidth=1.5)
    ax.set(xlabel='x2', ylabel='y')
    ax.legend(loc="lower right")
    plt.show()