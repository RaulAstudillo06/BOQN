import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import savefig

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#SMALL_SIZE = 11
MEDIUM_SIZE = 11
BIGGER_SIZE = 13

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

main_path = '/home/raul/Projects/BOQN/experiments_results/'
secondary_path = ''
experiment = 'langermann'
#type_of_data = 'underlying_regret'
type_of_data = 'underlying_optimal_values'
n_replications = 10
n_iterations = 100
#algorithms = ['Random', 'ParEGO', 'uTS', 'uEI']
#legend_names = ['Random', 'ParEGO', 'TS-UU', 'EI-UU']
algorithms = ['Random', 'EI', 'EIQN']
legend_names = ['Random', 'EI', 'EI-QN', 'naive EI-QN']
colors = ['violet',  'red', 'blue', 'cyan']
dcolors = ['darkviolet', 'darkred', 'darkblue', 'darkcyan']
markers = ['^', 's', 'o', 'X']
#plt.figure()
plt.figure(figsize=(9, 6))
#plt.figure(figsize=(8, 5))
x_axis = [i+1 for i in range(n_iterations)]
for i in range(len(algorithms)):
    #measure_of_performance = np.loadtxt(main_path + secondary_path + '/' + experiment + '/' + experiment + '_' + algorithms[i] + '_' + type_of_data + '_stats.txt')[:n_iterations]
    try:
        measure_of_performance = np.loadtxt(main_path + secondary_path + experiment + '_' + algorithms[i] + '_stats.txt')[:n_iterations]
        plt.errorbar(x_axis, measure_of_performance[:, 0], yerr=1.96 * measure_of_performance[:, 1] / np.sqrt(n_replications), marker=markers[i], markersize=3.5,
                     markeredgecolor=dcolors[i], markevery=1, color=colors[i], ecolor=colors[i], errorevery=1, capsize=1.5, label=legend_names[i])
    except:
        pass

plt.title(experiment)
plt.xlabel('function evaluations')
plt.ylabel('value')
#plt.ylabel('$\log_{10}$(regret)')
plt.legend(loc='lower right')
plt.grid(True)
savefig(experiment + '.eps')
savefig(experiment + '.pdf')
plt.show()