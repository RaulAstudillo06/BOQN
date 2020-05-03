import numpy as np
import matplotlib.pyplot as plt

main_path = '/home/raul/Projects/BOQN/experiments_results/'
secondary_path = ''
experiment = 'queues_in_series_2_'
sampling_policy = 'eiqn'
type_of_data = 'value'
#type_of_data = 'underlying_regret'
n_files = 10
n_iterations = 40
data = np.empty((n_iterations, n_files))
min_number_of_iterations = n_iterations
i_aux = 0
for i in range(n_files):
    try:
        print(i)
        if type_of_data == 'underlying_regret':
            aux = np.log10(np.loadtxt(main_path + secondary_path + '/' + experiment + '/historical_underlying_regret/' + experiment + '_' +sampling_policy + '_' + type_of_data + '_' + str(i + 1) + '.txt', unpack=True)[:n_iterations])
        elif type_of_data == 'value':
            aux = np.loadtxt(main_path + secondary_path + experiment + '_' + sampling_policy + '_' + str(i + 1) + '.txt', unpack=True)[:n_iterations]
        elif type_of_data == 'integrated_optimal_values':
            aux = np.log10(1. - np.loadtxt(main_path + secondary_path + '/' + experiment + '/historical_integrated_optimal_values/' + experiment + '_' +sampling_policy + '_' + type_of_data + '_' + str(i + 1) + '.txt', unpack=True)[:n_iterations])
        l = len(aux)
        data[:l, i_aux] = aux
        for j in range(l, n_iterations):
            data[j, i_aux] = aux[l-1]
        if l < min_number_of_iterations:
            min_number_of_iterations = l
            replication_with_fewest_iterations = i
        data[:, i_aux] = data[:, i_aux]
        i_aux += 1
    except:
        pass


n_available_files = i_aux
data = data[:, :n_available_files]
print('Minum number of iterations: {}'.format(min_number_of_iterations))
if min_number_of_iterations < n_iterations:
    print('File with fewest iterations: {}'.format(replication_with_fewest_iterations + 1))
print('Number of available files: {}'.format(n_available_files))
data_stats = np.zeros((n_iterations, 2))
data_stats[:, 0] = np.mean(data, axis=1)
data_stats[:, 1] = np.std(data, axis=1)
np.savetxt(main_path + secondary_path + experiment + '_' +sampling_policy + '_stats.txt', data_stats)
plt.plot(data_stats[:, 0], label=experiment + '_' +sampling_policy)
plt.plot(data_stats[:, 0] + data_stats[:, 1])
plt.plot(data_stats[:, 0] - data_stats[:, 1])
plt.show()