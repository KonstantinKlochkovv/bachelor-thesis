import numpy as np
from SALib.analyze import sobol
import pickle
import matplotlib.pyplot as plt

population = 2e5
max_sizes = np.arange(1e4, population, 1e4)
captions = ['cum_infections',
            'cum_deaths',
            'max_new_infections',
            'max_new_critical',
            'time_peak_infections']

colors = plt.cm.plasma(np.linspace(0, 0.8, 2))

with open('pkls/problem.pkl', 'rb') as file:
    problem = pickle.load(file)


# for i, caption in enumerate(captions):
#     A4_WIDTH = 8.27  # Ширина A4
#     A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
#     fig, ax = plt.subplots(2, 2, figsize=(A4_WIDTH, A4_HEIGHT))

#     for j, index in enumerate(['S1', 'ST']):
#         for big_size in max_sizes:
#             data = np.load(f'pkls/big_start_res_{int(big_size)}.npy') 

#             for k, parameter in enumerate(['flow_to_small', 'flow_to_big']):
#                 indices = sobol.analyze(problem, data[:,i], calc_second_order=False)
#                 ax[j][k].errorbar(x=big_size//1e4, y=indices[index][k], yerr=indices[f'{index}_conf'][k], color=colors[0], capsize=5, fmt='.')

#                 indices = sobol.analyze(problem, data[:,i+len(captions)], calc_second_order=False)
#                 ax[j][k].errorbar(x=big_size//1e4, y=indices[index][k], yerr=indices[f'{index}_conf'][k], color=colors[1], capsize=5, fmt='.')

#                 ax[j][k].set_title(f'{index}_{parameter}_{caption}')
    
#     plt.tight_layout()
#     plt.savefig(f'graphs/index_{caption}.pdf')



A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(2, 5, figsize=(A4_WIDTH, A4_HEIGHT))

index = 'S1'
for i in range(2):
    for k, caption in enumerate(captions):
        for big_size in max_sizes:
            data = np.load(f'pkls/bignsmall_{int(big_size)}.npy') 

            indices = sobol.analyze(problem, data[:,k], calc_second_order=False)
            ax[i][k].errorbar(x=big_size//1e4, y=indices[index][i], yerr=indices[f'{index}_conf'][i], color=colors[0], capsize=5, fmt='.', label=i)

            indices = sobol.analyze(problem, data[:,k+len(captions)], calc_second_order=False)
            ax[i][k].errorbar(x=big_size//1e4, y=indices[index][i], yerr=indices[f'{index}_conf'][i], color=colors[1], capsize=5, fmt='.', label=i)

            ax[i][k].set_title(f'{caption}')

ax[0][0].set_ylabel('flow_to_small')
ax[1][0].set_ylabel('flow_to_big')
plt.tight_layout()
plt.savefig(f'graphs/flow_to_small_S1_index.pdf')
