from SALib.sample import sobol
import numpy as np
import matplotlib.pyplot as plt

N = 256
seed = 42

colors = plt.cm.plasma(np.linspace(0, 1, 1))

np.random.seed(seed)

A4_WIDTH = 8.27 
A4_HEIGHT = 8.27/2  
fig, ax = plt.subplots(1, 2, figsize=(A4_WIDTH, A4_HEIGHT), sharex=True)
ax[0].scatter(np.random.rand(6*N,1),np.random.rand(6*N,1), color=colors[0])

problem = {
    'num_vars': 2,
    'names': ['x1', 'x2'],
    'bounds': [[0, 1],
               [0, 1]]
}

param_values = sobol.sample(problem, N)
ax[1].scatter(param_values[:,0],param_values[:,1], color=colors[0])

plt.savefig('sampling.pdf')