import concurrent.futures
from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import multiprocessing as mp

sim = cv.Sim(rand_seed=0, pop_size=1000000,
                            n_days=700, variants=[cv.variant(label=f'{x}', variant={'rel_beta':x}, days=0, n_imports=30) for x in np.linspace(0.4,1,7)], verbose=-1)
# sim = cv.Sim(rand_seed=0, pop_size=1000000,
#                             n_days=500, variants=[cv.variant(label=f'{x}', variant={'rel_beta':x}, days=0, n_imports=30) for x in [0.4]], verbose=-1)

sim.run()
sim.plot('variant')

# print(sim.results)