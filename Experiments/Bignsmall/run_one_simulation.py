import sys
sys.path.append('/home/klochkov_k/poem_container_server/covasim_webapp')
sys.path.append('/home/klochkov_k/poem_container_server/poem_envir/lib/python3.10/site-packages')
sys.path.append('/home/klochkov_k/poem_container_server/covasim_webapp/synthpops')
import traceback

from covasim.tourist_layer import TourismParameters
import seaborn as sns
import matplotlib.pyplot as plt
import covasim as cv
import numpy as np
import multiprocessing
import glob
import pickle
import sciris as sc
import synthpops as sp

# import optuna

cites_count = 2
rand_seed = 42
duration = 150 
pop_sizes = [90000,10000]
real_flow_to_small = 0.00035
real_flow_to_big = 0.005


if __name__ == '__main__':
    adjacency_matrix = np.array([[0, real_flow_to_small],[real_flow_to_big, 0]])
    tourism_parameters = TourismParameters(adjacency_matrix = adjacency_matrix)

    big_pop = sp.Pop.load(f'populations/big_90000.ppl')
    big = cv.Sim(pars={"pop_type": 'synthpops'}, rand_seed=rand_seed, pop_size=max(pop_sizes), 
                            n_days=duration, variants=cv.variant('wild', days=0, n_imports=30), label=f"{max(pop_sizes)} people", verbose=-1).init_people(prepared_pop=big_pop)

    small_pop = sp.Pop.load(f'populations/small_10000.ppl')
    small = cv.Sim(pars={"pop_type": 'synthpops'}, rand_seed=rand_seed, pop_size=min(pop_sizes), 
                            n_days=duration, variants=cv.variant('wild', days=0, n_imports=0), label=f"{min(pop_sizes)} people", verbose=-1).init_people(prepared_pop=small_pop)
    
    msim = cv.MultiSim([big, small], tourism_parameters=tourism_parameters)
    msim.run(n_cpus=cites_count)

    np.save('sims/big_cum_inf.npy', msim.sims[0].results['cum_infections'])
    np.save('sims/small_cum_inf.npy', msim.sims[1].results['cum_infections'])
