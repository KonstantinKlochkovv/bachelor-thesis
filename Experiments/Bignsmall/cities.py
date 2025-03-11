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

from concurrent.futures import ProcessPoolExecutor

from SALib.sample import saltelli
from SALib.analyze import sobol


variable_parameters = np.array(['flow_to_small', 'flow_to_big'])

bounds = np.array([[0.001, 0.1], [0.001, 0.1]])
length = 1024
workers = 15

cites_count = 2
rand_seed = 42
duration = 150
population = 100000


def generate_parameters(variable_parameters, bounds, length):
    problem = {
        'num_vars': len(variable_parameters),
        'names': variable_parameters,
        'bounds': bounds
    }

    return problem, saltelli.sample(problem, length, calc_second_order=False)




def run_one_simulation(*args):
    try:
        adjacency_matrix = np.array([[0,args[0]],[args[1], 0]])
        tourism_parameters = TourismParameters(adjacency_matrix = adjacency_matrix)
    except:
        print('No tourism parameters')
    
    big_pop = sp.Pop.load(f'pops/big_{int(args[2])}.ppl')
    big = cv.Sim(pars={"pop_type": 'synthpops'}, rand_seed=rand_seed, pop_size=int(args[2]), 
                            n_days=duration, variants=cv.variant('wild', days=0, n_imports=30), label=f"{int(args[2])} people", verbose=-1).init_people(prepared_pop=big_pop)

    small_pop = sp.Pop.load(f'pops/small_{int(population-args[2])}.ppl')
    small = cv.Sim(pars={"pop_type": 'synthpops'}, rand_seed=rand_seed, pop_size=int(population-args[2]), 
                            n_days=duration, variants=cv.variant('wild', days=0, n_imports=0), label=f"{int(population-args[2])} people", verbose=-1).init_people(prepared_pop=small_pop)
    
    msim = cv.MultiSim([big, small], tourism_parameters=tourism_parameters)
    msim.run(n_cpus=cites_count)

    return np.array([msim.sims[0].results['cum_infections'][-1],
                     msim.sims[0].results['cum_deaths'][-1],
                     np.max(msim.sims[0].results['new_infections']),
                     np.max(msim.sims[0].results['new_critical']),
                     np.argmax(msim.sims[0].results['new_infections']),
                     msim.sims[1].results['cum_infections'][-1],
                     msim.sims[1].results['cum_deaths'][-1],
                     np.max(msim.sims[1].results['new_infections']),
                     np.max(msim.sims[1].results['new_critical']),
                     np.argmax(msim.sims[1].results['new_infections'])])




def process_task(args):
    print(args)
    return run_one_simulation(*args)




if __name__ == '__main__':
    problem, parameter_samples = generate_parameters(variable_parameters, bounds, length)

    np.save('pkls/parameters.npy', parameter_samples)
    with open('pkls/problem.pkl', 'wb') as file:
        pickle.dump(problem, file)

    # with ProcessPoolExecutor(max_workers=workers) as executor:
    #     batch_results = []

    #     for i, result in enumerate(executor.map(process_task, parameter_samples)):
    #         batch_results.append(result)

    #         if len(batch_results) == batch_length:
    #             print(f'{i} ITERATIONS SAVED')
    #             # print(batch_results)
    #             np.save(f'results/res_{i//batch_length}.npy', batch_results)
    #             batch_results = []

    #     if len(batch_results) > 0:
    #         np.save(f'results/res_{i//batch_length + 1}.npy', batch_results)
    #         del batch_results
    

    # captions = ['indexes_cum_inf_big_city', 
    #             'indexes_cum_det_big_city',
    #             'indexes_max_inf_big_city',
    #             'indexes_cum_inf_small_city',
    #             'indexes_cum_det_small_city',
    #             'indexes_max_inf_small_city']


    max_sizes = np.arange(population//2, population, 10000)
    
    for max_size in max_sizes:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batch_results = []
            for i, result in enumerate(executor.map(process_task, np.column_stack((parameter_samples, max_size*np.ones(parameter_samples.shape[0]))))):
                batch_results.append(result)

        np.save(f'pkls/big_start_res_{int(max_size)}.npy', batch_results)

        # for i, caption in enumerate(captions):
        #    with open(f'sobol/{caption}_{int(max_size)}.pkl', 'wb') as file:
        #        pickle.dump(sobol.analyze(problem, batch_results[:,i]), file)
