from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial
import concurrent.futures

duration = 1000

sizes = np.load('data/populations.npy', allow_pickle=True)
labels = np.load('data/labels.npy', allow_pickle=True)

cities_count = len(labels)


def run_simulation(seed, start_city, beta):
    adjacency_matrix = np.load('data/flows.npy', allow_pickle=True)

    tourism_parameters = TourismParameters(adjacency_matrix = adjacency_matrix)

    sims = []
    for i in range(cities_count):
        if i == start_city:
            imports = 30
        else:
            imports = 0

        sim = cv.Sim(pop_type='synthpops', rand_seed=seed, pop_size=sizes[i],
                            n_days=duration, variants=cv.variant(label='wild', variant={'rel_beta':beta}, days=0, n_imports=imports), pop_infected=0, label=labels[i], verbose=-1)
        sim.load_population(popfile=f"pops/{labels[i]}.ppl")

        sims.append(sim)

    msim = cv.MultiSim(sims, tourism_parameters=tourism_parameters)
    msim.run(n_cpus=cities_count)

    return np.array([[msim.sims[i].results['new_infections'].values,
                      msim.sims[i].results['new_critical'].values,
                      msim.sims[i].results['new_deaths'].values] for i in range(cities_count)])




if __name__ == '__main__':
    for start_city in [9, 12]:
        num_processes = 1
        beta = 1.0
        seeds = [0]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(partial(run_simulation, start_city=start_city, beta=beta), seeds))

        np.save(f'pkls/{labels[start_city]}.npy', np.array(results))