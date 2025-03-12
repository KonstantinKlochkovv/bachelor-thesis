from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial
import concurrent.futures

betas = np.linspace(1,0.5,6)
flows = np.logspace(-2,-5,7)
days = np.array([[150, 150, 150, 200, 200, 200, 200],
                [150, 150, 150, 200, 200, 300, 300],
                [200, 200, 200, 250, 250, 400, 400],
                [200, 200, 200, 300, 300, 350, 350],
                [300, 300, 300, 300, 300, 300, 300],
                [400, 400, 400, 450, 450, 450, 450]])




cities_count = 2

def run_simulation(seed, beta, flow, duration):
    adjacency_matrix = np.array([[0, flow],
                                 [flow, 0]])

    tourism_parameters = TourismParameters(adjacency_matrix = adjacency_matrix)

    sims = []
    for i in range(cities_count):
        if i == 0:
            imports = 10
        else:
            imports = 0

        sim = cv.Sim(pop_type='synthpops', rand_seed=seed, pop_size=1e5,
                            n_days=duration, variants=cv.variant(label='wild', variant={'rel_beta':beta}, days=0, n_imports=imports), pop_infected=0, label=f"{i} city", verbose=-1)
        sim.load_population(popfile=f"pops/100k.ppl")

        sims.append(sim)

    msim = cv.MultiSim(sims, tourism_parameters=tourism_parameters)
    msim.run(n_cpus=cities_count)

    return np.array([[msim.sims[i].results['new_infections'].values,
                      msim.sims[i].results['new_critical'].values,
                      msim.sims[i].results['new_deaths'].values] for i in range(cities_count)])




if __name__ == '__main__':
    for i, beta in enumerate(betas):
        for j, flow in enumerate(flows):
            num_processes = 15
            seeds = list(range(30))
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                results = list(executor.map(partial(run_simulation, beta=beta, flow=flow, duration=days[i,j]), seeds))
            
            np.save(f'pkls/basicflows_{beta}_{flow}.npy', np.array(results))