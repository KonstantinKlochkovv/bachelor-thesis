from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial
import concurrent.futures

cities_count = 5
big_size = 660000
small_size = 110000
flow_to_msk = 0.15
flow_to_obl = 0.03/4
flow_to_neighbour = 0.01
flow_to_corner = 0.002

pop_sizes = [big_size, small_size, small_size, small_size, small_size]
pop_files = ['msk', 'mobl', 'mobl', 'mobl', 'mobl']

betas = np.linspace(1,0.5,6)
flows = np.logspace(0,-4,5)
days = np.array([[150, 150, 150, 200, 200],
                [150, 150, 150, 200, 300],
                [200, 200, 200, 250, 400],
                [200, 200, 300, 300, 350],
                [300, 300, 300, 300, 600],
                [500, 500, 400, 450, 900]])

def run_simulation(seed, pattern, beta, flow_multiplyer, duration):
    adjacency_matrix = flow_multiplyer * np.array([[0, flow_to_obl, flow_to_obl, flow_to_obl, flow_to_obl],
                                                [flow_to_msk, 0, flow_to_neighbour, flow_to_neighbour, flow_to_corner],
                                                [flow_to_msk, flow_to_neighbour, 0, flow_to_corner, flow_to_neighbour],
                                                [flow_to_msk, flow_to_neighbour, flow_to_corner, 0, flow_to_neighbour],
                                                [flow_to_msk, flow_to_corner, flow_to_neighbour, flow_to_neighbour, 0]])

    tourism_parameters = TourismParameters(adjacency_matrix = adjacency_matrix,
                                        time_relax = 1,
                                        contact_count = 40,
                                        beta = 0.6)

    sims = []
    for i in range(cities_count):
        if i == pattern or pattern == 'all':
            imports = 10
        else:
            imports = 0
            
        sim = cv.Sim(pop_type='synthpops', rand_seed=seed, pop_size=pop_sizes[i],
                            n_days=duration, variants=cv.variant(label='wild', variant={'rel_beta':beta}, days=0, n_imports=imports), pop_infected=0, label=f"{i} city", verbose=-1)
        sim.load_population(popfile=f"pops/{pop_files[i]}.ppl")

        sims.append(sim)

    msim = cv.MultiSim(sims, tourism_parameters=tourism_parameters)
    msim.run(n_cpus=cities_count)

    return np.array([[msim.sims[i].results['new_infections'].values,
                      msim.sims[i].results['new_critical'].values,
                      msim.sims[i].results['new_deaths'].values] for i in range(cities_count)])




if __name__ == '__main__':
    for import_pattern in [0, 1, 'all']:
        for i, beta in enumerate(betas):
            for j, flow in enumerate(flows):
                try:
                    np.load(f'pkls/diffsatellites_{import_pattern}_{beta}_{flow}.npy')
                except:
                    num_processes = 6
                    seeds = list(range(30))
                    
                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                        results = list(executor.map(partial(run_simulation, pattern=import_pattern, beta=beta, flow_multiplyer=flow, duration=days[i,j]), seeds))
                    
                    np.save(f'pkls/diffsatellites_{import_pattern}_{beta}_{flow}.npy', np.array(results))
