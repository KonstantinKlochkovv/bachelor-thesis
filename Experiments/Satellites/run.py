###
# Численность населения города по данным Росстата составляет 13 149 803[1] человек (2024)
# Численность населения Подмосковья по данным Росстата составляет 8 651 260[1] чел. (2024)
# https://rosstat.gov.ru/storage/mediabank/%D0%A1hisl_MO_01-01-2024.xlsx
# статья: в мск 1.3 млн, из мск 0.4 млн
# масштаб /200
# мск 66к, мобл 43к, 4 города по 11к
# в мск поток 0.15, в мобл 0.03
# beta tourist 0.6
### 

from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial
import concurrent.futures

cities_count = 5
rand_seed = 0
duration = 150
big_size = 660000
small_size = 110000
flow_to_msk = 0.15
flow_to_obl = 0.03/4
flow_to_neighbour = 0.01
flow_to_corner = 0.002

pop_sizes = [big_size, small_size, small_size, small_size, small_size]
pop_files = ['msk', 'mobl', 'mobl', 'mobl', 'mobl']
imports = [10, 0, 0, 0, 0]


def run_simulation(seed, start_city, start_day, multiplyer):
    adjacency_matrix = np.array([[0, flow_to_obl, flow_to_obl, flow_to_obl, flow_to_obl],
                                [flow_to_msk, 0, flow_to_neighbour, flow_to_neighbour, flow_to_corner],
                                [flow_to_msk, flow_to_neighbour, 0, flow_to_corner, flow_to_neighbour],
                                [flow_to_msk, flow_to_neighbour, flow_to_corner, 0, flow_to_neighbour],
                                [flow_to_msk, flow_to_corner, flow_to_neighbour, flow_to_neighbour, 0]])

    tourism_parameters = TourismParameters(adjacency_matrix = adjacency_matrix,
                                        time_relax = 1,
                                        contact_count = 40,
                                        beta = 0.6,
                                        intervention_data = {i: [{
                                            'start_day': start_day, 
                                            'duration': 150, 
                                            'mult_coef': multiplyer,
                                            'to_city_index': j
                                        } for j in range(cities_count) if j != i] for i in range(cities_count)})

    sims = []
    for i in range(cities_count):
        if i == start_city:
            imports = 10
        else:
            imports = 0
            
        sim = cv.Sim(pop_type='synthpops', rand_seed=seed, pop_size=pop_sizes[i],
                            n_days=duration, variants=cv.variant('wild', days=0, n_imports=imports), pop_infected=0, label=f"{i} city", verbose=-1)
        sim.load_population(popfile=f"pops/{pop_files[i]}.ppl")

        sims.append(sim)

    msim = cv.MultiSim(sims, tourism_parameters=tourism_parameters)
    msim.run(n_cpus=cities_count)

    return np.array([[msim.sims[i].results['new_infections'].values,
                      msim.sims[i].results['new_critical'].values,
                      msim.sims[i].results['new_deaths'].values] for i in range(cities_count)])




if __name__ == '__main__':
    for start_city in [0, 1]:
        for start_day in [10, 20, 30, 40, 50, 60, 65, 70, 80, 90]:
            for multiplyer in [0.1, 0.01]:
                num_processes = 5
                seeds = list(range(30))

                with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                    results = list(executor.map(partial(run_simulation, start_city=start_city, start_day=start_day, multiplyer=multiplyer), seeds))
            

                np.save(f'pkls/{start_city}_all_{start_day}_{multiplyer}.npy', np.array(results))