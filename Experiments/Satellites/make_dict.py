from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import multiprocessing as mp

cites_count = 5
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

start_day = 10
multiplyer = 0.1

intervention_data = {i: [{
                            'start_day': start_day, 
                            'duration': 150, 
                            'mult_coef': multiplyer,
                            'to_city_index': j
                        } for j in range(cites_count) if (j != i and (i == 1 or j == 1))] for i in range(cites_count)}

print(intervention_data)