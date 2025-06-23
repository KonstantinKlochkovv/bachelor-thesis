from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial
import concurrent.futures

labels = np.load('data/labels.npy', allow_pickle=True)

if __name__ == '__main__':
    for start_city in [9, 12]:
        seed = 0

        with open(f'pkls/seed_{seed}_city_{start_city}_pickle.pkl', 'rb') as file:
            msim = pickle.load(file)

        tourism_stats = {i: msim.sims[i].tourism_stats for i in range(len(labels))}
    
        with open(f'pkls/tourism_stats_{seed}_{labels[start_city]}.pkl', 'wb') as file:
            pickle.dump(tourism_stats, file)