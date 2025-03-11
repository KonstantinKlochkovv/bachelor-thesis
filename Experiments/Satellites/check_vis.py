import sys
sys.path.append('../../../poem/covasim_webapp')
sys.path.append('../../../poem/covasim_webapp/synthpops')

from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

with open("check_before.pkl", "rb") as f:
    msim = pickle.load(f)

stats = msim.sims[0].tourism_stats['in_city_count'] + msim.sims[1].tourism_stats['in_city_count']
ax.plot(np.arange(len(stats)), stats, label='Пуассон')

with open("check_after.pkl", "rb") as f:
    msim = pickle.load(f)

stats = msim.sims[0].tourism_stats['in_city_count'] + msim.sims[1].tourism_stats['in_city_count']
ax.plot(np.arange(len(stats)), stats, label='Пуассон без 0')

with open("check_after_after.pkl", "rb") as f:
    msim = pickle.load(f)

stats = msim.sims[0].tourism_stats['in_city_count'] + msim.sims[1].tourism_stats['in_city_count']
print(stats)
ax.plot(np.arange(len(stats)), stats, label='Пуассон без 0 + фикс туристов')

ax.legend()
ax.set_xlim(1,100)
ax.set_ylim(194000,201000)
ax.set_xlabel('День')
ax.set_ylabel('Количество людей в двух городах in_city')
plt.tight_layout()
plt.savefig("after.png")

