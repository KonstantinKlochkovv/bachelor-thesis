import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import seaborn as sns
import math


def round_to_first_significant(x):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))))
                 

betas = np.linspace(1,0.5,6)
flows = np.logspace(-2,-5,7)
days = np.array([[150, 150, 150, 200, 200, 200, 200],
        [150, 150, 150, 200, 200, 300, 300],
        [200, 200, 200, 250, 250, 400, 400],
        [200, 200, 200, 300, 300, 350, 350],
        [300, 300, 300, 300, 300, 300, 300],
        [400, 400, 400, 450, 450, 450, 450]])

cities_count = 2

colors = plt.cm.plasma(np.linspace(0, 0.8, cities_count))

A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT), sharex=True, sharey=True)
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = np.load(f'pkls/basicflows_{beta}_{flow}.npy')

        ax[i][j].plot(np.arange(data.shape[-1]), np.mean(data[:,0,0,:], axis=0), color=colors[0])
        ax[i][j].fill_between(np.arange(data.shape[-1]), np.mean(data[:,0,0,:], axis=0)-np.std(data[:,0,0,:], axis=0), np.mean(data[:,0,0,:], axis=0)+np.std(data[:,0,0,:], axis=0), alpha=0.2, color=colors[0])
        ax[i][j].plot(np.arange(data.shape[-1]), np.mean(data[:,1,0,:], axis=0), color=colors[1])
        ax[i][j].fill_between(np.arange(data.shape[-1]), np.mean(data[:,1,0,:], axis=0)-np.std(data[:,1,0,:], axis=0), np.mean(data[:,1,0,:], axis=0)+np.std(data[:,1,0,:], axis=0), alpha=0.2, color=colors[1])

plt.tight_layout()
plt.savefig('graphs/flows_epids.pdf')




t_stats_day = np.zeros((len(betas), len(flows)))
t_stats_max = np.zeros((len(betas), len(flows)))

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = np.load(f'pkls/basicflows_{beta}_{flow}.npy')

        t_stat_day, p_value_day = sps.ttest_ind(np.argmax(data[:,1,0,:], axis=1), np.argmax(data[:,0,0,:], axis=1), equal_var=False)
        t_stats_day[i,j] = t_stat_day

        t_stat_max, p_value_max = sps.ttest_ind(np.max(data[:,1,0,:], axis=1), np.max(data[:,0,0,:], axis=1), equal_var=False)
        t_stats_max[i,j] = t_stat_max


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 8.27/2  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(1, 2, figsize=(A4_WIDTH, A4_HEIGHT), sharey=True)

sns.heatmap(t_stats_day, annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
            cmap="plasma", cbar_kws={'label': 't-статистика'}, ax=ax[0])

sns.heatmap(t_stats_max, annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
            cmap="plasma", cbar_kws={'label': 't-статистика'}, ax=ax[1])

ax[0].set_xlabel("Поток")
ax[0].set_ylabel("Трансмиссивность")
ax[0].set_title("t-статистики дня пика")
ax[1].set_xlabel("Поток")
ax[1].set_title("t-статистики пика инфицирований")

plt.tight_layout()
plt.savefig('graphs/flows_heatmap.pdf')