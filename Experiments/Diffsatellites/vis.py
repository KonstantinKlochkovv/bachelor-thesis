import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import seaborn as sns
import math
from scipy import signal

def round_to_first_significant(x):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))))

def filter_fliers(data):
    return np.array([row for row in data if np.all(np.max(row, axis=1) > 100)])

cities_count = 5
seed_count = 30
betas = np.linspace(1,0.5,6)
flows = np.logspace(0,-2,5)[::-1]
import_patterns = [0, 1]

maes = np.zeros((len(import_patterns), len(betas), len(flows)))
t_stats = np.zeros((len(betas), len(flows)))


for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        temp = {key:[] for key in import_patterns}

        for import_pattern in import_patterns:
            data = np.load(f'pkls/diffsatellites_{import_pattern}_{beta}_{flow}.npy')[:,:,0,:]

            for seed in range(seed_count):
                if np.max(np.min(data[seed,1:,:], axis=0)) > 100:
                    temp[import_pattern].append(np.mean(np.abs(data[seed,1:,:] - np.mean(data[seed,1:,:], axis=0)), axis=1)/np.max(np.mean(data[seed,1:,:], axis=0)))

            temp[import_pattern] = np.array(temp[import_pattern]).flatten()
            maes[import_pattern, i, j] = np.mean(temp[import_pattern])
        
        t_stat, p_value = sps.ttest_ind(temp[1], temp[0], equal_var=False)
        t_stats[i, j] = t_stat

        # print(beta, flow, len(temp[0]), len(temp[1]))

        

A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 8.27/2  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(1, 2, figsize=(A4_WIDTH, A4_HEIGHT))

sns.heatmap(t_stats, annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
        cmap="plasma", ax=ax[1], cbar_kws={'label': 't-статистика'})

sns.heatmap(maes[1, :, :] - maes[0, :, :], annot=True, fmt=".3f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
        cmap="plasma", ax=ax[0], cbar_kws={'label': 'Среднее значение $\Delta$MAE/max'})

ax[1].set_xlabel("Множитель транспортного потока")
ax[1].set_ylabel("Трансмиссивность")
ax[0].set_xlabel("Множитель транспортного потока")
ax[0].set_ylabel("Трансмиссивность")

plt.tight_layout()
plt.savefig('graphs/diffsatellites_heatmaps.png', dpi=600)
plt.savefig('graphs/diffsatellites_heatmaps.pdf')


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(1, len(import_patterns), figsize=(A4_WIDTH, A4_HEIGHT))
colors = plt.cm.plasma(np.linspace(0, 0.8, cities_count))

for import_pattern in import_patterns:
    data = np.load(f'pkls/diffsatellites_{import_pattern}_1.0_1.0.npy')[:,:,0,:]

    for city in range(cities_count):
        for seed in range(seed_count-15):
            if np.max(data[seed,city,:]) > 100 and city > 0:
                ax[import_pattern].plot(np.arange(len(data[seed,city,:])), data[seed,city,:], color=colors[city])

plt.savefig('log.pdf')

A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(2, len(flows), figsize=(A4_WIDTH, A4_HEIGHT), sharey=True)

for import_pattern in [0, 1]:
    for j,flow in enumerate(flows):
        data = np.load(f'pkls/diffsatellites_{import_pattern}_1.0_{flow}.npy')[:,:,0,:]

        for city in range(1, cities_count):
            for row in data[:,city,:]:
                ax[import_pattern][j].plot(np.arange(data.shape[-1]), row, color=colors[city], linewidth=1, alpha=0.1) 
        
        ax[import_pattern][j].tick_params(axis='both', which='major', labelsize=4)
        ax[import_pattern][j].tick_params(axis='both', which='minor', labelsize=4)
        ax[import_pattern][j].set_xlim(0,200)

for j,flow in enumerate(flows):
    ax[-1][j].set_xlabel(round_to_first_significant(flow))

ax[0][0].set_ylabel('Начало в хабе')
ax[1][0].set_ylabel('Начало в сателлите')
fig.supxlabel('Множитель транспортного потока')
fig.supylabel('Город начала эпидемии')
plt.tight_layout()
plt.savefig('graphs/diffsatellites_demos.png', dpi=600)
plt.savefig('graphs/diffsatellites_demos.pdf')

A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT), sharex=True, sharey=True)
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        for import_pattern in [0, 1]:
            data = np.load(f'pkls/diffsatellites_{import_pattern}_{beta}_{flow}.npy')[:,:,0,:]
            
            for city in range(1, cities_count):
                for row in data[:,city,:]:
                    ax[i][j].plot(np.arange(data.shape[-1]), row, color=colors[city], linewidth=1, alpha=0.1) 

        ax[i][j].tick_params(axis='both', which='major', labelsize=4)
        ax[i][j].tick_params(axis='both', which='minor', labelsize=4)
        ax[i][j].set_xlim(0,900)

for i,beta in enumerate(betas):
    ax[i][0].set_ylabel(beta)

for j,flow in enumerate(flows):
    ax[-1][j].set_xlabel(round_to_first_significant(flow))

fig.supylabel("Трансмиссивность")
fig.supxlabel('Транспортный поток (доля населения в день)')
# plt.xlim(-5,10)
plt.tight_layout()
plt.savefig('graphs/diffsatellites_epids_lines.png', dpi=600)
plt.savefig('graphs/diffsatellites_epids_lines.pdf')

print(np.logspace(0,-4,9))
print(np.load(f'pkls/diffsatellites_{import_pattern}_1.0_1.0.npy')[:,:,:,:].shape)

A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(1, 2, figsize=(A4_WIDTH, A4_HEIGHT), sharey=True)

days = 80

data = np.load(f'pkls/diffsatellites_1_1.0_0.1.npy')[:,:,0,:days]

corr = signal.correlate(data[0,1,:], data[0,2,:])
corr /= np.max(corr)
lags = signal.correlation_lags(len(data[0,1,:]), len(data[0,2,:]))
print(lags[np.argmax(corr)])

ax[0].plot(lags, corr)

corr = signal.correlate(np.mean(data[:,1,:], axis=0), np.mean(data[:,2,:], axis=0))
corr /= np.max(corr)
lags = signal.correlation_lags(len(np.mean(data[:,1,:], axis=0)), len(np.mean(data[:,2,:], axis=0)))
print(lags[np.argmax(corr)])

ax[1].plot(lags, corr)

plt.savefig('smth.png')