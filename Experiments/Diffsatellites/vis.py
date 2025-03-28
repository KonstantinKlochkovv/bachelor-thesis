import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import seaborn as sns
import math

def round_to_first_significant(x):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))))

cities_count = 5
seed_count = 30
betas = np.linspace(1,0.5,6)
flows = np.logspace(0,-4,5)[::-1]
import_patterns = [0, 1]

correlations = np.zeros((len(import_patterns), cities_count, len(betas), len(flows)))

for import_pattern in import_patterns:
    for i,beta in enumerate(betas):
        for j,flow in enumerate(flows):
            data = np.load(f'pkls/diffsatellites_{import_pattern}_{beta}_{flow}.npy')[:,:,0,:]
            data_all = np.load(f'pkls/diffsatellites_all_{beta}_{flow}.npy')[:,:,0,:]
            print(data.shape)

            for city in range(cities_count):
                for seed in range(seed_count):
                    corr, p_value = sps.pearsonr(data[seed,city,:], data_all[seed,city,:])
                    if np.isnan(corr):
                        correlations[import_pattern, city, i, j] += 0
                    else:
                        correlations[import_pattern, city, i, j] += corr

            correlations[import_pattern, :, i, j] /= seed_count



A4_WIDTH = 2*8.27  # Ширина A4
A4_HEIGHT = 11.69/2  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(import_patterns), cities_count, figsize=(A4_WIDTH, A4_HEIGHT))

for import_pattern in import_patterns:
    for city in range(cities_count):
        sns.heatmap(correlations[import_pattern, city, :, :], annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
            cmap="plasma", cbar=False, ax=ax[import_pattern, city])
        
        ax[import_pattern, city].set_title(f'Город {city}')
        ax[import_pattern, city].set_ylabel('Трансмиссивность')
        ax[import_pattern, city].set_xlabel('Множитель потока')

ax[0,0].set_ylabel('Начало в 0 \n Трансмиссивность')
ax[1,0].set_ylabel('Начало в 1 \n Трансмиссивность')

plt.tight_layout()
plt.savefig('graphs/diffsatellites_correlations.png', dpi=600)
plt.savefig('graphs/diffsatellites_correlations.pdf')