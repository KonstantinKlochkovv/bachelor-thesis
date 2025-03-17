import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


cities_count = 5
colors = plt.cm.plasma(np.linspace(0, 1, cities_count))

def filter_fliers(data):
    return np.array([row for row in data if np.all(np.max(row, axis=1) > 100)])

data = np.load('pkls/1_all_20_0.01.npy')

print(data[:,:,:,:].shape)
for row in data:
    print(row.shape)
    # print(np.all(np.max(row, axis=1) > 100))

print(np.any([True, False]))

# days = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# A4_WIDTH = 8.27  # Ширина A4
# A4_HEIGHT = 11.69  # Высота A4 (можно уменьшить, если нужно)
# fig, ax = plt.subplots(len(days), 4, figsize=(A4_WIDTH, A4_HEIGHT), sharex=True)

# experiment_id = 0
# for all in [True, False]:
#     for multiplyer in [0.01, 0.1]:
#         for i, start_day in enumerate(days):
#             if all:
#                 data = filter_fliers(np.load(f'pkls/1_all_{start_day}_{multiplyer}.npy')[:,:,0,:])
#             else:
#                 data = filter_fliers(np.load(f'pkls/1_{start_day}_{multiplyer}.npy')[:,:,0,:])

#             for j in range(cities_count):
#                 for row in data[:,j,:]:
#                     ax[i][experiment_id].plot(np.arange(len(row)), row, color=colors[j], alpha=0.1)
        
#         experiment_id += 1

# plt.tight_layout()
# plt.savefig('graphs/lines.pdf')