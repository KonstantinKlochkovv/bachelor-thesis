import pickle
import numpy as np
import matplotlib.pyplot as plt

sizes = np.load('data/sizes.npy', allow_pickle=True)
labels = np.load('data/labels.npy', allow_pickle=True)
cities_count = len(labels)
colors = plt.cm.plasma(np.linspace(0, 1, cities_count))

for scale in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    fig, ax = plt.subplots()

    print(labels)

    for i, label in enumerate(labels):
        data = np.load(f'Results/pkls/{label}_{scale}.npy')
        print(data.shape)
        print(data[:,0,2,:])
        print(data[:,:,2,:].shape)
        # means = np.mean(np.sum(data[:,:,0,:], axis=1), axis=0)
        # stds = np.std(np.sum(data[:,:,0,:], axis=1), axis=0)
        means = np.mean(data[:,9,0,:], axis=0)
        stds = np.std(data[:,9,0,:], axis=0)

        ax.plot(np.arange(len(means)), means, color=colors[i], label=label)
        ax.fill_between(np.arange(len(means)), means-stds, means+stds, alpha=0.2, color=colors[i])

    ax.legend()
    plt.savefig(f'Results/res_{scale}.pdf')