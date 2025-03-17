import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def filter_fliers(data):
    return np.array([row for row in data if np.all(np.max(row[:,0,:], axis=1) > 100)])

cities_count = 5
colors = plt.cm.plasma(np.linspace(0, 1, cities_count))

for start in [0, 1]:
    data = np.load(f'pkls/{start}.npy')

    means = np.mean(np.sum(data[:,:,0,:], axis=1), axis=0)
    stds = np.std(np.sum(data[:,:,0,:], axis=1), axis=0)

    fig, ax = plt.subplots()

    ax.plot(np.arange(len(means)), means, color=colors[0])
    ax.fill_between(np.arange(len(means)), means-stds, means+stds, alpha=0.2, color=colors[0])

    ax.set_xlim(5,95)
    ax.set_ylabel('Число новых инфицирований')
    ax.set_xlabel('День')
    plt.savefig(f'graphs/epid{start}.pdf')

scales = [6, 1, 1, 1, 1]
dt = [np.load(f'pkls/{i}.npy') for i in range(2)]

infs = {'Город начала': [], 'Город': [], 'Пик зараженных': [], 'Пик критических': [], 
        'Пик мертвых': [], 'День пика инфицирований': []}

for i in range(2*cities_count):
    for seed in range(30):
        infs['Город начала'].append(i//cities_count)
        infs['Город'].append(i%cities_count)
        # print(np.max(dt[i//cities_count][:,i%cities_count,0,:], axis=1))
        infs['Пик зараженных'].append(np.max(dt[i//cities_count][:,i%cities_count,0,:], axis=1)[seed]/scales[i%cities_count])
        infs['Пик критических'].append(np.max(dt[i//cities_count][:,i%cities_count,1,:], axis=1)[seed]/scales[i%cities_count])
        infs['Пик мертвых'].append(np.max(dt[i//cities_count][:,i%cities_count,2,:], axis=1)[seed]/scales[i%cities_count])
        infs['День пика инфицирований'].append(np.argmax(dt[i//cities_count][:,i%cities_count,0,:], axis=1)[seed])


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(2, 2, figsize=(A4_WIDTH, A4_HEIGHT), sharex=True)
   
sns.boxplot(data=pd.DataFrame(infs), y='Пик зараженных', x='Город', ax=ax[0][0], hue='Город начала', palette='plasma', linewidth=.75)
sns.boxplot(data=pd.DataFrame(infs), y='Пик критических', x='Город', ax=ax[0][1], hue='Город начала', palette='plasma', linewidth=.75)
sns.boxplot(data=pd.DataFrame(infs), y='Пик мертвых', x='Город', ax=ax[1][0], hue='Город начала', palette='plasma', linewidth=.75)
sns.boxplot(data=pd.DataFrame(infs), y='День пика инфицирований', x='Город', ax=ax[1][1], hue='Город начала', palette='plasma', linewidth=.75)

ax[0][1].legend_.remove() 
ax[1][0].legend_.remove() 
ax[1][1].legend_.remove() 

plt.savefig('graphs/boxs.pdf')

labels = ['Все потоки уменьшены в 10 раз', 'Все потоки уменьшены в 100 раз', 'Потоки крупного города уменьшены в 10 раз', 'Потоки крупного города уменьшены в 100 раз']
y_labels = ['Пик зараженных', 'Пик критических', 'Пик мертвых', 'День пика инфицирований']
colors = plt.cm.plasma(np.linspace(0, 1, len(labels)+1))

for start in [0, 1]:
    A4_WIDTH = 8.27  # Ширина A4
    A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
    fig, ax = plt.subplots(2, 2, figsize=(A4_WIDTH, A4_HEIGHT), sharex=True)

    for i in range(4):
        experiment_id = 0
        data = filter_fliers(np.load(f'pkls/{start}.npy'))
        if i < 3:
            infs = np.max(np.sum(data[:,:,i,:], axis=1), axis=1)
        else:
            infs = np.argmax(np.sum(data[:,:,0,:], axis=1), axis=1)
        ax[i//2][i%2].axhline(np.mean(infs), color=colors[-1])
        ax[i//2][i%2].fill_between(np.linspace(0,100,100),np.ones(100)*(np.mean(infs)-np.std(infs)),np.ones(100)*(np.mean(infs)+np.std(infs)), alpha=0.3, color=colors[-1])

        for all in [True, False]:
            for multiplyer in [0.01, 0.1]:
                for start_day in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                    if all:
                        data = filter_fliers(np.load(f'pkls/{start}_all_{start_day}_{multiplyer}.npy'))
                    else:
                        data = filter_fliers(np.load(f'pkls/{start}_{start_day}_{multiplyer}.npy'))
                    if i < 3:
                        infs = np.max(np.sum(data[:,:,i,:], axis=1), axis=1)
                    else:
                        infs = np.argmax(np.sum(data[:,:,0,:], axis=1), axis=1)

                    if start_day == 10:
                        ax[i//2][i%2].errorbar(start_day + np.linspace(-1.5,1.5,4)[experiment_id], np.mean(infs), yerr=np.std(infs), capsize=5, fmt='.', color=colors[experiment_id], label=labels[experiment_id])
                    else:
                        ax[i//2][i%2].errorbar(start_day + np.linspace(-1.5,1.5,4)[experiment_id], np.mean(infs), yerr=np.std(infs), capsize=5, fmt='.', color=colors[experiment_id])
                    # sns.kdeplot(infs, label=f'{start_day}_{multiplyer}')
                experiment_id += 1

        
        ax[i//2][i%2].set_xlim(5,95)
        ax[i//2][i%2].set_xlabel('День введения мер')
        ax[i//2][i%2].set_ylabel(y_labels[i])

    ax[1][0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'graphs/hists{start}.pdf')