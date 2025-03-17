import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data = np.load('pkls/0_10_0.01.npy')

# print(data)

# print(np.mean(data, axis=0)[:,0,:])

means = np.mean(data, axis=0)[:,0,:]
stds = np.std(data, axis=0)[:,0,:]

fig, ax = plt.subplots()
for i in range(len(means)):
    ax.plot(np.arange(len(means[i])), means[i], label=i)
    ax.fill_between(np.arange(len(means[i])), means[i]-stds[i], means[i]+stds[i], alpha=0.2)
ax.legend()
ax.set_ylim(0,5000)
plt.savefig('graphs/pic1.pdf')


labels = ['Все потоки уменьшены в 10 раз', 'Все потоки уменьшены в 100 раз', 'Потоки сателлита уменьшены в 10 раз', 'Потоки сателлита уменьшены в 100 раз']
y_labels = ['Пик зараженных', 'Пик критических', 'Пик умерших', 'День пика инфицирований']
colors = plt.cm.plasma(np.linspace(0, 1, len(labels)+1))


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.3  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(2, 2, figsize=(A4_WIDTH, A4_HEIGHT), sharex=True)

for i in range(4):
    experiment_id = 0
    data = np.load(f'pkls/1.npy')
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
                    data = np.load(f'pkls/1_all_{start_day}_{multiplyer}.npy')
                else:
                    data = np.load(f'pkls/1_{start_day}_{multiplyer}.npy')
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

    # if i == 3:
    #     ax[i//2][i%2].legend(loc='lower right', bbox_to_anchor =(0.5,-0.97))
    # ax.set_ylim(30000,38000)
ax[1][0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('graphs/hists1.pdf')


# # print(np.cumsum(data, axis=2))
# # cummeans = np.mean(np.cumsum(data, axis=2), axis=0)
# # stds = np.std(np.cumsum(data, axis=2), axis=0)
# # fig, ax = plt.subplots()
# # for i in range(len(cummeans)):
# #     ax.plot(np.arange(len(cummeans[i])), cummeans[i], label=i)
# #     ax.fill_between(np.arange(len(cummeans[i])), cummeans[i]-stds[i], cummeans[i]+stds[i], alpha=0.2)
# # ax.legend()
# # plt.show()

# # data = np.array([np.load(f'4 cities/pkls/res_{i}.npy') for i in range(50)])
# print(np.max(data[:,1,:], axis=1))
# print(np.argmax(data[:,1,:], axis=1))
# print(np.sum(data[:,1,:], axis=1))