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
flows = np.logspace(-2,-5,7)[::-1]
days = np.array([[150, 150, 150, 200, 200, 200, 200],
        [150, 150, 150, 200, 200, 300, 300],
        [200, 200, 200, 250, 250, 400, 600],
        [200, 200, 300, 300, 300, 350, 600],
        [300, 300, 300, 300, 300, 600, 600],
        [500, 500, 400, 450, 450, 900, 900]])

cities_count = 2

colors = plt.cm.plasma(np.linspace(0, 0.8, cities_count))


def filter_fliers(data):
    return np.array([row for row in data if np.all(np.max(row, axis=1) > 100)])


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT), sharex=True, sharey=True)
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = np.load(f'pkls/basicflows_{beta}_{flow}.npy')[:,:,0,:]
        # print(beta, flow, data.shape)
        data = filter_fliers(data)
        # print(beta, flow, data.shape)
        ax[i][j].plot(np.arange(data.shape[-1]), np.median(data[:,0,:], axis=0), color=colors[0])
        ax[i][j].fill_between(np.arange(data.shape[-1]), np.median(data[:,0,:], axis=0)-np.std(data[:,0,:], axis=0), np.median(data[:,0,:], axis=0)+np.std(data[:,0,:], axis=0), alpha=0.2, color=colors[0])
        ax[i][j].plot(np.arange(data.shape[-1]), np.median(data[:,1,:], axis=0), color=colors[1])
        ax[i][j].fill_between(np.arange(data.shape[-1]), np.median(data[:,1,:], axis=0)-np.std(data[:,1,:], axis=0), np.median(data[:,1,:], axis=0)+np.std(data[:,1,:], axis=0), alpha=0.2, color=colors[1])
        ax[i][j].tick_params(axis='both', which='major', labelsize=4)
        ax[i][j].tick_params(axis='both', which='minor', labelsize=4)
        ax[i][j].set_xlim(0,500)

plt.tight_layout()
plt.savefig('graphs/flows_epids.png', dpi=600)
plt.savefig('graphs/flows_epids.pdf')


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT), sharex=True, sharey=True)
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = np.load(f'pkls/basicflows_{beta}_{flow}.npy')[:,:,0,:]
        # print(beta, flow, data.shape)
        data = filter_fliers(data)
        # print(beta, flow, data.shape)
        for row in data[:,1,:]:
            ax[i][j].plot(np.arange(data.shape[-1]), row, color=colors[1], linewidth=1, alpha=0.01)
        for row in data[:,0,:]:
            ax[i][j].plot(np.arange(data.shape[-1]), row, color=colors[0], linewidth=1, alpha=0.01)
        

        ax[i][j].tick_params(axis='both', which='major', labelsize=4)
        ax[i][j].tick_params(axis='both', which='minor', labelsize=4)
        ax[i][j].set_xlim(0,500)

for i,beta in enumerate(betas):
    ax[i][0].set_ylabel(beta)

for j,flow in enumerate(flows):
    ax[-1][j].set_xlabel(round_to_first_significant(flow))

fig.supylabel("Трансмиссивность")
fig.supxlabel('Транспортный поток (доля населения в день)')
# plt.xlim(-5,10)
plt.tight_layout()
plt.savefig('graphs/flows_epids_lines.png', dpi=600)
plt.savefig('graphs/flows_epids_lines.pdf')



A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT))
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = filter_fliers(np.load(f'pkls/basicflows_{beta}_{flow}.npy')[:,:,0,:])

        ax[i][j].hist(np.max(data[:,0,:], axis=1), alpha=0.5, color=colors[0], bins=30)
        ax[i][j].hist(np.max(data[:,1,:], axis=1), alpha=0.5, color=colors[1], bins=30)
        ax[i][j].tick_params(axis='both', which='major', labelsize=4)
        ax[i][j].tick_params(axis='both', which='minor', labelsize=4)

for i,beta in enumerate(betas):
    ax[i][0].set_ylabel(beta)

for j,flow in enumerate(flows):
    ax[-1][j].set_xlabel(round_to_first_significant(flow))

fig.supylabel("Трансмиссивность")
fig.supxlabel('Транспортный поток (доля населения в день)')
plt.tight_layout()
plt.savefig('graphs/flows_hists_infs.png', dpi=600)
plt.savefig('graphs/flows_hists_infs.pdf')


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT))
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = filter_fliers(np.load(f'pkls/basicflows_{beta}_{flow}.npy')[:,:,0,:])

        ax[i][j].hist(np.argmax(data[:,0,:], axis=1), alpha=0.5, color=colors[0], bins=30)
        ax[i][j].hist(np.argmax(data[:,1,:], axis=1), alpha=0.5, color=colors[1], bins=30)
        ax[i][j].tick_params(axis='both', which='major', labelsize=4)
        ax[i][j].tick_params(axis='both', which='minor', labelsize=4)

for i,beta in enumerate(betas):
    ax[i][0].set_ylabel(beta)

for j,flow in enumerate(flows):
    ax[-1][j].set_xlabel(round_to_first_significant(flow))

fig.supylabel("Трансмиссивность")
fig.supxlabel("Транспортный поток (доля населения в день)")
plt.tight_layout()
plt.savefig('graphs/flows_hists_day.png', dpi=600)
plt.savefig('graphs/flows_hists_day.pdf')



A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT), sharex=True, sharey=True)
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = np.load(f'pkls/basicflows_{beta}_{flow}.npy')

        infs1, infs2 = np.max(data[:,0,0,:], axis=1), np.max(data[:,1,0,:], axis=1)

        c = 0
        for k in range(len(infs1)):
            if infs1[k] < 100 or infs2[k] < 100:
                ax[i][j].plot(np.arange(data.shape[-1]), data[k,0,0,:], color=colors[0])
                ax[i][j].plot(np.arange(data.shape[-1]), data[k,1,0,:], color=colors[1])
                c+=1
        # print(beta,flow,c/150)

plt.tight_layout()
plt.savefig('graphs/flows_strange_epids.png', dpi=600)
plt.savefig('graphs/flows_strange_epids.pdf')




t_stats_day = np.zeros((len(betas), len(flows)))
t_stats_max = np.zeros((len(betas), len(flows)))
mean_delta_day = np.zeros((len(betas), len(flows)))

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = filter_fliers(np.load(f'pkls/basicflows_{beta}_{flow}.npy')[:,:,0,:])

        t_stat_day, p_value_day = sps.ttest_ind(np.argmax(data[:,1,:], axis=1), np.argmax(data[:,0,:], axis=1), equal_var=False)
        t_stats_day[i,j] = t_stat_day

        t_stat_max, p_value_max = sps.ttest_ind(np.max(data[:,1,:], axis=1), np.max(data[:,0,:], axis=1), equal_var=False)
        t_stats_max[i,j] = t_stat_max

        mean_delta_day[i,j] = np.mean(np.argmax(data[:,1,:], axis=1) - np.argmax(data[:,0,:], axis=1))


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 8.27/2  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(1, 2, figsize=(A4_WIDTH, A4_HEIGHT), sharey=True)

sns.heatmap(t_stats_day, annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
            cmap="plasma", cbar_kws={'label': 't-статистика'}, ax=ax[1])

sns.heatmap(t_stats_max, annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
            cmap="plasma", cbar_kws={'label': 't-статистика'}, ax=ax[0])

ax[1].set_xlabel("Транспортный поток (доля населения в день)")
ax[0].set_ylabel("Трансмиссивность")
ax[1].set_title("t-статистики дня пика")
ax[0].set_xlabel("Транспортный поток (доля населения в день)")
ax[0].set_title("t-статистики пика инфицирований")

plt.tight_layout()
plt.savefig('graphs/flows_heatmap_thesis.png', dpi=600)
plt.savefig('graphs/flows_heatmap_thesis.pdf')


A4_WIDTH = 1+8.27  # Ширина A4
A4_HEIGHT = 8.27/2  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(1, 2, figsize=(A4_WIDTH, A4_HEIGHT))

sns.heatmap(t_stats_day, annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
            cmap="plasma", cbar_kws={'label': 't-статистика'}, ax=ax[1])

sns.heatmap(mean_delta_day, annot=True, fmt=".1f", xticklabels=[round_to_first_significant(flow) for flow in flows], yticklabels=betas,
            cmap="plasma", cbar_kws={'label': r'$\overline{\Delta t}$'}, ax=ax[0])

ax[1].set_xlabel("Транспортный поток (доля населения в день)")
ax[1].set_ylabel("Трансмиссивность")
ax[1].set_title("t-статистики дня пика")
ax[0].set_xlabel("Транспортный поток (доля населения в день)")
ax[0].set_ylabel("Трансмиссивность")
ax[0].set_title("Средний сдвиг пика")

plt.subplots_adjust(wspace=1, hspace=0)
plt.tight_layout()
plt.savefig('graphs/flows_heatmap_conference.png', dpi=600)
plt.savefig('graphs/flows_heatmap_conference.pdf')


A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 8.27/2  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(1, 2, figsize=(A4_WIDTH, A4_HEIGHT))

colors = plt.cm.plasma(np.linspace(0, 1, len(betas)))

for i,beta in enumerate(betas):
    ax[1].plot(flows, t_stats_day[i], 'o-', color=colors[i], label=f'{beta}')
    ax[0].plot(flows, mean_delta_day[i], 'o', color=colors[i])

    # slope, intercept, r_value, p_value, std_err = sps.linregress(np.log10(flows[2:]), t_stats_day[i][2:])
    # ax[0].plot(flows, slope*np.log10(flows)+intercept, color=colors[i], linestyle='-', label=f'{beta}; $R^2 = {r_value**2:.3f}$')

    slope, intercept, r_value, p_value, std_err = sps.linregress(np.log10(flows), mean_delta_day[i])
    ax[0].plot(flows, slope*np.log10(flows)+intercept, color=colors[i], linestyle='-', label=f'{beta}; $R^2 = {r_value**2:.3f}$')



ax[1].legend(title='Трансмиссивность')
ax[1].set_xlabel('Транспортный поток (доля населения в день)')
ax[1].set_ylabel('t-статистики дня пика')
ax[1].set_xscale('log')

ax[0].legend(title='Трансмиссивность')
ax[0].set_xlabel('Транспортный поток (доля населения в день)')
ax[0].set_ylabel('Средний сдвиг пика')
ax[0].set_xscale('log')

plt.subplots_adjust(wspace=2, hspace=0)
plt.tight_layout()
plt.savefig('graphs/flows_lines.png', dpi=600)
plt.savefig('graphs/flows_lines.pdf')