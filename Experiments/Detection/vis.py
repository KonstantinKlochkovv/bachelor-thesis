import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.interpolate import make_smoothing_spline
from dtw import dtw

sizes = np.array([6,1,1,1,1])
lengths = np.array([[0, 1, 1, 1, 1],
                    [1, 0, 2**0.5, 2**0.5, 2],
                    [1, 2**0.5, 0, 2, 2**0.5],
                    [1, 2**0.5, 2, 0, 2**0.5],
                    [1, 2, 2**0.5, 2**0.5, 0]])

time = 80
cities = 5
flow = 0.1
seed = 10

for import_pattern in [0,1]:
    meds, stds = {i: np.array([]) for i in range(cities) if i != import_pattern}, {i: np.array([]) for i in range(cities) if i != import_pattern}
    times = range(10,80,5)
    for time in times:
        data = np.load(f'pkls/diffsatellites_{import_pattern}_1.0_{flow}.npy')[seed,:,0,:time]
        data[import_pattern,0] = 0

        for i, curve in enumerate(data):
            if i == import_pattern:
                continue
            
            alignment = dtw(curve/sizes[i], data[import_pattern,:]/sizes[import_pattern], keep_internals=True)
            shifts = alignment.index2 - alignment.index1
            meds[i] = np.append(meds[i], np.median(shifts))
            stds[i] = np.append(stds[i], np.std(shifts))
    
    fig, ax = plt.subplots()
    for key in meds.keys():
        ax.plot(times, meds[key], label=key)
        ax.fill_between(times, meds[key]-stds[key], meds[key]+stds[key], alpha=0.1)
    ax.legend()
    plt.savefig(f'graphs/{import_pattern}.png')


for import_pattern in [0,1]:
    meds, stds = {i: np.array([]) for i in range(cities) if i != import_pattern}, {i: np.array([]) for i in range(cities) if i != import_pattern}
    times = range(10,80,5)
    for time in times:
        data = np.load(f'pkls/diffsatellites_{import_pattern}_1.0_{flow}.npy')[seed,:,0,:time]
        data[import_pattern,0] = 0

        for i, curve in enumerate(data):
            if i == import_pattern:
                continue
            
            spl_curve = make_smoothing_spline(np.arange(len(curve)), curve/sizes[i], lam=10)
            spl_ref = make_smoothing_spline(np.arange(len(data[import_pattern,:])), data[import_pattern,:]/sizes[import_pattern], lam=10)

            alignment = dtw(spl_curve(np.arange(len(curve))), spl_ref(np.arange(len(data[import_pattern,:]))), keep_internals=True)
            shifts = alignment.index2 - alignment.index1
            meds[i] = np.append(meds[i], np.median(shifts))
            stds[i] = np.append(stds[i], np.std(shifts))
    
    fig, ax = plt.subplots()
    for key in meds.keys():
        ax.plot(times, meds[key], label=key)
        ax.fill_between(times, meds[key]-stds[key], meds[key]+stds[key], alpha=0.1)
    ax.legend()
    plt.savefig(f'graphs/smooth_{import_pattern}.png')


        
# print(data)
# fig, ax = plt.subplots()
# for i,curve in enumerate(data):
#     print(curve)
#     ax.plot(np.arange(len(curve)), curve/sizes[i], label=i)
# ax.legend()
# plt.savefig('graphs/epid.png')

for import_pattern in [0,1]:
    data = np.load(f'pkls/diffsatellites_{import_pattern}_1.0_{flow}.npy')[seed,:,0,:time]
    data[import_pattern,0] = 0

    for i, curve in enumerate(data):
        if i == import_pattern:
            continue

        fig, ax = plt.subplots()
        alignment = dtw(curve/sizes[i], data[import_pattern,:]/sizes[import_pattern], keep_internals=True)
        alignment.plot(type="twoway", xlab="День", ylab="Нормированное число новых зараженных")
        # shifts = alignment.index2 - alignment.index1
        # plt.hist(shifts, bins=10, edgecolor="black")
        # plt.xlabel("Величина сдвига (индексы)")
        # plt.ylabel("Количество точек")
        # plt.title("Распределение сдвигов DTW")
        plt.savefig(f'graphs/epid_{import_pattern}_{i}.png')  

# for import_pattern in [0,1]:
#     for data in np.load(f'Experiments/Detection/pkls/diffsatellites_{import_pattern}_1.0_1.0.npy')[:,:,0,:time]:
#         print(data)
#         # data[import_pattern,0] = 0
#         # print(data)
#         df = np.array([[import_pattern,i,signal.correlation_lags(data.shape[1], data.shape[1])[np.argmax(signal.correlate(signal.medfilt(data[import_pattern,:], 5), signal.medfilt(data[i,:], 5)))], 
#                         np.log(sizes[import_pattern]) + np.log(sizes[i]), np.log(lengths[import_pattern,i])] 
#                         for i in range(data.shape[0]) if i != import_pattern and np.max(data[i,:]) > 5])
#         # df = np.array([[import_pattern,i,signal.correlation_lags(data.shape[1], data.shape[1])[np.argmax(signal.correlate(data[import_pattern,:], data[i,:]))], 
#         #                 np.log(sizes[import_pattern]) + np.log(sizes[i]), np.log(lengths[import_pattern,i])] 
#         #                 for i in range(data.shape[0]) if i != import_pattern and np.max(data[i,:]) > 5])
#         print(df)

#         size = 5
#         if np.min((df[:,2])) > 0:
#             fig, ax = plt.subplots(1,size)
#             for k,curve in enumerate(data):
#                 for i,lam in enumerate(np.linspace(1,10,size)):
#                     spl = make_smoothing_spline(np.arange(len(curve)),curve/np.max(curve),lam=lam)
#                     ax[i].plot(np.arange(len(curve)),spl(np.arange(len(curve)))/np.max(spl(np.arange(len(curve)))), label=k)
#             ax[0].legend()
#             plt.show()

#         # for j in range(data.shape[0]):
#         #     if j == import_pattern:
#         #         continue
            

#         # for i in range(data.shape[0]):
#         #     for j in range(i):
#         #         correlations = signal.correlate(signal.medfilt(data[i,:], 5), signal.medfilt(data[j,:], 5))
#         #         correlations /= np.max(correlations)
#         #         lags = signal.correlation_lags(time, time)


