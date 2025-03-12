import matplotlib.pyplot as plt
import numpy as np

betas = np.linspace(1,0.5,6)
flows = np.logspace(-2,-5,7)
days = np.array([[150, 150, 150, 200, 200, 200, 200],
        [150, 150, 150, 200, 200, 300, 300],
        [200, 200, 200, 250, 250, 400, 400],
        [200, 200, 200, 300, 300, 350, 350],
        [300, 300, 300, 300, 300, 300, 300],
        [400, 400, 400, 450, 450, 450, 450]])

A4_WIDTH = 8.27  # Ширина A4
A4_HEIGHT = 11.69/2.5  # Высота A4 (можно уменьшить, если нужно)
fig, ax = plt.subplots(len(betas), len(flows), figsize=(A4_WIDTH, A4_HEIGHT), sharex=True)
   

for i,beta in enumerate(betas):
    for j,flow in enumerate(flows):
        data = np.load(f'pkls/basicflows_{beta}_{flow}.npy')

        ax[i][j].plot(np.arange(data.shape[-1]), np.mean(data[:,0,0,:], axis=0))
        ax[i][j].plot(np.arange(data.shape[-1]), np.mean(data[:,1,0,:], axis=0))
        # print(np.mean(data[:,:,0,:], axis=2))
        print(np.mean(data[:,1,0,:], axis=0))
plt.show()