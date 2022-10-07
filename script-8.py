import os
import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()


def width(s):
    return float(s.split('_')[0])


def depth(s):
    return float(s.split('_')[1].split('.npy')[0])


# %% ___________________________________________________________________________________________________________________
path = 'sim_output/10-05-2022/'
names = [i.name for i in os.scandir('sim_output/10-05-2022/')]
names.remove('z.npy')
names.remove('v_grid.npy')

names.sort(key=depth)
names.sort(key=width)

z = np.load(path + 'z.npy') * 1e3
v_grid = np.load(path + 'v_grid.npy') * 1e-12
wl_grid = sc.c * 1e6 * 1e-12 / v_grid

fig, ax = plt.subplots(1, 2, figsize=np.array([13.69, 4.8]))
for i in range(len(names)):
    data = np.load(path + names[i])
    data = abs(data) ** 2
    data /= data.max()

    [i.clear() for i in ax]

    ax[0].pcolormesh(wl_grid, z, data)
    ax[0].set_xlabel("wavelength ($\mathrm{\mu m}$)")

    ax[1].plot(wl_grid, 10 * np.log10(data[-1] / data[-1].max()))
    ax[1].set_xlabel("wavelength ($\mathrm{\mu m}$)")
    fig.suptitle(f'{i}')

    plt.pause(.01)
