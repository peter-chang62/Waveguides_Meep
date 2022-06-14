import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import scipy.integrate as spi

clipboard_and_style_sheet.style_sheet()


def get_band(x, num_band):
    return x[num_band::8]


def plot_mode(z, band, k_index):
    plt.figure()
    plt.imshow(get_band(z, band)[k_index][::-1, ::-1].T)
    return get_band(z, band)[k_index]


x = np.load('sim_output/E.npy')
x = x.__abs__() ** 2
z = x[:, :, :, 2]
z = z.reshape((12, 8, *z.shape[1:]))

width = np.round(15 * 40)
height = np.round(0.7 * 40)
edge_side = int(np.round((30 * 40 - 15 * 40) / 2))
edge_top = int(np.round((10 * 40 - .7 * 40) / 2))

mode = get_band(z, 2)[3]

lb = np.fft.fftshift(mode.T, 0)[-int(np.round(height * 0.75))][edge_side:-edge_top]
lt = np.fft.fftshift(mode.T, 0)[int(np.round(height * 0.75))][edge_side:-edge_top]
