# %% package imports
import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.interpolate as spi
import meep as mp
from TFW_meep import materials as mtp
from TFW_meep import waveguide_dispersion as wg
import pynlo_extras as pe


cr.style_sheet()


def loss(plot=False):
    u = np.load("unguided_dispersion.npy")
    g = np.load("guided_dispersion.npy")
    k_points = np.load("k_points.npy")
    u_index = np.load("unguided_index.npy")

    k_points = np.append(
        # only unguided -> re-index backgrounds -> truncate -> then re-index forward
        k_points[u_index][::-1][: len(u)][::-1],
        k_points[u_index[-1] + 1 :][: len(g)],
        axis=0,
    )
    data = np.vstack([u[::-1], g])

    # throw out bad simulation data points
    # data = np.append(data[6:16], data[18:], axis=0)
    # k_points = np.append(k_points[6:16], k_points[18:], axis=0)

    um = 1e-6
    v_grid = k_points[:, 0] * sc.c / um
    w_grid = v_grid * 2 * np.pi
    b = k_points[:, 1] * 2 * np.pi / um
    b_1 = np.gradient(b, w_grid, edge_order=2)
    loss = data[:, 1] * sc.c / um * b_1

    if plot:
        plt.figure()
        plt.plot(1 / data[:, 0], loss, "o")

    gridded = spi.interp1d(v_grid, loss, bounds_error=False, fill_value=(1e20, 0))
    return gridded


# %%
wl_min = 450e-9
wl_max = 4.18e-6

v_min = sc.c / wl_max
v_max = sc.c / wl_min

e_p = 300e-12
t_fwhm = 50e-15
time_window = 10e-12

pulse = pe.light.Pulse.Sech(
    2**11,
    v_min,
    v_max,
    sc.c / 1550e-9,
    e_p,
    t_fwhm,
    time_window,
)


etch_width = 1.65
etch_depth = 0.7
film_thickness = 1.0
TFW = wg.ThinFilmWaveguide(
    etch_width,
    etch_depth,
    film_thickness,
    mtp.Al2O3,
    mp.Medium(index=1),
    resolution=30,
    cell_width=8,
    cell_height=6,
    num_bands=1,
)


ppln = pe.materials.PPLN()


def eps_func_wvgd(freq):
    um = 1e-6

    nu = freq * sc.c / um
    return ppln.n(nu) ** 2


res = TFW.calc_dispersion(
    wl_min=0.4,
    wl_max=5,
    NPTS=100,
    freq_array=None,
    eps_func_wvgd=eps_func_wvgd,
    eps_func_sbstrt=None,
)

um = 1e-6
v_grid = res.freq * sc.c / um
b = 2 * np.pi * res.kx.flatten() / um
beta = spi.interp1d(v_grid, b)(pulse.v_grid)

length = 5e-3
a_eff = 1e-12
model = ppln.generate_model(
    pulse,
    a_eff=a_eff,
    length=length,
    polling_period=None,
    polling_sign_callable=None,
    beta=beta,
    alpha=-loss()(pulse.v_grid),
    is_gaussian_beam=False,
)

sim = model.simulate(
    length,
    dz=pe.utilities.estimate_step_size(model, local_error=1e-6),
    local_error=1e-6,
    n_records=100,
    plot=None,
)
