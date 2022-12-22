"""sim data analysis the arrays were saved as np.c_[res.freq, beta, beta1, beta2]

This script runs all the dispersion curves through PyNLO"""

import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as sc
import scipy.integrate as scint
import os
from scipy.interpolate import interp1d
import pynlo_connor as pynlo
from pynlo_connor import utility as utils

try:
    import materials as mtp

    on_linux = True
except ImportError:
    Al2O3 = np.load('convenience/freq_epsilon_data.npy')
    Al2O3 = InterpolatedUnivariateSpline(Al2O3[:, 0], Al2O3[:, 1])
    on_linux = False

clipboard_and_style_sheet.style_sheet()


def width(s):
    return float(s.split('_')[0])


def depth(s):
    return float(s.split('_')[1].split('.npy')[0])


def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = scint.simpson(scint.simpson(I)) ** 2 / scint.simpson(scint.simpson(I ** 2))
    area /= resolution ** 2
    return area


# %%____________________________________________________________________________________________________________________
resolution = 30
conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9  # already multiplied for some sim data (should be obvious)
path_wvgd = 'sim_output/07-19-2022/'
names_wvgd = [i.name for i in os.scandir(path_wvgd + 'dispersion-curves/')]

width_limit = 1.245
[names_wvgd.remove(i) for i in names_wvgd.copy() if width(i) < width_limit]

names_wvgd = sorted(names_wvgd, key=depth)
names_wvgd = sorted(names_wvgd, key=width)

# %%____________________________________________________________________________________________________________________


def get_disp(n): return np.load(path_wvgd + 'dispersion-curves/' + names_wvgd[n])
def get_eps(n): return np.load(path_wvgd + 'eps/' + names_wvgd[n])
def get_field(n): return np.squeeze(np.load(path_wvgd + 'E-fields/' + names_wvgd[n]))


# %%____________________________________________________________________________________________________________________
def plot_eps(n, ax=None):
    if ax is not None:
        ax.imshow(get_eps(n)[::-1, ::-1].T, interpolation='spline36', cmap='binary')
    else:
        plt.imshow(get_eps(n)[::-1, ::-1].T, interpolation='spline36', cmap='binary')


def plot_field(n, k_index, alpha=0.9, ax=None):
    if ax is not None:
        ax.imshow(get_field(n)[k_index][::-1, ::-1].T, cmap='RdBu', alpha=alpha)
    else:
        plt.imshow(get_field(n)[k_index][::-1, ::-1].T, cmap='RdBu', alpha=alpha)


def plot_mode(n, k_index, new_figure=True, ax=None):
    assert not np.all([new_figure, ax is not None])
    if new_figure:
        plt.figure()
    plot_eps(n, ax)
    plot_field(n, k_index, 0.9, ax)
    data = get_disp(n)  # np.c_[res.freq, beta, beta1, beta2]
    freq = data[:, 0]
    wl = 1 / freq[k_index]
    if ax is not None:
        ax.set_title(f'{np.round(width(names_wvgd[n]), 3)} x {np.round(depth(names_wvgd[n]), 3)} x 1'
                     + ' $\\mathrm{\\mu m}$' '\n' +
                     "$\\mathrm{\\lambda = }$" + '%.2f' % wl
                     + ' $\\mathrm{\\mu m}$' + '\n' +
                     '$\\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
                     ' $\\mathrm{\\mu m^2}$')
        ax.axis(False)
    else:
        plt.title(f'{np.round(width(names_wvgd[n]), 3)} x {np.round(depth(names_wvgd[n]), 3)} x 1'
                  + ' $\\mathrm{\\mu m}$' '\n' +
                  "$\\mathrm{\\lambda = }$" + '%.2f' % wl
                  + ' $\\mathrm{\\mu m}$' + '\n' +
                  '$\\mathrm{A_{eff}}$ = %.3f' % mode_area(get_field(n)[k_index]) +
                  ' $\\mathrm{\\mu m^2}$')
        plt.axis(False)


def get_bp_ind(wl_grid, wl_ll, wl_ul):
    return np.where(np.logical_and(wl_grid >= wl_ll, wl_grid <= wl_ul), 1, 0)


def e_p_in_window(wl_grid, dv, a_v, wl_ll, wl_ul):
    h = get_bp_ind(wl_grid, wl_ll, wl_ul)
    a_v_filt = a_v * h
    return scint.simps(abs(a_v_filt) ** 2, axis=1, dx=dv)


def instantiate_pulse(n_points, v_min, v_max, e_p=300e-3 * 1e-9, t_fwhm=50e-15):
    v0 = sc.c / 1560e-9  # sc.c / 1550 nm
    pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
    pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing
    return pulse


def load_waveguide(pulse, n_sim):
    pulse: pynlo.light.Pulse
    v_grid = pulse.v_grid
    v0 = pulse.v0

    b_data = get_disp(n_sim)
    wl, b = 1 / b_data[:, 0], b_data[:, 1]
    k = b * 1e6 / (2 * np.pi)  # 1/m
    nu = sc.c / (wl * 1e-6)
    n = sc.c * k / nu
    n_wvgd = interp1d(nu, n, kind='cubic', bounds_error=True)

    n_eff = n_wvgd(v_grid)
    beta = n_eff * 2 * np.pi * v_grid / sc.c  # n * w / sc.c

    ind_center_wl = np.argmin(abs(wl - 1.55))
    a_eff = mode_area(get_field(n_sim)[ind_center_wl]) * 1e-12  # um^2 -> m^2 @ lamda = 1560 nm

    # 2nd order nonlinearity
    d_eff = 27e-12  # 27 pm / V
    chi2_eff = 2 * d_eff
    g2 = utils.chi2.g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff)

    # 3rd order nonlinearity
    chi3_eff = 5200e-24
    g3 = utils.chi3.g3_spm(n_eff, a_eff, chi3_eff)

    mode = pynlo.media.Mode(
        v_grid=v_grid,
        beta_v=beta,
        g2_v=g2,
        g2_inv=None,
        g3_v=g3,
        z=0.0
    )
    return mode


def simulate(pulse, mode, length=3e-3, npts=100):
    model = pynlo.model.SM_UPE(pulse, mode)
    local_error = 1e-6
    dz = model.estimate_step_size(n=20, local_error=local_error)

    z_grid = np.linspace(0, length, npts)
    pulse_out, z, a_t, a_v = model.simulate(z_grid, dz=dz, local_error=local_error, n_records=None, plot=None)
    return pulse_out, z, a_t, a_v


def aliasing_av(a_v):
    x = abs(a_v) ** 2
    x /= x.max()
    x_v_min = x[:, 0]
    x_v_max = x[:, -1]

    ind_v_min_aliasing = np.where(x_v_min > 1e-3)[0]
    ind_v_max_aliasing = np.where(x_v_max > 1e-3)[0]
    List = []
    side = []
    if len(ind_v_min_aliasing) > 0:
        List.append(ind_v_min_aliasing[0])
        side.append("long")
    if len(ind_v_max_aliasing) > 0:
        List.append(ind_v_max_aliasing[0])
        side.append("short")
    if len(List) > 0:
        ind = np.argmin(List)
        return List[ind], side[ind]
    else:
        return False


# ______________________________________________________________________________________________________________________
ind_pwr_3_5_300pJ = np.array([13, 14, 15, 16, 17, 18, 31, 32, 33, 34, 35, 36, 47,
                              48, 49, 50, 51, 52, 53, 64, 65, 66, 67, 68, 69, 70,
                              71, 82, 83, 84, 85, 86, 87, 88, 89, 100, 101, 102, 105,
                              119, 120, 121, 138, 139])
ind_pwr_3_5_100pJ = np.array([13, 30, 48, 65, 83, ])

# ______________________________________________________________________________________________________________________
# h = 1
# N = np.arange(len(names_wvgd))
# center = len(N) // 4
# chunks = N[:center], N[center:center * 2], N[center * 2:center * 3], N[center * 3:]  # launch 4 consoles
# for n in chunks[3]:
#     n_points = 2 ** 13
#     v_min = sc.c / ((5000 - 10) * 1e-9)  # sc.c / 5000 nm
#     v_max = sc.c / ((400 + 10) * 1e-9)  # sc.c / 400 nm
#     e_p = 100e-12
#     t_fwhm = 50e-15
#     pulse = instantiate_pulse(n_points=n_points,
#                               v_min=v_min,
#                               v_max=v_max,
#                               e_p=e_p,
#                               t_fwhm=t_fwhm)
#
#     mode = load_waveguide(pulse, n)
#     pulse_out, z, a_t, a_v = simulate(pulse, mode, length=20e-3, npts=250)
#     p_v_dB = abs(a_v) ** 2
#     p_v_dB /= p_v_dB.max()
#     p_v_dB = 10 * np.log10(p_v_dB)
#     wl = sc.c / pulse.v_grid
#     ind_alias = aliasing_av(a_v)
#
#     np.save(f"sim_output/12-20-2022/a_v/{names_wvgd[n]}", a_v)
#     np.save(f"sim_output/12-20-2022/a_t/{names_wvgd[n]}", a_t)
#
#     print(len(N) // center - h)
#     h += 1
#
# np.save("sim_output/12-20-2022/v_grid.npy", pulse.v_grid)
# np.save("sim_output/12-20-2022/t.npy", pulse.t_grid)
# np.save("sim_output/12-20-2022/z.npy", z)

# ______________________________________________________________________________________________________________________
names_spm = [i.name for i in os.scandir('sim_output/12-20-2022/a_v/')]
names_spm = sorted(names_spm, key=depth)
names_spm = sorted(names_spm, key=width)


def load_a_v(n): return np.load(f'sim_output/12-20-2022/a_v/{names_spm[n]}')
def load_a_t(n): return np.load(f'sim_output/12-20-2022/a_t/{names_spm[n]}')


v_grid = np.load('sim_output/12-20-2022/v_grid.npy')
wl = sc.c / v_grid
t = np.load('sim_output/12-20-2022/t.npy')
z = np.load('sim_output/12-20-2022/z.npy')


def plot2D(n, fig=None, ax=None, a_v_only=False):
    a_v = load_a_v(n)
    a_t = load_a_t(n)

    p_v_dB = abs(a_v) ** 2
    p_v_dB /= p_v_dB.max()
    p_v_dB = 10 * np.log10(p_v_dB)

    p_t_dB = abs(a_t) ** 2
    p_t_dB /= p_t_dB.max()
    p_t_dB = 10 * np.log10(p_t_dB)
    if fig is None:
        if not a_v_only:
            fig, ax = plt.subplots(1, 2, figsize=np.array([8.66, 4.8]))
        else:
            fig, ax = plt.subplots(1, 1)
    else:
        assert ax is not None
    if not a_v_only:
        ax[0].pcolormesh(wl * 1e6, z * 1e3, p_v_dB, vmin=-40, vmax=0)
        ax[0].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
        ax[0].set_ylabel("propagation distance (mm)")

        ax[1].pcolormesh(t * 1e15, z * 1e3, p_t_dB, vmin=-40, vmax=0)
        ax[1].set_xlim(-500, 500)
        ax[1].set_xlabel("t (fs)")
        ax[1].set_ylabel("propagation distance (mm)")
    else:
        ax.pcolormesh(wl * 1e6, z * 1e3, p_v_dB, vmin=-40, vmax=0)
        ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
        ax.set_ylabel("propagation distance (mm)")
    return fig, ax


def plot_single(n, length, fig=None, ax=None, a_v_only=False, a_t_only=False):
    assert not np.all([a_v_only, a_t_only])
    a_v = load_a_v(n)
    a_t = load_a_t(n)

    ind = np.argmin(abs(z - length))
    a_v = a_v[ind]
    a_t = a_t[ind]

    p_v_dB = abs(a_v) ** 2
    p_v_dB /= p_v_dB.max()
    p_v_dB = 10 * np.log10(p_v_dB)

    wl_label = "wavelength ($\\mathrm{\\mu m}$)"
    t_label = "t (fs)"

    if fig is None:
        if a_v_only or a_t_only:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = plt.subplots(1, 2)
    else:
        assert ax is not None

    if a_v_only:
        ax.plot(wl * 1e6, p_v_dB)
        ax.set_ylim(-40, 0)
        ax.set_xlabel(wl_label)
        ax.set_ylabel("a.u. (dB)")
    elif a_t_only:
        ax.plot(t * 1e15, abs(a_t) ** 2 / max(abs(a_t) ** 2))
        ax.set_xlim(-100, 100)
        ax.set_xlabel(t_label)
        ax.set_ylabel("a.u.")
    else:
        ax[0].plot(wl * 1e6, p_v_dB)
        ax[0].set_ylim(-40, 0)
        ax[0].set_xlabel(wl_label)
        ax[0].set_ylabel("a.u. (dB)")
        ax[1].plot(t * 1e15, abs(a_t) ** 2 / max(abs(a_t) ** 2))
        ax[1].set_xlim(-100, 100)
        ax[1].set_xlabel(t_label)
        ax[1].set_ylabel("a.u.")
    return fig, ax


def plot_all(n, k_index, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(2, 2, figsize=np.array([11.84, 7.94]))
    else:
        assert ax is not None
    ax = ax.flatten()
    plot2D(n, fig, ax)
    plot_mode(n, k_index, False, ax[2])
    freq, b, b1, b2 = get_disp(n).T
    ax[3].plot(1 / freq, b2 * conversion, 'o-')
    ax[3].grid(True)
    ax[3].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
    ax[3].set_ylabel("$\\mathrm{ps^2/km}$")
    return fig, ax


dv = np.diff(v_grid)[0]
power = np.asarray([e_p_in_window(wl, dv, load_a_v(i), 3e-6, 5e-6) for i in range(len(names_spm))])

# save = True
# for i in range(len(names_spm)):
#     if i == 0:
#         fig, ax = plot_all(i, 4)
#     else:
#         [i.clear() for i in ax]
#         plot_all(i, 4, fig, ax)
#     if save:
#         plt.savefig(f'fig/{i}.png')
#     else:
#         plt.pause(.05)


fig, ax = plot2D(ind_pwr_3_5_100pJ[0], a_v_only=True)
fig.dpi = 300
fig, ax = plt.subplots(1, 1, dpi=300)
plot_mode(ind_pwr_3_5_100pJ[0], 5, new_figure=False, ax=ax)

# for n in ind_pwr_3_5_100pJ:
#     fig, ax = plot2D(n, a_v_only=True)
#     fig.dpi = 300
#     fig, ax = plt.subplots(1, 1, dpi=300)
#     plot_mode(n, 5, new_figure=False, ax=ax)

fig, ax = plot_single(ind_pwr_3_5_100pJ[0], 3.1e-3, a_v_only=True)
fig.dpi = 300
ax.set_xlim(.6, 3.6)
fig, ax = plot_single(ind_pwr_3_5_100pJ[0], 3.1e-3, a_t_only=True)
fig.dpi = 300

# plot_single(ind_pwr_3_5_100pJ[1], 3.8e-3, fig, ax)
# plot_single(ind_pwr_3_5_100pJ[2], 4.6e-3, fig, ax)
# plot_single(ind_pwr_3_5_100pJ[3], 6.6e-3, fig, ax)
# plot_single(ind_pwr_3_5_100pJ[4], 8.4e-3, fig, ax)

plt.show()
