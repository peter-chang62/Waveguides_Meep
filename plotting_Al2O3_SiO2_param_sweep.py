# %% -----
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import os
from TFW_meep import waveguide_dispersion as wg
from meep import materials as mt
from scipy.constants import c
import pynlo
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm
from TFW_meep import materials as mtp
import meep as mp


def load(h):
    path = "sim_output/02-23-2024/Al2O3/"
    names = [i.name for i in os.scandir(path)]
    names = [i for i in names if str(float(h)) + "_t" in i]
    if len(names) == 0:
        print("no file found")
        return
    else:
        assert len(names) == 1
        (f,) = names
        return np.load(path + f)


# ----- Gayer paper Sellmeier equation for ne (taken from PyNLO 1 / omega is in um
# n = squrt[epsilon] so epsilon = n^2
def eps_func_wvgd(omega):
    # omega is in inverse micron
    um = 1e-6
    v = omega * c / um
    return n_MgLN_G(v, T=24.5, axis="e") ** 2


def n_MgLN_G(v, T=24.5, axis="e"):
    """
    Range of Validity:
        - 500 nm to 4000 nm
        - 20 C to 200 C
        - 48.5 mol % Li
        - 5 mol % Mg

    Gayer, O., Sacks, Z., Galun, E. et al. Temperature and wavelength
    dependent refractive index equations for MgO-doped congruent and
    stoichiometric LiNbO3 . Appl. Phys. B 91, 343–348 (2008).

    https://doi.org/10.1007/s00340-008-2998-2

    """
    if axis == "e":
        a1 = 5.756  # plasmons in the far UV
        a2 = 0.0983  # weight of UV pole
        a3 = 0.2020  # pole in UV
        a4 = 189.32  # weight of IR pole
        a5 = 12.52  # pole in IR
        a6 = 1.32e-2  # phonon absorption in IR
        b1 = 2.860e-6
        b2 = 4.700e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis == "o":
        a1 = 5.653  # plasmons in the far UV
        a2 = 0.1185  # weight of UV pole
        a3 = 0.2091  # pole in UV
        a4 = 89.61  # weight of IR pole
        a5 = 10.85  # pole in IR
        a6 = 1.97e-2  # phonon absorption in IR
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6

    else:
        raise ValueError("axis needs to be o or e")

    wvl = c / v * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        (a1 + b1 * f)
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


# %% -----
ps = 1e-12
km = 1e3
um = 1e-6

width = np.arange(0.5, 2.1, 0.1)
freq_Al2O3 = wg.get_omega_axis(0.8, 4.5, 100)

# indexing goes:
# etch-depth, width, (wavelength, effective area, beta2)
x1000 = load(1)
x850 = load(0.85)
x700 = load(0.7)
x550 = load(0.55)
x400 = load(0.4)

# %% -----
n = 256
v0 = c / 1550e-9
e_p = 100e-12
t_fwhm = 50e-15
min_time_window = 10e-12

v_min = c / 4.5e-6
v_max = c / 0.8e-6
pulse = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    min_time_window,
)

# %% -----
x = x850
depth = np.arange(0.1, 0.850, 0.1)
DK = np.zeros((x.shape[0], x.shape[1], pulse.n))
A_EFF = np.zeros((x.shape[0], x.shape[1], pulse.n))
for i in tqdm(range(x.shape[0])):
    for j in range(x.shape[1]):
        a_eff = x[i, j][:, 0]
        beta = x[i, j][:, 1]

        spl_a_eff = InterpolatedUnivariateSpline(freq_Al2O3 * c / um, a_eff)
        spl_beta = InterpolatedUnivariateSpline(freq_Al2O3 * c / um, beta)
        a_eff = spl_a_eff(pulse.v_grid)
        beta = spl_beta(pulse.v_grid)

        length = 1e-3
        model = pynlo.materials.MgLN().generate_model(
            pulse,
            a_eff,
            length,  # really only relevant for gaussian beam
            g2_inv=None,
            beta=beta,
            is_gaussian_beam=False,
        )
        DK[i, j] = model.dispersive_wave_dk
        A_EFF[i, j] = a_eff
DK[A_EFF > 40] = np.nan

# %% ----- individual testing -----
etch_width = 2.0
etch_depth = 0.4
LiN_thickness = 0.85
resolution = 20
sim = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=LiN_thickness,
    substrate_medium=mtp.Al2O3,
    waveguide_medium=mt.LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    z_offset_wvgd=1.5,
    num_bands=1,
)

res = sim.calc_dispersion(wl_min=0.8, wl_max=4.5, NPTS=100, eps_func_wvgd=eps_func_wvgd)

omega = res.freq * 2 * np.pi * c / um
beta = res.kx.flatten() * 2 * np.pi / um
a_eff = np.zeros(res.freq.size)
for n_f in range(res.freq.size):
    x = sim.E[n_f][0][:, :, mp.Ey].__abs__() ** 2
    a_eff[n_f] = wg.mode_area(x, sim.resolution[0])
(idx,) = (a_eff < 40).nonzero()
idx_longest = idx[0]
idx_pump = abs(res.freq - 1 / 1.560).argmin()

# pynlo convergence takes a long time when there is a huge range on gamma
v_min = res.freq[idx_longest] * c / um
e_p = 10e-12  # adjust pulse energy
min_time_window = 20e-12  # adjust time window
pulse = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    min_time_window,
)

spl_a_eff = InterpolatedUnivariateSpline(res.freq * c / um, a_eff)
spl_beta = InterpolatedUnivariateSpline(res.freq * c / um, beta)
a_eff = spl_a_eff(pulse.v_grid)
beta = spl_beta(pulse.v_grid)

length = 40e-3

paths, (dk, v_sfg, dk_sfg, v_dfg, dk_dfg) = pynlo.utility.chi2.dominant_paths(
    pulse.v_grid, beta, beta_qpm=None, full=True
)

wl_target = 3.5e-6
bandpass = 100e-9
idx = np.logical_and(
    c / (wl_target + bandpass / 2) < v_dfg, v_dfg < c / (wl_target - bandpass / 2)
).nonzero()
delta_beta_1 = dk_dfg[idx].mean()

wl_target = 4.0e-6
idx = np.logical_and(
    c / (wl_target + bandpass / 2) < v_dfg, v_dfg < c / (wl_target - bandpass / 2)
).nonzero()
delta_beta_2 = dk_dfg[idx].mean()

z_start_ramp = 15e-3
func = (
    lambda z: delta_beta_1
    if z < z_start_ramp
    else (delta_beta_2 - delta_beta_1) / (length - z_start_ramp) * (z - z_start_ramp)
    + delta_beta_1
)
z = np.linspace(0, length, 1000)
dk = np.array([func(i) for i in z])

dk = dk[z > z_start_ramp]
z = z[z > z_start_ramp]
z_invs, domains, poled = pynlo.utility.chi2.domain_inversions(z, dk)

model = pynlo.materials.MgLN().generate_model(
    pulse=pulse,
    a_eff=a_eff * 1e-6**2,
    length=length,  # really only relevant for gaussian beam
    g2_inv=z_invs,
    # g2_inv=None,
    beta=beta,
    is_gaussian_beam=False,
)

sim_pynlo = model.simulate(length, dz=None, local_error=1e-6, n_records=100, plot="wvl")

sim_pynlo.plot("wvl")

fig, ax = plt.subplots(1, 1)
ax.plot(pulse.wl_grid * 1e6, model.dispersive_wave_dk)
ax.grid(True)
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("$\\mathrm{\\Delta \\beta (m)^{-1}}$")
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
# ax.plot(
#     pulse.wl_grid * 1e6, pynlo.utility.chi3.g3_to_gamma(pulse.v_grid, model.g3).real
# )
ax.plot(pulse.wl_grid * 1e6, a_eff)
ax.grid(True)
# ax.set_ylabel("gamma $(\\mathrm{Wm})^{-1}$")
ax.set_ylabel("effective area ($\\mathrm{\\mu m^2}$)")
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
fig.tight_layout()

sim.plot_mode(0, idx_longest)
sim.plot_mode(0, idx_pump)

idx = np.logical_and(pulse.v_grid.min() < v_dfg, v_dfg < pulse.v_grid.max()).nonzero()
idx_r = idx[0].min(), idx[0].max()
idx_c = idx[1].min(), idx[1].max()
wl_dfg = c / v_dfg
wl_dfg[v_dfg < pulse.v_grid.min()] = np.nan
fig, ax = plt.subplots(1, 2, figsize=np.array([6.97, 3.04]))
img_dfg = ax[0].pcolormesh(
    pulse.wl_grid[idx_c[0] : idx_c[1]] * 1e6,
    pulse.wl_grid[idx_r[0] : idx_r[1]] * 1e6,
    wl_dfg[idx_r[0] : idx_r[1], idx_c[0] : idx_c[1]] * 1e6,
)
img_dk = ax[1].pcolormesh(
    pulse.wl_grid[idx_c[0] : idx_c[1]] * 1e6,
    pulse.wl_grid[idx_r[0] : idx_r[1]] * 1e6,
    2 * np.pi / dk_dfg[idx_r[0] : idx_r[1], idx_c[0] : idx_c[1]] * 1e6,
)
plt.colorbar(img_dfg, ax=ax[0])
plt.colorbar(img_dk, ax=ax[1])
[i.set_xlabel("wavelength ($\\mathrm{\\mu m}$)") for i in ax]
[i.set_ylabel("wavelength ($\\mathrm{\\mu m}$)") for i in ax]
ax[0].set_title("DFG wavelength")
ax[1].set_title("QPM poling period")
fig.tight_layout()
