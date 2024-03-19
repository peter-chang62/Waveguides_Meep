# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from meep import materials as mt
from TFW_meep import materials as mtp
from TFW_meep import waveguide_dispersion as wg
import scipy.constants as sc
from TFW_meep import geometry
from scipy.interpolate import UnivariateSpline
import copy
from scipy.constants import c
import clipboard


# %% ----- Gayer paper Sellmeier equation for ne (taken from PyNLO 1 / omega is in um
# n = squrt[epsilon] so epsilon = n^2
def eps_func_wvgd(omega):
    # omega is in inverse micron
    um = 1e-6
    v = omega * sc.c / um
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

    wvl = sc.c / v * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        (a1 + b1 * f)
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


# %% ----- create sim instance ------------------------------------------------
etch_width = 1.3
etch_depth = 0.2
resolution = 30
sim = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=0.4,
    substrate_medium=mt.SiO2,
    waveguide_medium=mt.LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    z_offset_wvgd=1.5,
    num_bands=1,
)

# %% ----- waveguide simulation -----------------------------------------------
res = sim.calc_dispersion(0.4, 1.77, 100, eps_func_wvgd=eps_func_wvgd)  # simulate

wl = 1 / res.freq
um = 1e-6
omega = res.freq * 2 * np.pi * c / um
# conversion = sc.c**-2 * 1e12**2 * 1e3**2 * 1e-9
beta = res.kx.flatten() * 2 * np.pi / um
beta1 = np.gradient(beta, omega, edge_order=2)
beta2 = np.gradient(beta1, omega, edge_order=2)

# %% ----- prep time domain sim -----------------------------------------------
# fcen = 1 / 3.8
# df = 0.1

# blk_wvgd, blk_sbstrt, blk_film = copy.deepcopy(sim.geometry)
# blk_wvgd = geometry.convert_block_to_trapezoid(blk_wvgd, angle_deg=80)
# blk_film.size.y = sim.lattice.size.y
# blk_sbstrt.size.y = sim.lattice.size.y

# blk_wvgd.material = mp.Medium(epsilon=eps_func_wvgd(fcen))
# blk_film.material = mp.Medium(epsilon=eps_func_wvgd(fcen))
# blk_sbstrt.material = mp.Medium(epsilon=sim.sbstrt_mdm.epsilon(fcen).diagonal()[-1])

# geometry = [blk_wvgd, blk_film, blk_sbstrt]

# dpml = 3
# PML = [mp.PML(dpml, mp.Y), mp.PML(dpml, mp.Z)]

# length = 0  # 1 um
# cell_size = mp.Vector3(
#     length, sim.lattice.size.y + dpml * 2, sim.lattice.size.z + dpml * 2
# )
# if length > 0:
#     blk_wvgd.height = length
#     blk_wvgd.center.x = 0

# sources = [
#     mp.EigenModeSource(
#         src=mp.GaussianSource(frequency=fcen, fwidth=df),
#         center=mp.Vector3(0, 0, 0),
#         size=mp.Vector3(
#             0,
#             sim.lattice.size.y,
#             sim.lattice.size.z,
#         ),
#         eig_match_freq=True,
#         eig_parity=mp.ODD_Y,
#     )
# ]

# # resolution = (1 / fcen / eps_func_wvgd(fcen) / 10) ** -1
# resolution = 20

# sim_td = mp.Simulation(
#     cell_size=cell_size,
#     geometry=geometry,
#     sources=sources,
#     boundary_layers=PML,
#     resolution=resolution,
#     symmetries=[mp.Mirror(mp.Y, phase=-1)],
# )

# %% ------ monitors ----------------------------------------------------------
# nonpml_vol = mp.Volume(
#     center=mp.Vector3(),
#     size=mp.Vector3(
#         length,
#         sim.lattice.size.y,
#         sim.lattice.size.z,
#     ),
# )
# dft_obj = sim_td.add_dft_fields([mp.Ey], np.array([fcen]), where=nonpml_vol)

# %% ------ run FDTD ----------------------------------------------------------
# Iy = []


# def step(sim):
#     sim: sim_td
#     arr = sim.get_dft_array(dft_obj, mp.Ey, 0).copy()
#     Iy.append(abs(arr) ** 2)


# sim_td.use_output_directory("eps/")
# sim_td.run(
#     mp.after_sources(mp.at_every(100, step)),
#     until_after_sources=10000,  # 1 centimeter
# )

# Iy = np.asarray(Iy)


# %% ------ look at results ---------------------------------------------------
# def video():
#     fig, ax = plt.subplots(1, 1)
#     for n, i in enumerate(Iy):
#         ax.clear()
#         ax.imshow(i.T[::-1])
#         ax.set_title((n + 1) * 100)
#         plt.pause(0.1)

ps = 1e-12
km = 1e3
fig, ax = plt.subplots(1, 1)
ax.plot(wl[:], beta2[:] / (ps**2 / km), "o")
s = UnivariateSpline(wl[::-1], beta2[::-1] / (ps**2 / km), s=0)
_ = np.linspace(wl.min(), wl.max(), 1000)
ax.plot(_, s(_), linewidth=2)
ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax.set_ylabel("$\\mathrm{\\beta_2 \\; (ps^2/km})$")
ax.set_title(str(np.round((sim.height - sim.etch_depth) * 1e3, 3)) + " nm slab")
fig.tight_layout()
