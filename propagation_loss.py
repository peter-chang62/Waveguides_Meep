# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from TFW_meep import waveguide_dispersion as wg
from TFW_meep import materials as mtp
from meep import materials as mt
import meep as mp
import pynlo_extras as pe
from scipy import constants as sc


# %% materials
ppln = pe.materials.PPLN()


def eps_func_wvgd(freq):
    um = 1e-6

    nu = freq * sc.c / um
    return ppln.n(nu) ** 2


substrate_medium = mtp.Al2O3
waveguide_medium = mt.LiNbO3

# %% create waveguide
etch_width = 3
etch_depth = 0.3
film_thickness = 0.630
TFW = wg.ThinFilmWaveguide(
    etch_width,
    etch_depth,
    film_thickness,
    substrate_medium,
    waveguide_medium,
    resolution=30,
    cell_width=8,
    cell_height=6,
    num_bands=1,
)

# %% run simulation to obtain a list of k_points the unguided modes are those
#    that lie above the light line, which are the ones that we want to solve
#    for propagation loss. You might want to run this simulation separately
#    because MPB is not meant to be called with multiple processes using
#    mpirun (it'll take longer!)
# wl_min = 3
# wl_max = 5
# res = TFW.calc_dispersion(
#     wl_min, wl_max, 100, eps_func_wvgd=eps_func_wvgd, eps_func_sbstrt=None
# )

# (unguided,) = (res.freq > np.squeeze(res.kx) / res._index_sbstrt).nonzero()
# k_points = np.c_[res.freq, np.squeeze(res.kx)]
# np.save("k_points.npy", k_points)
# np.save("unguided_index.npy", unguided)

# ----------------------- propagation loss calculation ------------------------

# %% add pml layers
dpml = 1
pml_layers = [
    mp.PML(thickness=dpml, direction=mp.Z),
    mp.PML(thickness=dpml, direction=mp.Y),
]
TFW.cell_width += dpml * 4  # extend the cell for the pml layers
TFW.cell_height += dpml * 4
sim = TFW.sim
sim.boundary_layers = pml_layers

k_points = np.load("k_points.npy")
unguided = np.load("unguided_index.npy")

# %% run for unguided frequencies, index from high v to low v, truncating when
#    the loss is too high
FREQ = []
DECAY = []
for fcen, kx in k_points[unguided][::-1]:
    # reset
    sim.reset_meep()

    # sources
    src = mp.GaussianSource(frequency=fcen, width=10)
    sources = [mp.Source(src, mp.Ey, center=mp.Vector3())]
    sim.sources = sources

    # set waveguide epsilon
    eps_wvgd = eps_func_wvgd(fcen)
    TFW.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
    TFW._blk_film.material = mp.Medium(epsilon=eps_wvgd)

    # set substrate epsilon
    eps_sbstrt = TFW.sbstrt_mdm.epsilon(fcen)[2, 2]
    TFW.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt)
    sim.geometry = TFW.geometry

    # harminv monitoring: offset from the symmetry plane slightly in the
    # horizontal direction
    df = 100e-3 / (1 / fcen) ** 2  # 100 nm
    h = mp.Harminv(mp.Ey, mp.Vector3(0, 0.1234, 0), fcen, df)

    k_point = mp.Vector3(kx, 0, 0)
    sim.k_point = k_point
    sim.run(mp.after_sources(h), until_after_sources=300)

    if len(h.modes) == 0:
        break
    else:
        # for propagation loss, I work close to the cut off frequency, there
        # shouldn't be other modes close by
        assert len(h.modes) == 1
        (mode,) = h.modes
        re = mode.freq
        im = abs(mode.decay)
        FREQ.append(re)
        DECAY.append(im)
unguided_dispersion = np.c_[FREQ, DECAY]
np.save("unguided_dispersion.npy", unguided_dispersion)

# %% run for guided frequencies, index from low v to high v, truncating when
#    the loss is sufficiently low
# FREQ = []
# DECAY = []
# for fcen, kx in k_points[unguided[-1] + 1 :]:
#     # reset
#     sim.reset_meep()

#     # sources
#     src = mp.GaussianSource(frequency=fcen, width=10)
#     sources = [mp.Source(src, mp.Ey, center=mp.Vector3())]
#     sim.sources = sources

#     # set waveguide epsilon
#     eps_wvgd = eps_func_wvgd(fcen)
#     TFW.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
#     TFW._blk_film.material = mp.Medium(epsilon=eps_wvgd)

#     # set substrate epsilon
#     eps_sbstrt = TFW.sbstrt_mdm.epsilon(fcen)[2, 2]
#     TFW.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt)
#     sim.geometry = TFW.geometry

#     # harminv monitoring: offset from the symmetry plane slightly in the
#     # horizontal direction
#     df = 100e-3 / (1 / fcen) ** 2  # 100 nm
#     h = mp.Harminv(mp.Ey, mp.Vector3(0, 0.1234, 0), fcen, df)

#     k_point = mp.Vector3(kx, 0, 0)
#     sim.k_point = k_point
#     sim.run(mp.after_sources(h), until_after_sources=300)

#     # for propagation loss, I work close to the cut off frequency, there
#     # shouldn't be other modes close by
#     assert len(h.modes) == 1
#     (mode,) = h.modes
#     re = mode.freq
#     im = abs(mode.decay)
#     FREQ.append(re)
#     DECAY.append(im)

#     if im <= 1e-5:
#         break

# guided_dispersion = np.c_[FREQ, DECAY]
# np.save("guided_dispersion.npy", guided_dispersion)

# dispersion = np.vstack([unguided_dispersion[::-1], guided_dispersion])
# np.save("dispersion.npy", dispersion)
