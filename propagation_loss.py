# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from TFW_meep import waveguide_dispersion as wg
from TFW_meep import materials as mtp
from meep import materials as mt
import meep as mp

# %% dimensions
etch_width = 2
etch_depth = 0.7
film_thickness = 1
substrate_medium = mtp.Al2O3
waveguide_medium = mt.LiNbO3

# %% create waveguide
TFW = wg.ThinFilmWaveguide(
    etch_width,
    etch_depth,
    film_thickness,
    substrate_medium,
    waveguide_medium,
    resolution=30,
    cell_width=10,
    cell_height=8,
    num_bands=1,
)

wl_min = 3.0
wl_max = 5.0

# %% run simulation to obtain a list of k_points
res = TFW.calc_dispersion(wl_min, wl_max, 19, eps_func_wvgd=None, eps_func_sbstrt=None)
k_points = res.kx
np.save("k_points.npy", np.c_[res.freq, np.squeeze(res.kx)])

# %% get sim
sim = TFW.sim

# sim is only used for visualization in TFW, the geometry objects were set to
# be non-dispersive, fixing that here:
sim.geometry = TFW.geometry

# %% construct source: place a gaussian source at the center of the waveguide,
#    set the time bandwidht to cover the desired frequency bandwidth, and make
#    sure you get the center frequency right.
fmin = 1 / wl_max
fmax = 1 / wl_min
df = fmax - fmin
fcen = df / 2 + fmin

src = mp.GaussianSource(frequency=fcen, fwidth=df)
sources = [mp.Source(src, mp.Ey, center=mp.Vector3())]

# %% pml layers
pml_layers = [
    mp.PML(thickness=1, direction=mp.Z),
    mp.PML(thickness=1, direction=mp.Y),
]

# %% set sources and pml layers
sim.sources = sources
sim.boundary_layers = pml_layers

# %% harminv monitoring: offset from the symmetry plane slightly in the
#    horizontal direction
h = mp.Harminv(mp.Ey, mp.Vector3(0, 0.1234, 0), fcen, df)

# %% run_k_points is basically a for loop for each k_point, and automatically
#    calls Harminv afterwards
k_points = np.load("k_points.npy")[:, 1]
k_points = [mp.Vector3(i, 0, 0) for i in k_points]
sim.run_k_points(300, k_points)
