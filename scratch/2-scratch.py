"""Calculate the dispersion of a 2D LiNbO3 waveguide"""

import meep as mp
import meep.materials as mt
import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

# %% Set up the geometry of the problem. One waveguide laid out in the x direction
wl_wvgd = 3.5
n_center_wvl = mt.LiNbO3.epsilon(1 / wl_wvgd)[2, 2]  # z polarization
w_wvgd = 0.5 * wl_wvgd / n_center_wvl  # width of the waveguide is half a wavelength wide

center_wvgd = mp.Vector3(0, 0, 0)  # where the waveguide is centered
sy = 6  # size of the cell in y direction
sx = 0

dpml = 1  # PML thickness

# %% create the geometric objects from the above
blk = mp.Block(
    size=mp.Vector3(mp.inf, w_wvgd, mp.inf),
    center=center_wvgd
)
cell = mp.Vector3(sx, sy, 0)
PML = mp.PML(dpml, direction=mp.Y)

# %% set the appropriate media for the geometric objects
blk.material = mt.LiNbO3

# %% create the geometry and boundary layers list
geometry = [blk]
boundary_layers = [PML]

# %% create a gaussian source instance and place it at the front of the waveguide
bw = blk.material.valid_freq_range
f_src = float((np.diff(bw) / 2) + bw[0])
df_src = float(np.diff(bw) * 1)

src = mp.GaussianSource(frequency=f_src,
                        fwidth=df_src
                        )
pt_src_offset = mp.Vector3(0, w_wvgd * 0.25)
pt_src = center_wvgd + pt_src_offset
source = mp.Source(
    src=src,
    component=mp.Ez,
    center=pt_src,
)

Sources = [source]

# %% Done with sources, initialize the simulation instance
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    sources=Sources,
                    boundary_layers=boundary_layers,
                    resolution=50,
                    )
sim.use_output_directory('sim_output')

# %% Exploit symmetries (if there are any)
symx = mp.Symmetry(
    direction=mp.X,
    phase=1
)

sim.symmetries = [symx]

# %% Calculate Waveguide Dispersion set periodic boundary conditions
# for a given k_point, run the sim, then anlayze
# result with Harminv repeatedly at multile k_points
kmin, kmax = blk.material.valid_freq_range
kpts = mp.interpolate(19, [mp.Vector3(kmin), mp.Vector3(kmax)])
kx = np.array([i.x for i in kpts])
freq = sim.run_k_points(300, kpts)

# %%
plt.figure()
plt.plot([kx.min(), kx.max()], [kx.min(), kx.max()], 'k', label='light line')
for n in range(len(freq)):
    if len(freq[n]) > 0:
        [plt.plot(kx[n], i.real, marker='o', color='C0') for i in freq[n]]
plt.legend(loc='best')
