"""complex transmission of a 2D waveguide taper, you can set things up following mode-decomposition.py from
meep/examples/

This is sort of the basic example. In the mode-decomposition, they placed the flux monitor before the taper to
characterize reflection. So, they had to call load_minus_flux_data to avoid interference effects """

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import materials as mtp
import clipboard_and_style_sheet
import h5py

clipboard_and_style_sheet.style_sheet()


def play_video():
    with h5py.File('sim_output/06-18-2022/scratch-2-ez.h5', 'r') as f:
        with h5py.File('sim_output/06-18-2022/scratch-2-eps-000000.00.h5', 'r') as f2:
            data = f.get('ez')
            eps = np.array(f2.get('eps'))
            fig, ax = plt.subplots(1, 1)
            for n in range(data.shape[2]):
                ax.clear()
                ax.imshow(eps[::-1, ::-1].T, cmap='binary')
                ax.imshow(data[:, :, n][::-1, ::-1].T, cmap='jet', alpha=.9)
                plt.pause(.01)


# %% ___________________________________________________________________________________________________________________
# dimensions
w1 = 1.0  # width of waveguide 1
w2 = 2.0  # width of waveguide 2
Lw = 10.0  # length of waveguides 1 and 2

Lt = 8  # length of taper

dair = 3.0  # length of air region
dpml_x = 6.0  # length of PML in x direction
dpml_y = 2.0  # length of PML in y direction

sy = dpml_y + dair + w2 + dair + dpml_y

sx = dpml_x + Lw + Lt + Lw + dpml_x

# %% ___________________________________________________________________________________________________________________
# cell, geometry and boundary layers
cell_size = mp.Vector3(sx, sy, 0)

boundary_layers = [mp.PML(dpml_x, direction=mp.X),
                   mp.PML(dpml_y, direction=mp.Y)]

# linear taper
vertices_taper = [mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
                  mp.Vector3(-0.5 * Lt, 0.5 * w1),
                  mp.Vector3(0.5 * Lt, 0.5 * w2),
                  mp.Vector3(0.5 * sx + 1, 0.5 * w2),
                  mp.Vector3(0.5 * sx + 1, -0.5 * w2),
                  mp.Vector3(0.5 * Lt, -0.5 * w2),
                  mp.Vector3(-0.5 * Lt, -0.5 * w1),
                  mp.Vector3(-0.5 * sx - 1, -0.5 * w1)]

vertices_straight = [mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
                     mp.Vector3(0.5 * sx + 1, 0.5 * w1),
                     mp.Vector3(0.5 * sx + 1, -0.5 * w1),
                     mp.Vector3(-0.5 * sx - 1, -0.5 * w1)]

geometry_taper = [mp.Prism(vertices_taper, height=mp.inf, material=mp.Medium(epsilon=12.0))]
geometry_straight = [mp.Prism(vertices_straight, height=mp.inf, material=mp.Medium(epsilon=12.0))]

# %% ___________________________________________________________________________________________________________________
# sources
lcen = 6.67  # mode wavelength
fcen = 1 / lcen  # mode frequency
df = 0.2
src_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.2 * Lw)
sources = [mp.EigenModeSource(src=mp.GaussianSource(frequency=fcen, fwidth=df),
                              center=src_pt,
                              size=mp.Vector3(y=sy - 2 * dpml_y),
                              eig_match_freq=True,
                              eig_parity=mp.ODD_Z + mp.EVEN_Y)]

# %% ___________________________________________________________________________________________________________________
resolution = 25  # pixels/μm
sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    geometry=geometry_taper,
                    sources=sources,
                    symmetries=[mp.Mirror(mp.Y)]
                    )

# %% ___________________________________________________________________________________________________________________
mon_pt_after = mp.Vector3(0.5 * sx - dpml_x - 0.7 * Lw)
nfreq = 100
flux = sim.add_flux(fcen,
                    0.2,  # rough bandwidth of the waveguide
                    nfreq,
                    mp.ModeRegion(center=mon_pt_after,
                                  size=mp.Vector3(y=sy - 2 * dpml_y)))
# %% ___________________________________________________________________________________________________________________
sim.use_output_directory('sim_output/06-18-2022')
sim.run(
    # mp.at_beginning(mp.output_epsilon),
    # mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt_after, 1e-3)
)

res_taper = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.EVEN_Y + mp.ODD_Z)
alpha_taper = res_taper.alpha[0, :, 0]

# play_video()
# plt.close()

# %% ______________________________________ Round 2_____________________________________________________________________
sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    geometry=geometry_straight,
                    sources=sources,
                    symmetries=[mp.Mirror(mp.Y)]
                    )

# %% ___________________________________________________________________________________________________________________
mon_pt_after = mp.Vector3(0.5 * sx - dpml_x - 0.7 * Lw)
nfreq = 100
flux = sim.add_flux(fcen,
                    0.2,  # rough bandwidth of the waveguide
                    nfreq,
                    mp.ModeRegion(center=mon_pt_after,
                                  size=mp.Vector3(y=sy - 2 * dpml_y)))
# %% ___________________________________________________________________________________________________________________
sim.use_output_directory('sim_output/06-18-2022')
sim.run(
    # mp.at_beginning(mp.output_epsilon),
    # mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt_after, 1e-3)
)

res_straight = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.EVEN_Y + mp.ODD_Z)
alpha_straight = res_straight.alpha[0, :, 0]

# play_video()
# plt.close()

# %% ___________________________________________________________________________________________________________________
trans = alpha_taper / alpha_straight
fig, (ax, ax3) = plt.subplots(1, 2, figsize=np.array([11.91, 5.27]))
ax2 = ax.twinx()
ax.plot(np.array(flux.freq), trans.__abs__() ** 2, '.-')
ax2.plot(np.array(flux.freq), np.unwrap(np.arctan2(trans.imag, trans.real)) * 180 / np.pi, '.-', color='C1')
ax3.plot(np.array(flux.freq), alpha_straight.__abs__() ** 2, '.-')
ax.set_ylim(ymax=1.001)
