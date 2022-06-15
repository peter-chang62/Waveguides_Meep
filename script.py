import meep.materials as mt
import waveguide_dispersion as wg
import materials as mtp
import meep as mp

ridge = wg.RidgeWaveguide(
    width=5,
    height=4,
    substrate_medium=mtp.Al2O3,  # dispersive
    waveguide_medium=mt.LiNbO3,  # dispersive
    resolution=20,  # 20 -> 40 made neglibile difference!
    num_bands=8,
    cell_width=13,
    cell_height=12
)

# at 3 micron you need 5-6 bands (anyways, 4 was too small)
# keep in mind you really just need 10 pts to do a prety good spline
# can roughly guide 3.9!
ridge.width = 3
ridge.height = .7
ridge.cell_width = 5
ridge.cell_height = 5
ridge.num_bands = 4

# ridge.wvgd_mdm = mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(1 / 1.55).diagonal())
# ridge.sbstrt_mdm = mp.Medium(epsilon_diag=mtp.Al2O3.epsilon(1 / 1.55).diagonal())
# res = ridge.calc_w_from_k(.4, 5, 15)
# res = ridge.calc_dispersion(.8, 2, 15)

# %%____________________________________________________________________________________________________________________
# res.plot_dispersion()
#
# plot = lambda n: ridge.plot_mode(res.sm_bands[n], n)
# for n in range(len(res.kx)):
#     plot(n)
