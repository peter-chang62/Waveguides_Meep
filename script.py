import meep.materials as mt
import TFW_meep.waveguide_dispersion as wg
import TFW_meep.materials as mtp
# import meep as mp
import matplotlib.pyplot as plt
import clipboard as cr

plt.ion()

ridge = wg.RidgeWaveguide(
    width=5,
    height=4,
    substrate_medium=mtp.Al2O3,  # dispersive
    waveguide_medium=mt.LiNbO3,  # dispersive
    resolution=20,  # 20 -> 40 made neglibile difference!
    num_bands=8,
    cell_width=13,
    cell_height=12,
)

# at 3 micron you need 5-6 bands (anyways, 4 was too small)
# keep in mind you really just need 10 pts to do a prety good spline
# can roughly guide 3.9!
ridge.width = 3
ridge.height = 0.7
ridge.cell_width = 8
ridge.cell_height = 5
ridge.num_bands = 1

# ridge.wvgd_mdm = mp.Medium(epsilon=mt.LiNbO3.epsilon(1 / 1.55)[2, 2])
# ridge.sbstrt_mdm = mp.Medium(epsilon=mtp.Al2O3.epsilon(1 / 1.55)[2, 2])
# res = ridge.calc_w_from_k(.8, 3.5, 30)
res = ridge.calc_dispersion(0.8, 3.5, 30)

# %%___________________________________________________________________________
res.plot_dispersion()
# [ridge.plot_mode(0, n, 1) for n in range(len(res.kx))]
