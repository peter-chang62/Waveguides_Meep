import meep as mp
import meep.materials as mt
import numpy as np
import copy
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import utilities as util
import h5py
import time

clipboard_and_style_sheet.style_sheet()


class RidgeWaveGuide:
    def __init__(self, width, height, substrate_medium, waveguide_medium,
                 resolution=64, num_bands=4, cell_width=2, cell_height=2):

        self.lattice = mp.Lattice(size=mp.Vector3(0, cell_width, cell_height))

        self.blk_wvgd = mp.Block(size=mp.Vector3(mp.inf, width, height))
        self.blk_sbstrt = mp.Block(size=mp.Vector3(mp.inf, mp.inf, self._hght_sbsrt),
                                   center=mp.Vector3(0, 0, -self._z_offst_sbstrt))

        self.sbstrt_mdm = substrate_medium
        self.wvgd_mdm = waveguide_medium

        self.geometry = [self.blk_wvgd, self.blk_sbstrt]

        self.ms = mpb.ModeSolver(geometry_lattice=self.lattice,
                                 geometry=self.geometry,
                                 resolution=resolution,
                                 num_bands=num_bands)

        self.redef_sim()

    def redef_sim(self):
        geometry_sim = copy.deepcopy(self.geometry)
        geometry_sim[0].material = mp.Medium(epsilon=self.wvgd_mdm.epsilon(1 / 1.55)[2, 2])
        geometry_sim[1].material = mp.Medium(epsilon=self.sbstrt_mdm.epsilon(1 / 1.55)[2, 2])
        self.sim = mp.Simulation(cell_size=self.lattice.size,
                                 geometry=geometry_sim,
                                 resolution=self.resolution[0])

    def redef_sbstrt_dim(self):
        self.blk_sbstrt.size.z = self._hght_sbsrt
        self.blk_sbstrt.center.z = -self._z_offst_sbstrt

    @property
    def _hght_sbsrt(self):
        return (self.cell_height / 2) - (self.height / 2)

    @property
    def _z_offst_sbstrt(self):
        return (self._hght_sbsrt / 2) + (self.height / 2)

    @property
    def width(self):
        return self.blk_wvgd.size.y

    @width.setter
    def width(self, width):
        self.blk_wvgd.size.y = width
        self.redef_sim()

    @property
    def height(self):
        return self.blk_wvgd.size.z

    @height.setter
    def height(self, height):
        self.blk_wvgd.size.z = height
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def cell_width(self):
        return self.lattice.size.y

    @cell_width.setter
    def cell_width(self, sy):
        self.lattice.size.y = sy
        self.redef_sim()

    @property
    def cell_height(self):
        return self.lattice.size.z

    @cell_height.setter
    def cell_height(self, sz):
        self.lattice.size.z = sz
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def resolution(self):
        return self.ms.resolution

    @resolution.setter
    def resolution(self, resolution):
        self.ms.resolution = resolution
        self.redef_sim()

    @property
    def num_bands(self):
        return self.ms.num_bands

    @num_bands.setter
    def num_bands(self, num):
        self.ms.num_bands = num

    @property
    def wvgd_mdm(self):
        return self._wvgd_mdm

    @wvgd_mdm.setter
    def wvgd_mdm(self, medium):
        assert isinstance(medium, mp.Medium), \
            f"waveguide medium must be a mp.Medium instance but got type {type(medium)}"
        medium: mp.Medium

        self.blk_wvgd.material = medium
        self._wvgd_mdm = medium

    @property
    def sbstrt_mdm(self):
        return self._sbstrt_mdm

    @sbstrt_mdm.setter
    def sbstrt_mdm(self, medium):
        assert isinstance(medium, mp.Medium), \
            f"substrate medium must be a mp.Medium instance but got type {type(medium)}"
        medium: mp.Medium

        self.blk_sbstrt.material = medium
        self._sbstrt_mdm = medium

    def plot2D(self):
        self.sim.plot2D()

    def calculate_dispersion(self, wl_min, wl_max, NPTS):
        if (self.wvgd_mdm.valid_freq_range[-1] == 1e20) and (self.sbstrt_mdm.valid_freq_range[-1] == 1e20):
            # run a NON dispersive calculation
            return self._calc_non_dispersive(wl_min, wl_max, NPTS)
        else:
            # otherwise run a dispersive calculation
            return self._calc_dispersive(wl_min, wl_max, NPTS)

    def _calc_dispersive(self, wl_min, wl_max, NPTS):
        print("RUNNING A DISPERSIVE CALCULATION")

        k_points = mp.interpolate(NPTS, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])
        FREQ = np.zeros((len(k_points), self.num_bands))
        EPS_WVGD = np.zeros(len(k_points))
        EPS_SBSTRT = np.zeros(len(k_points))

        start = time.time()
        for n, k in enumerate(k_points):
            eps_wvgd = self.wvgd_mdm.epsilon(k.x)[2, 2]
            eps_sbstrt = self.sbstrt_mdm.epsilon(k.x)[2, 2]
            self.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
            self.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt)

            self.ms.k_points = [k]
            self.ms.run()

            FREQ[n] = self.ms.all_freqs[0]
            EPS_WVGD[n] = eps_wvgd.real
            EPS_SBSTRT[n] = eps_sbstrt.real

            print(f'____________________________{len(k_points) - n}________________________________________')

        stop = time.time()
        print(f'finished after {(stop - start) / 60} minutes')

        class results:
            def __init__(self):
                self.kx = np.array([i.x for i in k_points])
                self.freq = FREQ
                self.eps_wvgd = EPS_WVGD
                self.eps_sbstrt = EPS_SBSTRT

        return results()

    def _calc_non_dispersive(self, wl_min, wl_max, NPTS):
        print("RUNNING WITH A FIXED EPSILON")

        k_points = mp.interpolate(NPTS, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])
        self.ms.k_points = k_points
        self.ms.run()

        class results:
            def __init__(self):
                self.kx = np.array([i.x for i in k_points])

        res = results()
        res.freq = self.ms.all_freqs
        return res


# %%____________________________________________________________________________________________________________________
wl_wvgd = 3.5  # um
n_cntr_wl = mt.LiNbO3.epsilon((1 / wl_wvgd))[2, 2]  # ne polarization
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl

ridge = RidgeWaveGuide(
    width=wdth_wvgd,
    height=.5,
    # substrate_medium=mp.Medium(index=1.45),
    # waveguide_medium=mp.Medium(index=3.45),
    substrate_medium=mt.SiO2,
    waveguide_medium=mt.LiNbO3,
    resolution=30,
    num_bands=4,
    cell_width=8,
    cell_height=8
)

# %%____________________________________________________________________________________________________________________
res = ridge.calculate_dispersion(.4, 1.77, 19)

# %%____________________________________________________________________________________________________________________
plt.figure()
[plt.plot(res.kx, res.freq[:, n], '.-') for n in range(res.freq.shape[1])]
plt.plot(res.kx, res.kx, 'k', label='light line')
plt.legend(loc='best')
plt.xlabel("k ($\mathrm{\mu m}$)")
plt.ylabel("$\mathrm{\\nu}$ ($\mathrm{\mu m}$)")
plt.ylim(.25, 2.5)
