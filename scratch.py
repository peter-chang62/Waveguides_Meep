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
                 resolution=20, num_bands=4, sy=8, sz=8):
        # passed as arguments from __init__()
        assert isinstance(substrate_medium, mp.Medium), \
            f"substrate_medium must be a mp.Medium instance but got type {type(substrate_medium)}"
        assert isinstance(substrate_medium, mp.Medium), \
            f"substrate_medium must be a mp.Medium instance but got type {type(waveguide_medium)}"
        substrate_medium: mp.Medium
        waveguide_medium: mp.Medium
        self.wdth = width
        self.hght = height
        self.sbstrt_mdm = substrate_medium
        self.wvgd_mdm = waveguide_medium

        # additional attributes
        self._sy = sy
        self._sz = sz

        hght_sbstrt = (self._sz / 2) - (self.hght / 2)
        offst_sbstrt = (hght_sbstrt / 2) + (self.hght / 2)

        self.blk_wvgd = mp.Block(size=mp.Vector3(mp.inf, self.wdth, self.hght))
        self.blk_sbstrt = mp.Block(size=mp.Vector3(mp.inf, mp.inf, hght_sbstrt),
                                   center=mp.Vector3(0, 0, -offst_sbstrt))

        self._lattice = mp.Lattice(size=mp.Vector3(0, self._sy, self._sz))
        self._geometry = [self.blk_wvgd, self.blk_sbstrt]

        self.ms = mpb.ModeSolver(geometry_lattice=self._lattice,
                                 geometry=self._geometry,
                                 resolution=resolution,
                                 num_bands=num_bands)

        geometry_sim = copy.deepcopy(self._geometry)
        geometry_sim[0].material = mp.Medium(epsilon=self.wvgd_mdm.epsilon(1 / 1.55)[2, 2])
        geometry_sim[1].material = mp.Medium(epsilon=self.sbstrt_mdm.epsilon(1 / 1.55)[2, 2])
        self.sim = mp.Simulation(cell_size=self._lattice.size,
                                 geometry=geometry_sim,
                                 resolution=resolution)

    @property
    def resolution(self):
        return self.ms.resolution

    @resolution.setter
    def resolution(self, resolution):
        self.ms.resolution = resolution

    @property
    def num_bands(self):
        return self.ms.num_bands

    @num_bands.setter
    def num_bands(self, num):
        self.ms.num_bands = num

    @property
    def wvgd_mdm(self):
        return self.blk_wvgd.material

    @wvgd_mdm.setter
    def wvgd_mdm(self, medium):
        assert isinstance(medium, mp.Medium), \
            f"substrate_medium must be a mp.Medium instance but got type {type(medium)}"

        self.blk_wvgd.material = medium

    @property
    def sbstrt_mdm(self):
        return self.blk_sbstrt.material

    @sbstrt_mdm.setter
    def sbstrt_mdm(self, medium):
        assert isinstance(medium, mp.Medium), \
            f"substrate_medium must be a mp.Medium instance but got type {type(medium)}"

        self.blk_sbstrt.material = medium

    def plot2D(self):
        self.sim.plot2D()

    def calculate_dispersion(self, wl_min, wl_max, NPTS):
        if (self.wvgd_mdm.valid_freq_range[-1] == 1e20) and (self.sbstrt_mdm.valid_freq_range[-1] == 1e20):
            return self._calc_dispersive(wl_min, wl_max, NPTS)
        else:
            return self._calc_non_dispersive(wl_min, wl_max, NPTS)

    def _calc_dispersive(self, wl_min, wl_max, NPTS):
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

        return np.array([i.x for i in k_points]), FREQ, EPS_WVGD, EPS_SBSTRT

    def _calc_non_dispersive(self, wl_min, wl_max, NPTS):
        k_points = mp.interpolate(NPTS, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])
        self.ms.k_points = k_points
        self.ms.run()
        return np.array([i.x for i in k_points]), self.ms.all_freqs


# %%____________________________________________________________________________________________________________________
ridge = RidgeWaveGuide(
    width=.5,
    height=.22,
    substrate_medium=mp.Medium(index=1.45),
    waveguide_medium=mp.Medium(index=3.45),
    resolution=64,
    num_bands=4,
    sy=2,
    sz=2
)

# %%____________________________________________________________________________________________________________________
kx, freq, eps_wvgd, eps_sbstrt = ridge.calculate_dispersion(1 / 2.0, 1 / .1, 20)
