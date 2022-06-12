"""MPB can solve for omega's given k, or solve for k's given omegas. The former is called with ms.run() (and its
variations), and the latter is called using ms.find_k(). In the RidgeWaveguide class, I use the former to calculate
the dispersion relation. You can use the latter if you are interested to know what modes can propagate at a given
frequency. For example, you can imagine that for a frequency well above the cutoff, there are multiple eigenmodes
that can propagate which results in different k's. MPB solves for the modes starting with highest order and going
down to the fundamental mode. This is because information retained from the higher order modes can be used to
accelerate the calculations for the lower ones. You can pass arguments to ms.find_k(), for example, if you wanted it
to retain the E and H fields that are calculated. This would allow you to plot the field cross-sections inside the
waveguide for the different modes. """

import meep as mp
import meep.materials as mt
import numpy as np
import copy
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import utilities as util
import h5py
import time

clipboard_and_style_sheet.style_sheet()


def store_fields(ms, which_band, cls):
    assert isinstance(cls, RidgeWaveguide), \
        f"cls must be an instance of RidgeWaveguide but got {type(cls)}"
    cls: RidgeWaveguide
    ms: mpb.ModeSolver
    cls.E.append(ms.get_efield(which_band=which_band, bloch_phase=False))
    cls.H.append(ms.get_hfield(which_band=which_band, bloch_phase=False))


class RidgeWaveguide:
    """
    The RidgeWaveguide class is for calculating the waveguide dispersion of a rectangular waveguide sitting on top of
    an infinitely large substrate.

    Relevant properties (dimensions, and material dispersion) can be easily modified after class instantiation. This
    way, one can easily sweep waveguide parameters (such as waveguide width / height).
    """

    def __init__(self, width, height, substrate_medium, waveguide_medium,
                 resolution=64, num_bands=4, cell_width=2, cell_height=2):

        self.init_finished = False

        # create the lattice (cell)
        self.lattice = mp.Lattice(size=mp.Vector3(0, cell_width, cell_height))

        # create the waveguide and the substrate
        # the substrate dimensions are calculated from the waveguide and
        # lattice dimensions (so it has to be created after the first two)
        self.blk_wvgd = mp.Block(size=mp.Vector3(mp.inf, width, height))
        self.blk_sbstrt = mp.Block(size=mp.Vector3(mp.inf, mp.inf, self._hght_sbsrt),
                                   center=mp.Vector3(0, 0, -self._z_offst_sbstrt))

        # set the substrate and waveguide medium
        self.sbstrt_mdm = substrate_medium
        self.wvgd_mdm = waveguide_medium

        # list of MEEP objects that will be passed to the mode solver
        self.geometry = [self.blk_wvgd, self.blk_sbstrt]

        # create the mode solver instance
        # self.geometry and self.lattice are passed by pointer, whereas
        # self.num_bands and self.resolution are passed by copy
        self.ms = mpb.ModeSolver(geometry_lattice=self.lattice,
                                 geometry=self.geometry,
                                 resolution=resolution,
                                 num_bands=num_bands)

        # create an mp.Simulation instance
        # this is only used for visualization (sim.plot2D())
        # plotting is a useful tool to see if the dimensions are set to what
        # you had intended to
        self.redef_sim()

        self.band_funcs = []
        self.run = self.ms.run_yeven
        self.store_fields = True
        self.init_finished = True

    def redef_sim(self):
        """
        re-initialize the mp.Simulation instance used for visualization
        """

        geometry_sim = copy.deepcopy(self.geometry)
        geometry_sim[0].material = mp.Medium(epsilon_diag=self.wvgd_mdm.epsilon(1 / 1.55).diagonal())
        geometry_sim[1].material = mp.Medium(epsilon_diag=self.sbstrt_mdm.epsilon(1 / 1.55).diagonal())
        self.sim = mp.Simulation(cell_size=self.lattice.size,
                                 geometry=geometry_sim,
                                 resolution=self.resolution[0])

    def redef_sbstrt_dim(self):
        """
        if waveguide dimensions that affect the substrate dimensions are changed, the substrate dimensions need to be
        appropriately modified
        """

        self.blk_sbstrt.size.z = self._hght_sbsrt
        self.blk_sbstrt.center.z = -self._z_offst_sbstrt

    @property
    def _hght_sbsrt(self):
        # calculate the appropriate height of the substrate
        return (self.cell_height / 2) - (self.height / 2)

    @property
    def _z_offst_sbstrt(self):
        # calculate the appropriate vertical offset of the substrate
        # to place it directly beneath the waveguide
        return (self._hght_sbsrt / 2) + (self.height / 2)

    @property
    def width(self):
        # return the width of the waveguide
        return self.blk_wvgd.size.y

    @width.setter
    def width(self, width):
        # set the width of the waveguide
        self.blk_wvgd.size.y = width
        self.redef_sim()

    @property
    def height(self):
        # return the height of the waveguide
        return self.blk_wvgd.size.z

    @height.setter
    def height(self, height):
        # set the height of the waveguide
        self.blk_wvgd.size.z = height
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def cell_width(self):
        # return the width of the supercell
        return self.lattice.size.y

    @cell_width.setter
    def cell_width(self, sy):
        # set the width of the supercell
        self.lattice.size.y = sy
        self.redef_sim()

    @property
    def cell_height(self):
        # return the height of the supercell
        return self.lattice.size.z

    @cell_height.setter
    def cell_height(self, sz):
        # set the height of the supercell
        self.lattice.size.z = sz
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def resolution(self):
        # return the resolution of the simulation grid
        return self.ms.resolution

    @resolution.setter
    def resolution(self, resolution):
        # set the resolution of the simulation grid
        self.ms.resolution = resolution
        self.redef_sim()

    @property
    def num_bands(self):
        # return the number of bands to be calculated
        return self.ms.num_bands

    @num_bands.setter
    def num_bands(self, num):
        # set the number of bands to be calculated
        self.ms.num_bands = num

    """MPB does not support dispersive materials. To get around this, we pass the k_points at which to calculate 
    omega one at a time, each time modifying epsilon. This is really the same run time because if you pass a list of 
    k_points, it simply calculates the omegas serially for each k_point in the list. 
    
    When modifying epsilon, however, we set the medium of the waveguide and substrate to a material with fixed 
    epsilon. So, it's important to keep a copy of the original dispersive mp.Medium() instance, for reference in the 
    dispersive calculations. This is stored in self._wvgd_mdm and self._sbstrt_mdm. """

    @property
    def wvgd_mdm(self):
        # return the waveguide medium
        return self._wvgd_mdm

    @wvgd_mdm.setter
    def wvgd_mdm(self, medium):
        # set the waveguide medium
        assert isinstance(medium, mp.Medium), \
            f"waveguide medium must be a mp.Medium instance but got type {type(medium)}"
        medium: mp.Medium

        self.blk_wvgd.material = medium
        self._wvgd_mdm = medium

        if self.init_finished:
            self.redef_sim()

    @property
    def sbstrt_mdm(self):
        # return the substrate medium
        return self._sbstrt_mdm

    @sbstrt_mdm.setter
    def sbstrt_mdm(self, medium):
        # set the substrate medium
        assert isinstance(medium, mp.Medium), \
            f"substrate medium must be a mp.Medium instance but got type {type(medium)}"
        medium: mp.Medium

        self.blk_sbstrt.material = medium
        self._sbstrt_mdm = medium

        if self.init_finished:
            self.redef_sim()

    def plot2D(self):
        """
        call sim.plot2D() to visualize the supercell
        """

        self.sim.plot2D()

    def _initialize_E_and_H_lists(self):
        self.E = []
        self.H = []

    def get_band(self, which_band):
        """
        :param which_band: which_band
        :return: E, H arrays
        """
        return self.E[which_band::self.num_bands], self.H[which_band::self.num_bands]

    def plot_mode(self, which_band, which_index_k):
        assert which_band < self.num_bands, \
            f"which_band must be <= {self.num_bands - 1} but got {which_band}"

        E, H = self.get_band(which_band)
        eps = self.ms.get_epsilon()

        for n, title in enumerate(['Ex', 'Ey', 'Ez']):
            plt.figure()
            x = E[which_index_k][:, :, n].__abs__() ** 2
            plt.imshow(eps[::-1, ::-1].T, interpolation='spline36', cmap='binary')
            plt.imshow(x[::-1, ::-1].T, cmap='RdBu', alpha=0.9)
            plt.axis(False)
            plt.title(title)

        for n, title in enumerate(['Hx', 'Hy', 'Hz']):
            plt.figure()
            x = H[which_index_k][:, :, n].__abs__() ** 2
            plt.imshow(eps[::-1, ::-1].T, interpolation='spline36', cmap='binary')
            plt.imshow(x[::-1, ::-1].T, cmap='RdBu', alpha=0.9)
            plt.axis(False)
            plt.title(title)

    def calc_dispersion(self, wl_min, wl_max, NPTS):
        """
        :param wl_min:
        :param wl_max:
        :param NPTS:
        :return:
        """

        if self.store_fields:
            self._initialize_E_and_H_lists()
            band_func = lambda ms, which_band: store_fields(ms, which_band, self)
            self.band_funcs = [band_func]

        else:
            self.band_funcs = []

        k_min, k_max = 1 / wl_max, 1 / wl_min
        f_center = (k_max - k_min) / 2 + k_min
        self.blk_wvgd.material = mp.Medium(epsilon_diag=self.wvgd_mdm.epsilon(f_center).diagonal())
        self.blk_sbstrt.material = mp.Medium(epsilon_diag=self.sbstrt_mdm.epsilon(f_center).diagonal())

        start = time.time()

        res = self.calc_w_from_k(wl_min, wl_max, NPTS)
        spl = UnivariateSpline(res.freq[:, 0], res.kx, s=0)

        # fields were stored by _calc_non_dispersive
        if self.store_fields:
            self._initialize_E_and_H_lists()

        OMEGA = np.linspace(k_min, k_max, NPTS)
        KX = []
        for n, omega in enumerate(OMEGA):
            self.blk_wvgd.material = mp.Medium(epsilon_diag=self.wvgd_mdm.epsilon(omega).diagonal())
            self.blk_sbstrt.material = mp.Medium(epsilon_diag=self.sbstrt_mdm.epsilon(omega).diagonal())

            kmag_guess = float(spl(omega))
            kx = self.find_k(mp.EVEN_Y, omega, 1, self.num_bands, mp.Vector3(1), 1e-4,
                             kmag_guess, kmag_guess * 0.1, kmag_guess * 10, *self.band_funcs)
            KX.append(kx)

            print(f'___________________________________{len(OMEGA) - n}__________________________________________')
        stop = time.time()
        print(f"finished in {(stop - start) / 60} minutes")

        if self.store_fields:
            self.E = np.squeeze(np.array(self.E))
            self.H = np.squeeze(np.array(self.H))

        class results:
            def __init__(self):
                parent: RidgeWaveguide
                self.kx = np.array(KX)
                self.freq = OMEGA

            def plot_dispersion(self):
                plt.figure()
                [plt.plot(self.kx[:, n], self.freq, '.-') for n in range(self.kx.shape[1])]
                plt.xlabel("k ($\mathrm{1/ \mu m}$)")
                plt.ylabel("$\mathrm{\\nu}$ ($\mathrm{1/ \mu m}$)")

        return results()

    def calc_w_from_k(self, wl_min, wl_max, NPTS):
        """
        :param wl_min: shortest wavelength
        :param wl_max: longest wavelength
        :param NPTS: number of k_points to interpolate from shortest -> longest wavelength
        :return: result instance with attributes kx (shape: kx), freq (shape: kx, num_bands)
        """

        k_points = mp.interpolate(NPTS, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])
        self.ms.k_points = k_points
        self.run(*self.band_funcs)

        # ____________________________________ Done ___________________________________________

        class results:
            def __init__(self, parent):
                parent: RidgeWaveguide
                self.kx = np.array([i.x for i in k_points])
                self.freq = parent.ms.all_freqs

            def plot_dispersion(self):
                plt.figure()
                [plt.plot(self.kx, self.freq[:, n], '.-') for n in range(self.freq.shape[1])]
                plt.xlabel("k ($\mathrm{\mu m}$)")
                plt.ylabel("$\mathrm{\\nu}$ ($\mathrm{\mu m}$)")

        if self.store_fields:
            self.E = np.squeeze(np.array(self.E))
            self.H = np.squeeze(np.array(self.H))

        return results(self)

    def find_k(self, p, omega, band_min, band_max, korig_and_kdir, tol,
               kmag_guess, kmag_min, kmag_max, *band_funcs):
        """
        :param p: parity
        :param omega: frequency
        :param band_min: minimum band index
        :param band_max: maximum band index
        :param korig_and_kdir: k direction (unit vector)
        :param tol: tolerance
        :param kmag_guess: guess for the wave-vector magnitude (n/lambda)
        :param kmag_min: minimum wave-vector magnitude
        :param kmag_max: maximum wave-vector magnitude
        :param band_funcs: additional arguments to pass to ms.find_k()
        :return: k (list of float(s))
        """

        args = [p, omega, band_min, band_max, korig_and_kdir, tol,
                kmag_guess, kmag_min, kmag_max, *band_funcs]

        if (self.wvgd_mdm.valid_freq_range[-1] == 1e20) and (self.sbstrt_mdm.valid_freq_range[-1] == 1e20):
            # materials are NON dispersive
            return self.ms.find_k(*args)

        else:
            # materials are dispersive
            # set the substrate and waveguide epsilon for the input wavelength
            # then run the simulation
            eps_wvgd = self.wvgd_mdm.epsilon(omega)
            eps_sbstrt = self.sbstrt_mdm.epsilon(omega)
            self.blk_wvgd.material = mp.Medium(epsilon_diag=eps_wvgd.diagonal())
            self.blk_sbstrt.material = mp.Medium(epsilon_diag=eps_sbstrt.diagonal())
            return self.ms.find_k(*args)
