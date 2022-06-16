# source = mp.EigenModeSource(src=src,
#                             center=pt_src,
#                             size=mp.Vector3(0, sy - 2 * dpml),
#                             eig_match_freq=True,
#                             eig_band=1,
#                             eig_parity=mp.EVEN_Y + mp.ODD_Z
#                             )

# mon_pt = mp.Vector3(-0.5 * sx + dpml + 5)
# flux_reg = mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy - 2 * dpml))
# flux = sim.add_flux(1 / wl_src, df_src, 100, flux_reg)

# run the sim ...

# res = sim.get_eigenmode_coefficients(flux, [1])
# incident_coeffs = res.alpha
# incident_flux = mp.get_fluxes(flux)
# incident_flux_data = sim.get_flux_data(flux)

# %% ___________________________________________________________________________________________________________________
# was using this to identify Ez single-mode
# def _get_cut_for_sm(self, which_band, k_index):
#     mode = self.E[k_index][which_band][:, :, 2].__abs__() ** 2
#     resolution = self.resolution[0]
#
#     wvgd_width = self.width * resolution
#     wvgd_height = self.height * resolution
#     cell_width = self.cell_width * resolution
#     cell_height = self.cell_height * resolution
#     edge_side_h = (cell_width - wvgd_width) / 2
#     edge_side_v = (cell_height - wvgd_height) / 2
#
#     # for some reason, the epsilon grid can be slightly different from
#     # what I expect, so scale accordingly to get the right waveguide indices
#     # on the simulation grid
#     eps = self.ms.get_epsilon()
#     h_factor = eps.shape[0] / cell_width
#     v_factor = eps.shape[1] / cell_height
#
#     edge_side_h = int(np.round(edge_side_h * h_factor))
#     edge_side_v = int(np.round(edge_side_v * v_factor))
#     cell_width = int(np.round(cell_width * h_factor))
#     cell_height = int(np.round(cell_height * v_factor))
#
#     h_center = np.fft.fftshift(mode.T, 0)[0][edge_side_h:cell_width - edge_side_h]
#     v_center = np.fft.fftshift(mode, 0)[0][edge_side_v:cell_height - edge_side_v]
#
#     return h_center, v_center
#
# def _index_rank_sm(self, k_index):
#     IND = []
#     for band in range(self.num_bands):
#         cnt_h, cnt_v = self._get_cut_for_sm(band, k_index)
#
#         # ___________________________________________________________________________
#         # fft, and look at first component after DC
#         # (or else DC component will always beat out everything else)
#         ind_cnt_h = len(cnt_h) // 2
#         ind_cnt_v = len(cnt_v) // 2
#         ft_h = fft(cnt_h)[ind_cnt_h:][1:].__abs__()
#         ft_v = fft(cnt_v)[ind_cnt_v:][1:].__abs__()
#         if any([np.argmax(ft_h) != 0, np.argmax(ft_v) != 0]):
#             ind = 0
#         else:
#             ind = scint.simps(cnt_h) + scint.simps(cnt_v)
#         # ___________________________________________________________________________
#
#         IND.append(ind)
#
#     return np.array(IND)
#
# def get_sm_band_at_k_index(self, k_index):
#     return np.argmax(self._index_rank_sm(k_index))
#
# def get_sm_band_for_k_axis(self, kx):
#     band = np.array([self.get_sm_band_at_k_index(i) for i in range(len(kx))])
#     return band
