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

# %%
