
keys = list(columns.keys())
for col in keys:
    if col not in [
            'u_psfFlux_mag', 
            'g_psfFlux_mag', 
            'r_psfFlux_mag', 
            'i_psfFlux_mag', 
            'z_psfFlux_mag',
            'y_psfFlux_mag',
            'g_psfFlux_SNR_log',
            'r_psfFlux_SNR_log',
            'i_psfFlux_SNR_log',
            'color_ug',
            'color_gr',
            'color_ri',
            'color_iz',
            'refExtendedness',
            ]:
        columns.pop(col, None)
        df.pop(col)

