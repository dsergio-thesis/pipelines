
keys = list(columns.keys())

# print(f"searching {keys}")
for col in keys:
    if col not in [
            'g_psfFlux_mag', 
            'r_psfFlux_mag', 
            'i_psfFlux_mag', 
            'g_free_cModelFlux_mag',
            'r_free_cModelFlux_mag',
            'i_free_cModelFlux_mag',
            'g_psfFlux_SNR_log',
            'r_psfFlux_SNR_log',
            'i_psfFlux_SNR_log',
            'g_free_cModelFlux_SNR_log',
            'r_free_cModelFlux_SNR_log',
            'i_free_cModelFlux_SNR_log',
            'refExtendedness',
            ]:
        columns.pop(col, None)
        if col in df.columns:
            # print(f"Removing {col} from histogram_select")
            df.pop(col)

