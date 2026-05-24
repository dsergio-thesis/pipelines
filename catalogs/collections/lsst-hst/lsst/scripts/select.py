
import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm import tqdm


"""
features per band: 
    - psf and cModel magnitudes (2 features) 
    - psf and cModel SNR (2 features, log-scaled)

15 color features (using cModel mags, but could also do PSF mags or both):
    - u-g color (mag_u - mag_g)
    - u-r color (mag_u - mag_r)
    - u-i color (mag_u - mag_i)
    - u-z color (mag_u - mag_z)
    - u-y color (mag_u - mag_y)

    - g-r color (mag_g - mag_r)
    - g-i color (mag_g - mag_i)
    - g-z color (mag_g - mag_z)
    - g-y color (mag_g - mag_y)

    - r-i color (mag_r - mag_i)
    - r-z color (mag_r - mag_z)
    - r-y color (mag_r - mag_y)
    
    - i-z color (mag_i - mag_z)
    - i-y color (mag_i - mag_y)

    - z-y color (mag_z - mag_y)

4 Adjacent curvatures:
    - curv_ug_gr = (mag_u - mag_g) - (mag_g - mag_r) = mag_u - 2*mag_g + mag_r
    - curv_gr_ri = (mag_g - mag_r) - (mag_r - mag_i) = mag_g - 2*mag_r + mag_i
    - curv_ri_iz = (mag_r - mag_i) - (mag_i - mag_z) = mag_r - 2*mag_i + mag_z
    - curv_iz_zy = (mag_i - mag_z) - (mag_z - mag_y) = mag_i - 2*mag_z + mag_y

Next: add diff/ratio of PSF and cModel for extendedness 

"""

# print("select...")
# print(df)

df_clean = pd.DataFrame()  # will hold cleaned data with new features
df_clean['objectId'] = df['objectId']
df_clean['ra'] = df['coord_ra']
df_clean['dec'] = df['coord_dec']
df_clean['tract'] = df['tract']
df_clean['patch'] = df['patch']
df_clean['detect_fromBlend'] = df['detect_fromBlend']
df_clean['detect_isIsolated'] = df['detect_isIsolated']
df_clean['refExtendedness'] = df['refExtendedness']
df_clean['color_gr'] = [np.nan] * len(df)
df_clean['color_ri'] = [np.nan] * len(df)
df_clean['color_iz'] = [np.nan] * len(df)
for band in ['u', 'g', 'r', 'i', 'z', 'y']:
    df_clean[f"{band}_psfFlux_SNR_log"] = [np.nan] * len(df)
    df_clean[f"{band}_psfFlux_mag"] = [np.nan] * len(df)
    df_clean[f"{band}_psfFlux_bad_flag"] = [np.nan] * len(df)

    df_clean[f"{band}_free_cModelFlux_SNR_log"] = [np.nan] * len(df)
    df_clean[f"{band}_free_cModelFlux_mag"] = [np.nan] * len(df)
    df_clean[f"{band}_free_cModelFlux_bad_flag"] = [np.nan] * len(df)

n = len(df)
print(f"Feature preprocesing for {n} objects...")

def flux_to_mag(flux):
    return -2.5 * np.log10(flux) + 31.4

bands = ['u', 'g', 'r', 'i', 'z', 'y']

label_counts = dict()

num_bands = len(bands)


for row in tqdm(df.itertuples(), total=n, desc="Extracting Photometric Features"):
    target_ra = row.coord_ra
    target_dec = row.coord_dec

    if hasattr(row, "label"):
        if (str(row.label) in label_counts):
            label_counts[str(row.label)] += 1
        else:
            label_counts[str(row.label)] = 1
   

    psf_mag_u = None
    psf_mag_u_flag = True
    psf_mag_g = None
    psf_mag_g_flag = True
    psf_mag_r = None
    psf_mag_r_flag = True
    psf_mag_i = None
    psf_mag_i_flag = True
    psf_mag_z = None
    psf_mag_z_flag = True
    psf_mag_y = None
    psf_mag_y_flag = True

    cmodel_mag_u = None
    cmodel_mag_u_flag = True
    cmodel_mag_g = None
    cmodel_mag_g_flag = True
    cmodel_mag_r = None
    cmodel_mag_r_flag = True
    cmodel_mag_i = None
    cmodel_mag_i_flag = True
    cmodel_mag_z = None
    cmodel_mag_z_flag = True
    cmodel_mag_y = None
    cmodel_mag_y_flag = True

    psf_bad = 0.0
    cmodel_bad = 0.0

    for bi, band in enumerate(bands):
        psf_flux = getattr(row, f"{band}_psfFlux", None)
        psf_err  = getattr(row, f"{band}_psfFluxErr", None)
        psf_flag = getattr(row, f"{band}_psfFlux_flag", False)

        cmodel_flux = getattr(row, f"{band}_free_cModelFlux", None)
        cmodel_err = getattr(row, f"{band}_free_cModelFluxErr", None)
        cmodel_flag = getattr(row, f"{band}_free_cModelFlux_flag", False)

        psf_mag = flux_to_mag(psf_flux) if psf_flux is not None and psf_flux > 0 else np.nan
        cmodel_mag = flux_to_mag(cmodel_flux) if cmodel_flux is not None and cmodel_flux > 0 else np.nan
        
        # sanitize missing/NaN
        invalid = (
            psf_flux is None
            or psf_err is None
            or cmodel_flux is None
            or cmodel_err is None
            or pd.isna(psf_flux)
            or pd.isna(psf_err)
            or pd.isna(psf_mag)
            or pd.isna(cmodel_mag)
        )

        if invalid:
            psf_flux = 0.0
            psf_err = 0.0
            psf_mag = np.nan 
            psf_snr = 0.0
            psf_bad = 1.0  # treat missing as bad
            cmodel_flux = 0.0
            cmodel_err = 0.0
            cmodel_mag = np.nan 
            cmodel_snr = 0.0
            cmodel_bad = 1.0  # treat missing as bad
            
        else:

            # SNR feature (clamp to non-negative)
            if psf_err > 0:
                psf_snr = float(psf_flux) / float(psf_err)
                psf_snr = np.log1p(max(0.0, psf_snr))
            else:
                psf_snr = 0.0

            if cmodel_err > 0:
                cmodel_snr = float(cmodel_flux) / float(cmodel_err)
                cmodel_snr = np.log1p(max(0.0, cmodel_snr))
            else:
                cmodel_snr = 0.0

            if band == 'g':
                psf_mag_g = psf_mag
                psf_mag_g_flag = psf_flag
                cmodel_mag_g = cmodel_mag
                cmodel_mag_g_flag = cmodel_flag
            elif band == 'r':
                psf_mag_r = psf_mag
                psf_mag_r_flag = psf_flag
                cmodel_mag_r = cmodel_mag
                cmodel_mag_r_flag = cmodel_flag
            elif band == 'i':
                psf_mag_i = psf_mag
                psf_mag_i_flag = psf_flag
                cmodel_mag_i = cmodel_mag
                cmodel_mag_i_flag = cmodel_flag
            elif band == 'u':
                psf_mag_u = psf_mag
                psf_mag_u_flag = psf_flag
                cmodel_mag_u = cmodel_mag
                cmodel_mag_u_flag = cmodel_flag
            elif band == 'z':
                psf_mag_z = psf_mag
                psf_mag_z_flag = psf_flag
                cmodel_mag_z = cmodel_mag
                cmodel_mag_z_flag = cmodel_flag
            elif band == 'y':
                psf_mag_y = psf_mag
                psf_mag_y_flag = psf_flag
                cmodel_mag_y = cmodel_mag
                cmodel_mag_y_flag = cmodel_flag
                
        df_clean.at[row.Index, f"{band}_psfFlux_mag"] = psf_mag
        df_clean.at[row.Index, f"{band}_psfFlux_SNR_log"] = psf_snr        
        df_clean.at[row.Index, f"{band}_cModelFlux_mag"] = cmodel_mag
        df_clean.at[row.Index, f"{band}_cModelFlux_SNR_log"] = cmodel_snr

        df_clean.at[row.Index, f"{band}_psfFlux_bad_flag"] = psf_bad
        df_clean.at[row.Index, f"{band}_cModelFlux_bad_flag"] = cmodel_bad

    psf_flux_mags = {
        "u": psf_mag_u,
        "g": psf_mag_g,
        "r": psf_mag_r,
        "i": psf_mag_i,
        "z": psf_mag_z,
        "y": psf_mag_y,
    }
    cmodel_flux_mags = {
        "u": cmodel_mag_u,
        "g": cmodel_mag_g,
        "r": cmodel_mag_r,
        "i": cmodel_mag_i,
        "z": cmodel_mag_z,
        "y": cmodel_mag_y,
    }
    psf_flux_flags = {
        "u": psf_mag_u_flag,
        "g": psf_mag_g_flag,
        "r": psf_mag_r_flag,
        "i": psf_mag_i_flag,
        "z": psf_mag_z_flag,
        "y": psf_mag_y_flag,
    }
    cmodel_flux_flags = {
        "u": cmodel_mag_u_flag,
        "g": cmodel_mag_g_flag,
        "r": cmodel_mag_r_flag,
        "i": cmodel_mag_i_flag,
        "z": cmodel_mag_z_flag,
        "y": cmodel_mag_y_flag,
    }

    def psf_color(b1, b2):
        m1, m2 = psf_flux_mags[b1], psf_flux_mags[b2]
        f1, f2 = psf_flux_flags[b1], psf_flux_flags[b2]
        return m1 - m2 if (m1 is not None and m2 is not None and not f1 and not f2) else np.nan
    def cmodel_color(b1, b2):
        m1, m2 = cmodel_flux_mags[b1], cmodel_flux_mags[b2]
        f1, f2 = cmodel_flux_flags[b1], cmodel_flux_flags[b2]
        return m1 - m2 if (m1 is not None and m2 is not None and not f1 and not f2) else np.nan
    
    colors = {}
    colors['ug'] = cmodel_color('u', 'g')
    colors['ur'] = cmodel_color('u', 'r')
    colors['ui'] = cmodel_color('u', 'i')
    colors['uz'] = cmodel_color('u', 'z')
    colors['uy'] = cmodel_color('u', 'y')
    colors['gr'] = cmodel_color('g', 'r')
    colors['gi'] = cmodel_color('g', 'i')
    colors['gz'] = cmodel_color('g', 'z')
    colors['gy'] = cmodel_color('g', 'y')
    colors['ri'] = cmodel_color('r', 'i')
    colors['rz'] = cmodel_color('r', 'z')
    colors['ry'] = cmodel_color('r', 'y')
    colors['iz'] = cmodel_color('i', 'z')
    colors['iy'] = cmodel_color('i', 'y')
    colors['zy'] = cmodel_color('z', 'y')

    curvatures = {}
    curvatures['ug_gr'] = (colors['ug'] - colors['gr']) if (not np.isnan(colors['ug']) and not np.isnan(colors['gr'])) else np.nan
    curvatures['gr_ri'] = (colors['gr'] - colors['ri']) if (not np.isnan(colors['gr']) and not np.isnan(colors['ri'])) else np.nan
    curvatures['ri_iz'] = (colors['ri'] - colors['iz']) if (not np.isnan(colors['ri']) and not np.isnan(colors['iz'])) else np.nan
    curvatures['iz_zy'] = (colors['iz'] - colors['zy']) if (not np.isnan(colors['iz']) and not np.isnan(colors['zy'])) else np.nan

    for color_name, color_value in colors.items():
        df_clean.at[row.Index, f"color_{color_name}"] = color_value
    for curv_name, curv_value in curvatures.items():
        df_clean.at[row.Index, f"curvature_{curv_name}"] = curv_value


# replace df with df_clean for output
df.drop(df.index, inplace=True)

for col in df.columns:
    del df[col]

for col in df_clean.columns:
    df[col] = df_clean[col].values

df.index = df_clean.index


columns.update({
    "objectId": "objectId",
    "ra": "Right Ascension (degrees)",
    "dec": "Declination (degrees)",
    "tract": "LSST Tract Number",
    "patch": "LSST Patch Number",
    "detect_fromBlend": "Detection from Blend Flag",
    "detect_isIsolated": "Detection Isolated Flag",
    "refExtendedness": "Reference Extendedness (0=point-like, 1=extended)",
    "label": "Object Class Label (if available)",
    "u_psfFlux_SNR_log": "Log SNR of u-band PSF Flux",
    "u_psfFlux_mag": "u-band PSF Magnitude",
    "g_psfFlux_SNR_log": "Log SNR of g-band PSF Flux",
    "g_psfFlux_mag": "g-band PSF Magnitude",
    "r_psfFlux_SNR_log": "Log SNR of r-band PSF Flux",
    "r_psfFlux_mag": "r-band PSF Magnitude",
    "i_psfFlux_SNR_log": "Log SNR of i-band PSF Flux",
    "i_psfFlux_mag": "i-band PSF Magnitude",
    "z_psfFlux_SNR_log": "Log SNR of z-band PSF Flux",
    "z_psfFlux_mag": "z-band PSF Magnitude",
    "y_psfFlux_SNR_log": "Log SNR of y-band PSF Flux",
    "y_psfFlux_mag": "y-band PSF Magnitude",
    "u_cModelFlux_SNR_log": "Log SNR of u-band cModel Flux",
    "u_cModelFlux_mag": "u-band cModel Magnitude",
    "g_cModelFlux_SNR_log": "Log SNR of g-band cModel Flux",
    "g_cModelFlux_mag": "g-band cModel Magnitude",
    "r_cModelFlux_SNR_log": "Log SNR of r-band cModel Flux",
    "r_cModelFlux_mag": "r-band cModel Magnitude",
    "i_cModelFlux_SNR_log": "Log SNR of i-band cModel Flux",
    "i_cModelFlux_mag": "i-band cModel Magnitude",
    "z_cModelFlux_SNR_log": "Log SNR of z-band cModel Flux",
    "z_cModelFlux_mag": "z-band cModel Magnitude",
    "y_cModelFlux_SNR_log": "Log SNR of y-band cModel Flux",
    "y_cModelFlux_mag": "y-band cModel Magnitude",
    "color_ug": "u-g Color (cModel mag)",
    "color_ur": "u-r Color (cModel mag)",
    "color_ui": "u-i Color (cModel mag)",
    "color_uz": "u-z Color (cModel mag)",
    "color_uy": "u-y Color (cModel mag)",
    "color_gr": "g-r Color (cModel mag)",
    "color_gi": "g-i Color (cModel mag)",
    "color_gz": "g-z Color (cModel mag)",
    "color_gy": "g-y Color (cModel mag)",
    "color_ri": "r-i Color (cModel mag)",
    "color_rz": "r-z Color (cModel mag)",
    "color_ry": "r-y Color (cModel mag)",
    "color_iz": "i-z Color (cModel mag)",
    "color_iy": "i-y Color (cModel mag)",
    "color_zy": "z-y Color (cModel mag)",
    "curvature_ug_gr": "Curvature between u-g and g-r colors",
    "curvature_gr_ri": "Curvature between g-r and r-i colors",
    "curvature_ri_iz": "Curvature between r-i and i-z colors",
    "curvature_iz_zy": "Curvature between i-z and z-y colors",
})





