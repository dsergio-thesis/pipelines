
import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm import tqdm


"""
3 features per band: 
    - flux Transformed (arcsinh)
    - x err Transformed (arcsinh)
    - log SNR (clamped to 0 if err=0)
    - mag (from flux, with safe handling of zero/negative flux)
    - x bad-flag (1 if any issues with flux/err, else 0)

15 color features:
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

Next: add diff/ratio PSF and cModel for morphology

"""

print("select...")
print(df)

df_clean = pd.DataFrame()  # will hold cleaned data with new features
df_clean['objectId'] = df['objectId']
df_clean['ra'] = df['coord_ra']
df_clean['dec'] = df['coord_dec']
df_clean['tract'] = df['tract']
df_clean['patch'] = df['patch']
df_clean['detect_fromBlend'] = df['detect_fromBlend']
df_clean['detect_isIsolated'] = df['detect_isIsolated']
df_clean['refExtendedness'] = df['refExtendedness']
# df_clean['label'] = df['label'] if 'label' in df.columns else [np.nan] * len(df)
df_clean['color_gr'] = [np.nan] * len(df)
df_clean['color_ri'] = [np.nan] * len(df)
df_clean['color_iz'] = [np.nan] * len(df)
for band in ['u', 'g', 'r', 'i', 'z', 'y']:
    df_clean[f"{band}_psfFlux_arcsinh"] = [np.nan] * len(df)
    df_clean[f"{band}_psfFluxErr_arcsinh"] = [np.nan] * len(df)
    df_clean[f"{band}_psfFlux_SNR_log"] = [np.nan] * len(df)
    df_clean[f"{band}_psfFlux_mag"] = [np.nan] * len(df)
    df_clean[f"{band}_psfFlux_bad_flag"] = [np.nan] * len(df)

    df_clean[f"{band}_free_cModelFlux_arcsinh"] = [np.nan] * len(df)
    df_clean[f"{band}_free_cModelFluxErr_arcsinh"] = [np.nan] * len(df)
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

# precompute safe scales (dataset-level)
# flux_scale = self.median_r_psfFlux if getattr(self, "median_r_psfFlux", 0) and self.median_r_psfFlux > 0 else 1.0
# err_scale  = self.median_r_psfFluxErr if getattr(self, "median_r_psfFluxErr", 0) and self.median_r_psfFluxErr > 0 else 1.0
flux_scale = 1.0
err_scale = 1.0

for row in tqdm(df.itertuples(), total=n, desc="Extracting Photometric Features"):
    target_ra = row.coord_ra
    target_dec = row.coord_dec

    if hasattr(row, "label"):
        if (str(row.label) in label_counts):
            # print(f"found label {str(row.label)}, adding to existing counts")
            label_counts[str(row.label)] += 1
        else:
            # print(f"found label {str(row.label)}, setting count to 1")
            label_counts[str(row.label)] = 1

    photometric_features = np.zeros((num_bands, 3), dtype=np.float32)
   

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

    for bi, band in enumerate(bands):
        psf_flux = getattr(row, f"{band}_psfFlux", None)
        psf_err  = getattr(row, f"{band}_psfFluxErr", None)
        psf_flag = getattr(row, f"{band}_psfFlux_flag", False)

        cModel_flux = getattr(row, f"{band}_free_cModelFlux", None)
        cModel_err = getattr(row, f"{band}_free_cModelFluxErr", None)
        cModel_flag = getattr(row, f"{band}_free_cModelFlux_flag", False)

        mag = flux_to_mag(flux) if flux is not None and flux > 0 else np.nan
        mag_model = flux_to_mag(model_flux) if model_flux is not None and model_flux > 0 else np.nan
        
        # sanitize missing/NaN
        if (flux is None 
            or err is None 
            or flag is None
            or model_flux is None
            or model_err is None
            or model_flag is None
            or (isinstance(flux, float) and np.isnan(flux)) 
            or (isinstance(err, float) and np.isnan(err)
            or (isinstance(mag, float) and np.isnan(mag))
            or (isinstance(mag_model, float) and np.isnan(mag_model))
                ):
            psf_x1 = 0.0
            psf_x2 = 0.0
            psf_x3 = 0.0
            psf_x4 = 0.0
            psf_bad = 1.0  # treat missing as bad
            cmag_x1 = 0.0
            cmag_x2 = 0.0
            cmag_x3 = 0.0
            cmag_x4 = 0.0
            cmag_bad = 1.0  # treat missing as bad
            
        else:
            # arcsinh scaling 
            psf_x1 = np.arcsinh(float(flux) / flux_scale)
            psf_x2 = np.arcsinh(float(err) / err_scale)

            # SNR feature (clamp to non-negative)
            if err > 0:
                snr = float(flux) / float(err)
                psf_x3 = np.log1p(max(0.0, snr))
            else:
                psf_x3 = 0.0

            psf_x4 = mag
            if band == 'g':
                mag_g = mag
                mag_g_flag = flag
            elif band == 'r':
                mag_r = mag
                mag_r_flag = flag
            elif band == 'i':
                mag_i = mag
                mag_i_flag = flag
            elif band == 'z':
                mag_z = mag
                mag_z_flag = flag
            elif band == 'u':
                mag_u = mag
                mag_u_flag = flag
            elif band == 'y':
                mag_y = mag
                mag_y_flag = flag

            psf_bad = 1.0 if bool(flag) else 0.0

        photometric_features[bi] = (x1, x3, x4)

        df_clean.at[row.Index, f"{band}_psfFlux_arcsinh"] = x1
        # df_clean.at[row.Index, f"{band}_psfFluxErr_arcsinh"] = x2
        df_clean.at[row.Index, f"{band}_psfFlux_SNR_log"] = x3
        df_clean.at[row.Index, f"{band}_psfFlux_mag"] = x4
        # df_clean.at[row.Index, f"{band}_psfFlux_bad_flag"] = bad

    mags = {
        "u": mag_u,
        "g": mag_g,
        "r": mag_r,
        "i": mag_i,
        "z": mag_z,
        "y": mag_y,
    }

    flags = {
        "u": mag_u_flag,
        "g": mag_g_flag,
        "r": mag_r_flag,
        "i": mag_i_flag,
        "z": mag_z_flag,
        "y": mag_y_flag,
    }

    def color(b1, b2):
        m1, m2 = mags[b1], mags[b2]
        f1, f2 = flags[b1], flags[b2]
        return m1 - m2 if (m1 is not None and m2 is not None and not f1 and not f2) else np.nan
    
    colors = {}
    colors['ug'] = color('u', 'g')
    colors['ur'] = color('u', 'r')
    colors['ui'] = color('u', 'i')
    colors['uz'] = color('u', 'z')
    colors['uy'] = color('u', 'y')
    colors['gr'] = color('g', 'r')
    colors['gi'] = color('g', 'i')
    colors['gz'] = color('g', 'z')
    colors['gy'] = color('g', 'y')
    colors['ri'] = color('r', 'i')
    colors['rz'] = color('r', 'z')
    colors['ry'] = color('r', 'y')
    colors['iz'] = color('i', 'z')
    colors['iy'] = color('i', 'y')
    colors['zy'] = color('z', 'y')

    curvatures = {}
    curvatures['ug_gr'] = (colors['ug'] - colors['gr']) if (not np.isnan(colors['ug']) and not np.isnan(colors['gr'])) else np.nan
    curvatures['gr_ri'] = (colors['gr'] - colors['ri']) if (not np.isnan(colors['gr']) and not np.isnan(colors['ri'])) else np.nan
    curvatures['ri_iz'] = (colors['ri'] - colors['iz']) if (not np.isnan(colors['ri']) and not np.isnan(colors['iz'])) else np.nan
    curvatures['iz_zy'] = (colors['iz'] - colors['zy']) if (not np.isnan(colors['iz']) and not np.isnan(colors['zy'])) else np.nan

    photometric_features = np.hstack([photometric_features.flatten(), 
        [color for color in colors.values()] + 
        [curv for curv in curvatures.values()]
                                      ])
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
    "color_gr": "g-r Color (mag)",
    "color_ri": "r-i Color (mag)",
    "color_iz": "i-z Color (mag)",
    "u_psfFlux_arcsinh": "Arcsinh Transformed u-band PSF Flux",
    "u_psfFlux_SNR_log": "Log SNR of u-band PSF Flux",
    "u_psfFlux_mag": "u-band PSF Magnitude",
    "g_psfFlux_arcsinh": "Arcsinh Transformed g-band PSF Flux",
    "g_psfFlux_SNR_log": "Log SNR of g-band PSF Flux",
    "g_psfFlux_mag": "g-band PSF Magnitude",
    "r_psfFlux_arcsinh": "Arcsinh Transformed r-band PSF Flux",
    "r_psfFlux_SNR_log": "Log SNR of r-band PSF Flux",
    "r_psfFlux_mag": "r-band PSF Magnitude",
    "i_psfFlux_arcsinh": "Arcsinh Transformed i-band PSF Flux",
    "i_psfFlux_SNR_log": "Log SNR of i-band PSF Flux",
    "i_psfFlux_mag": "i-band PSF Magnitude",
    "z_psfFlux_arcsinh": "Arcsinh Transformed z-band PSF Flux",
    "z_psfFlux_SNR_log": "Log SNR of z-band PSF Flux",
    "z_psfFlux_mag": "z-band PSF Magnitude",
    "y_psfFlux_arcsinh": "Arcsinh Transformed y-band PSF Flux",
    "y_psfFlux_SNR_log": "Log SNR of y-band PSF Flux",
    "y_psfFlux_mag": "y-band PSF Magnitude",
    "color_ug": "u-g Color (mag)",
    "color_ur": "u-r Color (mag)",
    "color_ui": "u-i Color (mag)",
    "color_uz": "u-z Color (mag)",
    "color_uy": "u-y Color (mag)",
    "color_gr": "g-r Color (mag)",
    "color_gi": "g-i Color (mag)",
    "color_gz": "g-z Color (mag)",
    "color_gy": "g-y Color (mag)",
    "color_ri": "r-i Color (mag)",
    "color_rz": "r-z Color (mag)",
    "color_ry": "r-y Color (mag)",
    "color_iz": "i-z Color (mag)",
    "color_iy": "i-y Color (mag)",
    "color_zy": "z-y Color (mag)",
    "curvature_ug_gr": "Curvature between u-g and g-r colors",
    "curvature_gr_ri": "Curvature between g-r and r-i colors",
    "curvature_ri_iz": "Curvature between r-i and i-z colors",
    "curvature_iz_zy": "Curvature between i-z and z-y colors",
})





