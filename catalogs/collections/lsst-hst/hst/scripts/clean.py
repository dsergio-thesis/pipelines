import numpy as np
import pandas as pd

print(f"*** 3D-HST Data Cleaning ***")

bad_values = {
    "Av": [-1],              # Invalid extinction
    "L_IR": [-99],           # Missing luminosity
    "beta": [-99],           # Missing UV slope
    "chi2": [-1],            # Invalid fit statistic
    "sfr": [-99],            # Missing SFR
    "sfr_IR": [-99],         # Missing IR SFR
    "sfr_UV": [-99],         # Missing UV SFR
    "z_best": [-99],         # Missing redshift
    "z_peak_grism": [-1],    # Missing grism redshift
    "z_peak_phot": [-99],    # Missing photometric redshift
    "z_spec": [-99.9],       # Missing spectroscopic redshift
}
for col, bad_vals in bad_values.items():
    if col in df.columns:
        df[col] = df[col].replace(bad_vals, np.nan) # replace bad values with nan

numeric_cols = [
    "Av", "L_IR", "beta", "chi2", "dec", "jh_mag", "lmass",
    "ra", "sfr", "sfr_IR", "sfr_UV", "lssfr",
    "z_best", "z_peak_grism", "z_peak_phot", "z_spec"
]

# convert to numeric, set non-convertible values to nan
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") 

# Av cannot be negative, set to nan
if "Av" in df.columns:
    df.loc[df["Av"] < 0, "Av"] = np.nan 

# redshift cannot be negative, set to nan
for col in ["z_best", "z_peak_grism", "z_peak_phot", "z_spec"]:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan 

# SFR and L_IR cannot be negative or zero, so set to nan
for col in ["sfr", "sfr_IR", "sfr_UV", "L_IR"]:
    if col in df.columns:
        df.loc[df[col] <= 0, col] = np.nan 

if "ra" in df.columns:
    df.loc[~df["ra"].between(0, 360), "ra"] = np.nan

if "dec" in df.columns:
    df.loc[~df["dec"].between(-90, 90), "dec"] = np.nan

all_nan_cols = df.columns[df.isna().all()].tolist()

# drop columns that are all nan after cleaning
df.drop(columns=all_nan_cols, inplace=True) 
print(f"Dropped {len(all_nan_cols)} columns that were all NaN after cleaning: {all_nan_cols}")
for col in all_nan_cols:
    if col in columns:
        columns.pop(col, None) # remove all-nan columns from columns dict for next node

# add log10 versions of SFR and L_IR, set to nan if original value is not positive
# add log versions of SFR and L_IR to the columns dict for the next node
for col in ["L_IR", "sfr", "sfr_IR", "sfr_UV"]:
    if col in df.columns:
        log_col = f"log10_{col}"
        df[log_col] = np.where(df[col] > 0, np.log10(df[col]), np.nan) 
        columns[log_col] = f"log10 {col}" 


