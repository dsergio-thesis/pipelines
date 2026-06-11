import numpy as np

print("** Data Selection **")


def clip_outliers(series, lower=0.01, upper=0.99):
    """Clip a numeric series to the specified quantile range."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


# Clip extreme values for visualization and exploratory analysis.
df["beta"] = clip_outliers(df["beta"])      # Clip to 1st and 99th percentiles
df["L_IR"] = clip_outliers(df["L_IR"])      # Clip to 1st and 99th percentiles

# Remove extreme chi-square and lssfr outliers.
df = df[df["chi2"] < df["chi2"].quantile(0.99)]
df = df[df["lssfr"] > df["lssfr"].quantile(0.01)]

# Apply binary labels from specific star-formation rate.
if "lssfr" not in df.columns:
    raise ValueError("3D-HST catalog is missing required column 'lssfr'.")

lssfr = df["lssfr"].to_numpy(dtype=float)

valid = np.isfinite(lssfr)
valid &= (lssfr > -50) & (lssfr < 50)

# Label definition:
# 0 = star-forming
# 1 = quiescent
# -1 = invalid or unlabeled
label = np.full(len(df), -1, dtype=np.int64)
label[valid & (lssfr > -11)] = 0
label[valid & (lssfr < -11)] = 1

df["label"] = label

selected = (df["label"] >= 0).sum()
removed = (df["label"] < 0).sum()

print(f"Selected {selected} labeled galaxies for visualization.")
print(f"Star-forming: {(df['label'] == 0).sum()}, Quiescent: {(df['label'] == 1).sum()}")
print(f"Removed {removed} invalid or unlabeled galaxies.")
