

print("** Selecting data for visualization **")

# for col in ["chi2", "L_IR", "sfr", "sfr_IR", "sfr_UV"]:
    # if col in df.columns:
        # extremes = df[col].quantile(0.9)
        # df[f"{col}_is_extreme"] = df[col] > extremes


def clip_outliers(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

df["beta"] = clip_outliers(df["beta"]) # beta has extreme outliers, so clip to 0.1 and 99.9 percentiles
df["L_IR"] = clip_outliers(df["L_IR"]) # L_IR has extreme outliers, so clip to 0.1 and 99.9 percentiles
df = df[df["chi2"] < df["chi2"].quantile(0.99)] # chi2 has extreme outliers, so remove top 1%
# remove outliers of lssfr
df = df[df["lssfr"] > df["lssfr"].quantile(0.05)] # remove bottom 5% of lssfr for visualization

