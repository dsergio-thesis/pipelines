

# for col in [
        # 'sfr', 
        # 'sfr_UV', 
        # 'sfr_IR', 
        # 'ra', 
        # 'dec']:

keys = list(columns.keys())
for col in keys:
    if col not in ['z_best', 'lmass', 'lssfr', 'Av', 'beta']:
        columns.pop(col, None)
        df.pop(col)
