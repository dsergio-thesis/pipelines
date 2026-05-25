# Overwrite the "B" column with the values of "A" multiplied by 2
# a3.py does not modify the "B" column, so this will not trigger a merge conflict
df["B"] = df["A"] * 2 

# Overwrite the "A" column with the values of "A" divided by 2
# a3.py also modifies the "A" column, this will trigger a merge conflict
df["A"] = df["A"] / 2

