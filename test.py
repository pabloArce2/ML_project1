import pandas as pd

url = "https://www.hastie.su.domains/Datasets/SAheart.data"

# Read like the R call (comma-sep, header row, first col = row names)
df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)

if df["famhist"].dtype != "O":  # numeric -> map to strings
    df["famhist"] = df["famhist"].map({0: "Absent", 1: "Present"}).astype("category")


x = df.drop(columns=["chd"])
y = pd.Categorical(df["chd"])

print(x)