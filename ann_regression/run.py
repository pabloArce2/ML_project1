import pandas as pd

DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"

def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    """Load the SAheart dataset from the supplied URL."""
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    return df
