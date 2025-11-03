import pandas as pd
from ann_regression.ann import build_ann
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score, KFold


DATA_URL = "https://www.hastie.su.domains/Datasets/SAheart.data"

def load_saheart(url: str = DATA_URL) -> pd.DataFrame:
    """Load the SAheart dataset from the supplied URL."""
    df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)
    df["famhist"] = df["famhist"].astype("category")
    return df

def make_X_y(df: pd.DataFrame, targets: list[str]):
    """
    Prepare features X and target y based on chosen targets.
    - Drops the target(s) from X.
    - One-hot encodes 'famhist' (categorical).
    """
    y = df[targets].astype(float)
    X = df.drop(columns=targets).copy()

    # One-hot encode 'famhist' if categorical
    if "famhist" in X.columns and str(X["famhist"].dtype) == "category":
        X = pd.get_dummies(X, columns=["famhist"], drop_first=True)

    return X, y

def cv_mse_for(h: int, alpha: float, X, y, n_splits: int = 5, random_state: int = 0) -> float:
    """
    Return mean CV MSE for a single (h, alpha) using KFold.
    """
    model = build_ann(h=h, alpha=alpha, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # cross_val_score returns NEGATIVE MSE for scoring='neg_mean_squared_error'
    scores = cross_val_score(model, X, np.ravel(y), cv=kf, scoring="neg_mean_squared_error")
    return -scores.mean()


if __name__ == "__main__":
    df = load_saheart()
    X, y = make_X_y(df, targets=["ldl"])

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # sklearn wants 1D y for regression
    y_train_1d = np.ravel(y_train.values if hasattr(y_train, "values") else y_train)
    y_test_1d  = np.ravel(y_test.values if hasattr(y_test, "values") else y_test)

    # build a tiny ANN: h=4 hidden units, alpha=0.01 (L2)
    model = build_ann(h=4, alpha=0.01, random_state=0)
    model.fit(X_train, y_train_1d)

    # evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test_1d, y_pred)
    print(f"Sanity-check MSE: {mse:.4f}")

    print("\n--- Alpha grid for h=4 ---")
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]

    for a in alphas:
        cv_mse = cv_mse_for(h=4, alpha=a, X=X, y=y, n_splits=5, random_state=0)
        print(f"alpha={a:<6} | CV(5) MSE={cv_mse:.4f}")






