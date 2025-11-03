import pandas as pd 
from sklearn.neural_network import MLPRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def build_ann(h: int, alpha: float, random_state: int = 0) -> Pipeline:
    """
    Return a standardized MLP regressor with one hidden layer.
    h = number of hidden units (complexity)
    alpha = L2 regularization strength (λ)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(h,),
            alpha=alpha,          
            random_state=random_state,
            max_iter=2000
        ))
    ])
