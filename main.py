import pandas as pd
from visualize import histogram, boxplot, histogram_overlay_by_chd, scatter_3d_combinations, scatter_matrix_subset
from pca import plot_pc_lines_3d, make_Xy, fit_pca, plot_scores_2d, plot_explained_variance

url = "https://www.hastie.su.domains/Datasets/SAheart.data"

# Read like the R call (comma-sep, header row, first col = row names)
df = pd.read_csv(url, sep=",", header=0, index_col=0, skipinitialspace=True)

'''if df["famhist"].dtype != "O":  # numeric -> map to strings
    df["famhist"] = df["famhist"].map({0: "Absent", 1: "Present"}).astype("category")'''


x = df.drop(columns=["chd"])
y = pd.Categorical(df["chd"])

#histogram(df, bins=20, ncols=3, title="Histograms", by_chd=0)
#histogram_overlay_by_chd(df, features=None, ncols=3)

#boxplot(df)
scatter_matrix_subset(df, [0,1,2,3,8], label="chd", bins=20, alpha=0.5, s=15)
scatter_3d_combinations(df, features=[0,1,2,3,8], label_col="chd", use_indices=True, ncols=2, s=10)

X, y, feature_names = make_Xy(df, target="chd")

# Fit PCA on standardized X
scaler, pca, Xz, B, V, sing_vals = fit_pca(X, n_components=None)

# Plots
# (a) 3D cloud of first 3 original standardized dims + PC directions, and 2D scores
#plot_pc_lines_3d(Xz, V, B, K=2, elev=12, azim=-150)

# (b) 2D scores (PC1 vs PC2) colored by CHD
#plot_scores_2d(B, y=y, title="SAheart: PC1 vs PC2 (colored by CHD)")

# (c) Explained variance
#plot_explained_variance(df, pca, M_label="SAheart PCA")