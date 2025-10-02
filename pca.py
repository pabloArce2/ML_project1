# pca_saheart.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# ----------------------------
# 2) Prepare matrix X and label y
# ----------------------------
def make_Xy(df, target="chd"):
    # numeric-only features for PCA (exclude target if numeric)
    num = df.select_dtypes(include=np.number)
    X = num.drop(columns=[target], errors="ignore").values
    y = df[target].values if target in df.columns else None
    feature_names = [c for c in num.columns if c != target]
    return X, y, feature_names

# ----------------------------
# 3) Standardize and PCA
# ----------------------------
def fit_pca(X, n_components=None):
    # standardize
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)  # None => all components
    B = pca.fit_transform(Xz)             # scores / projections
    V = pca.components_.T                 # columns = PCs
    sing_vals = np.sqrt(pca.explained_variance_ * (X.shape[0] - 1))  # σ_i
    return scaler, pca, Xz, B, V, sing_vals

# ----------------------------
# 4) Plots (3D PC lines, 2D PC scatter, explained variance)
# ----------------------------
def plot_pc_lines_3d(Xz, V, B, K=2, colors=('red', 'darkgreen', 'orange'), elev=15, azim=-155):
    """
    Plot standardized data cloud in 3D (first 3 features of Xz) and overlay K principal directions (in original feature basis).
    Note: If X has >3 features, we plot its first 3 standardized coordinates; lines show direction, not full reconstruction.
    """
    if Xz.shape[1] < 3:
        print("Not enough original dimensions (need >=3) to make a 3D point cloud plot; skipping this figure.")
        return

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title("Standardized data (first 3 original dims) + PC directions")
    ax.view_init(elev=elev, azim=azim)

    # scatter first 3 standardized coordinates (for a 'point cloud' feel)
    ax.scatter(Xz[:, 0], Xz[:, 1], Xz[:, 2], c='lightsteelblue', marker='.', s=10)
    # draw PC direction vectors from origin (projected to these first 3 dims)
    K = min(K, V.shape[1])
    for i in range(K):
        v = V[:, i]  # full vector in original-d space
        # show its first 3 components as a direction line
        line = v[:3] / np.linalg.norm(v[:3]) if np.linalg.norm(v[:3]) > 0 else v[:3]
        ax.plot([0, line[0]], [0, line[1]], [0, line[2]], color=colors[i % len(colors)], lw=3, label=f"PC{i+1}")

    ax.set_xlabel("orig z1")
    ax.set_ylabel("orig z2")
    ax.set_zlabel("orig z3")
    ax.legend()

    # Projected data B in 2D with basis axes
    ax2 = fig.add_subplot(122)
    ax2.set_title("Scores (B) on first two PCs")
    ax2.scatter(B[:, 0], B[:, 1], c='lightsteelblue', marker='.', s=10)
    # basis vectors in score space are standard basis e1, e2
    ax2.plot([0, 1], [0, 0], color=colors[0], lw=3, label="PC1")
    ax2.plot([0, 0], [0, 1], color=colors[1], lw=3, label="PC2")
    ax2.set_xlabel("PC1 score")
    ax2.set_ylabel("PC2 score")
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def plot_scores_2d(B, y=None, title="Scores on first two PCs"):
    plt.figure(figsize=(6, 5))
    if y is None:
        plt.scatter(B[:, 0], B[:, 1], c='lightsteelblue', marker='.', s=12)
    else:
        # color by class 0/1 (or more)
        classes = np.unique(y)
        for i, c in enumerate(classes):
            mask = (y == c)
            plt.scatter(B[mask, 0], B[mask, 1], s=14, alpha=0.7, label=str(c))
        plt.legend(title="label")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()

def plot_explained_variance(df, pca, M_label=None):
    M = len(pca.explained_variance_)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].set_title("Explained Variance")
    axs[0].set_xlabel("Principal Component")
    axs[0].set_ylabel("Variance")
    axs[0].bar(range(1, M + 1), pca.explained_variance_)

    axs[1].set_title("Accumulated Fraction of Explained Variance")
    axs[1].set_xlabel("Principal Component")
    axs[1].set_ylabel("Cumulative Ratio")
    axs[1].plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    axs[1].set_ylim(0, 1.05)
    if M_label:
        fig.suptitle(M_label)
    plt.tight_layout()
    plt.show()
    
    if hasattr(pca, "components_"):
        feature_names = df.select_dtypes(include=np.number).drop(columns=["chd"], errors="ignore").columns
        comps = pca.components_
        for i, comp in enumerate(comps, start=1):
            order = np.argsort(np.abs(comp))[::-1]
            top = order[:8]   # top 8 features
            print(f"\nPC{i}: top features")
            for j in top:
                print(f"  {feature_names[j]:<12} {comp[j]: .3f}")