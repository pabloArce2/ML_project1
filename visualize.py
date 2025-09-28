import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D)
import itertools

def histogram(
    df,
    bins=20,
    ncols=3,
    title="Histograms",
    by_chd=None,  # None = all, 0 = only negatives, 1 = only positives, "split" = overlay both
):
    num_df = df.select_dtypes(include=np.number).drop(columns=["chd"], errors="ignore")

    n = len(num_df.columns)
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=False)
    axs = axs.flatten()

    for j, col in enumerate(num_df.columns):
        ax = axs[j]
        if by_chd == "split" and "chd" in df.columns:
            ax.hist(
                df.loc[df["chd"] == 0, col],
                bins=bins, edgecolor="black", alpha=0.6, label="chd=0"
            )
            ax.hist(
                df.loc[df["chd"] == 1, col],
                bins=bins, edgecolor="black", alpha=0.6, label="chd=1"
            )
            ax.legend(fontsize=8)
        elif by_chd in (0, 1) and "chd" in df.columns:
            ax.hist(
                df.loc[df["chd"] == by_chd, col],
                bins=bins, edgecolor="black"
            )
        else:
            ax.hist(df[col], bins=bins, edgecolor="black")

        ax.set_title(col, fontsize=10, pad=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    # remove empty plots
    for k in range(j+1, len(axs)):
        fig.delaxes(axs[k])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def histogram_overlay_by_chd(
    df,
    features=None,        # None = all numeric except 'chd'; or list of names/indices
    bins=20,
    ncols=3,
    alpha=0.55,
    title="Distributions by CHD (proportion-normalized)",
    fmt="{:.2f}",         # number formatting for mean/std
):
    if "chd" not in df.columns:
        raise ValueError("Column 'chd' not found in DataFrame.")

    # numeric features only (exclude target)
    num_df = df.select_dtypes(include=np.number).drop(columns=["chd"], errors="ignore")

    # resolve features arg (names or integer indices)
    if features is None:
        cols = list(num_df.columns)
    else:
        cols = [num_df.columns[i] for i in features] if all(isinstance(f, int) for f in features) else list(features)

    if not cols:
        raise ValueError("No numeric features selected to plot.")

    n = len(cols)
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3.5*nrows),
        sharex=False, sharey=False,
        gridspec_kw={"wspace": 0.35, "hspace": 0.75},  # more vertical spacing
    )
    axs = np.atleast_1d(axs).ravel()

    mask0 = df["chd"] == 0
    mask1 = df["chd"] == 1
    n0 = max(mask0.sum(), 1)
    n1 = max(mask1.sum(), 1)

    for j, col in enumerate(cols):
        ax = axs[j]

        x0 = df.loc[mask0, col].dropna().values
        x1 = df.loc[mask1, col].dropna().values

        # shared bin edges so bars align
        combined = np.concatenate([x0, x1]) if (x0.size and x1.size) else (x0 if x0.size else x1)
        bin_edges = np.histogram_bin_edges(combined, bins=bins) if combined.size else bins

        # per-group weights (proportions)
        w0 = np.full_like(x0, 1.0 / n0, dtype=float) if x0.size else None
        w1 = np.full_like(x1, 1.0 / n1, dtype=float) if x1.size else None

        h0 = ax.hist(x0, bins=bin_edges, weights=w0, edgecolor="black", alpha=alpha, label="chd=0")
        h1 = ax.hist(x1, bins=bin_edges, weights=w1, edgecolor="black", alpha=alpha, label="chd=1")
        text = col
        ax.set_title(text, fontsize=12, pad=20)  # multiline title above plot
        # ---- mean & std annotations per group (above graph) ----
     
        if x0.size:
            m0, s0 = np.mean(x0), np.std(x0, ddof=1) if x0.size > 1 else 0.0
            ax.text(0.5, 1.08, f"0: μ={fmt.format(m0)}, σ={fmt.format(s0)}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6)

        if x1.size:
            m1, s1 = np.mean(x1), np.std(x1, ddof=1) if x1.size > 1 else 0.0
            ax.text(0.5, 1.02, f"1: μ={fmt.format(m1)}, σ={fmt.format(s1)}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6)

  

        ax.set_xlabel("Value")
        ax.set_ylabel("Proportion")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

    # remove unused panels
    for k in range(j + 1, len(axs)):
        fig.delaxes(axs[k])

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def boxplot(df, max_cols=10, ncols=3, title="Boxplots of SAheart (numeric)"):
    # numeric only
    num_df = df.select_dtypes(include="number")
    cols = list(num_df.columns)[:max_cols]
    n = len(cols)
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 5*nrows))
    axs = axs.ravel()

    for i, col in enumerate(cols):
        axs[i].boxplot(num_df[col].dropna(), vert=True, patch_artist=True)
        axs[i].set_title(col, fontsize=10, pad=6)
        axs[i].set_ylabel("Value")
        # remove x-axis ticks & labels
        axs[i].set_xticks([])

    # turn off unused axes
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def scatter_matrix_subset(df, features_idx, label="chd", bins=20, alpha=0.5, s=15):
    """
    Create a scatter matrix (pair plot) for selected feature indices, colored by binary label.
    
    Parameters
    ----------
    df : DataFrame
        Input data containing numeric features and a binary target column.
    features_idx : list of int
        Indices of the numeric features to include.
    label : str
        Target column name (binary, e.g. 0/1).
    bins : int
        Number of bins for histograms on the diagonal.
    alpha : float
        Transparency of scatter points.
    s : int
        Marker size for scatter plots.
    """
    # select numeric features
    num_df = df.select_dtypes(include=np.number)
    cols = [num_df.columns[i] for i in features_idx]
    p = len(cols)

    fig, axs = plt.subplots(p, p, figsize=(2.8*p, 2.8*p), sharex="col", sharey="row")
    fig.suptitle("Scatter Matrix (subset)", fontsize=14)

    m0 = df[label] == 0
    m1 = df[label] == 1

    for i in range(p):
        for j in range(p):
            ax = axs[i, j]
            if i == j:
                # diagonal: histograms
                ax.hist(df.loc[m0, cols[j]], bins=bins, alpha=0.6, label=f"{label}=0", edgecolor="black")
                ax.hist(df.loc[m1, cols[j]], bins=bins, alpha=0.6, label=f"{label}=1", edgecolor="black")
            else:
                # off-diagonal: scatter plots
                ax.scatter(df.loc[m0, cols[j]], df.loc[m0, cols[i]], s=s, alpha=alpha, label=f"{label}=0")
                ax.scatter(df.loc[m1, cols[j]], df.loc[m1, cols[i]], s=s, alpha=alpha, label=f"{label}=1")

            # labels only on edges
            if j == 0:
                ax.set_ylabel(cols[i])
            else:
                ax.set_ylabel(None)
            if i == p-1:
                ax.set_xlabel(cols[j])
            else:
                ax.set_xlabel(None)

    # one legend in bottom-right corner
    axs[-1, -1].legend(loc="upper right", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def scatter_3d_combinations(
    df,
    features,                 # list of indices OR names (length >= 3)
    label_col,                # e.g. "chd" or "Type"
    use_indices=True,         # True if `features` are column indices into numeric df
    bins=None,                # unused; here for parity with 2D versions if you add diags later
    ncols=2,                  # subplots per row in each figure
    s=30, alpha=0.6,          # marker size & transparency
    elev=20, azim=120,        # default view
    figsize_per_ax=(6, 5),    # (width, height) of each 3D subplot
    title_prefix="3D Scatter: ",
):
    """
    Generate 3D scatter plots for all 3-feature combinations from the given features.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features and `label_col`.
    features : list[int] or list[str]
        Feature indices or names to consider (>= 3).
    label_col : str
        Column name of the label to color by (binary or multi-class).
    use_indices : bool
        If True, `features` are integer indices into *numeric* columns of df.
        If False, `features` are treated as column names (will validate existence).
    ncols : int
        Number of 3D subplots per row in each figure.
    s, alpha : numbers
        Scatter marker size and transparency.
    elev, azim : numbers
        View angles for `Axes3D.view_init`.
    figsize_per_ax : (float, float)
        Size of each subplot; total figure size scales with grid size.
    title_prefix : str
        Prefix for each subplot title.
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in DataFrame.")

    # Build list of feature names from indices or names
    if use_indices:
        num_df = df.select_dtypes(include=np.number)
        all_num_cols = list(num_df.columns)
        try:
            cols = [all_num_cols[i] for i in features]
        except IndexError:
            raise ValueError("One or more feature indices are out of range for numeric columns.")
    else:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        cols = list(features)

    if len(cols) < 3:
        raise ValueError("Provide at least 3 features.")

    combos = list(itertools.combinations(cols, 3))
    if not combos:
        return  # nothing to plot

    # Group data by label once
    groups = list(df.groupby(label_col))

    # Layout: figure pages with ncols x nrows grid
    nplots = len(combos)
    nrows = 2 if nplots > ncols else 1  # sensible default; adjust below per page

    # We will paginate so each figure isn't too crowded
    max_per_fig = ncols * nrows

    # If too many combos, expand rows up to 3 per page before paginating
    while max_per_fig < min(nplots, 6) and nrows < 3:
        nrows += 1
        max_per_fig = ncols * nrows

    # Iterate in pages
    for start in range(0, nplots, max_per_fig):
        page_combos = combos[start:start + max_per_fig]
        rows = int(np.ceil(len(page_combos) / ncols))
        fig = plt.figure(figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * rows))

        for idx, (a, b, c) in enumerate(page_combos, start=1):
            ax = fig.add_subplot(rows, ncols, idx, projection='3d')
            for gi, (lab, sub) in enumerate(groups):
                ax.scatter(
                    sub[a], sub[b], sub[c],
                    s=s, alpha=alpha, label=str(lab), color=f"C{gi}"
                )
            ax.set_xlabel(a); ax.set_ylabel(b); ax.set_zlabel(c)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{title_prefix}{a}, {b}, {c}", fontsize=10, pad=10)

            # Put legend only on the last subplot of the page
            if idx == len(page_combos):
                ax.legend(title=label_col, fontsize=9, loc="upper left")

        plt.tight_layout()
        plt.show()