import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import sklearn.cluster as skcluster
import sklearn.metrics as skmetrics
import umap

from accnebtools.utils import NEB_ROOT

FONTSIZE = 44
FONTSIZE_LEGEND = 26


def setup_matplotlib(fontsize=32, fontsize_legend=26):
    rc_extra = {
        "font.size": fontsize,
        'legend.fontsize': fontsize_legend,
        'figure.figsize': (12, 9),
        'legend.frameon': True,
        'legend.edgecolor': '1',
        'legend.facecolor': 'inherit',
        'legend.framealpha': 0.6,
        'legend.markerscale': 1.4,
        # 'text.latex.preview': True,
        'text.usetex': True,
        'svg.fonttype': 'none',
        'text.latex.preamble': r'\usepackage{libertine}',
        'font.family': 'Linux Libertine',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'libertine',
        'mathtext.it': 'libertine:italic',
        'mathtext.bf': 'libertine:bold',
        'patch.facecolor': '#0072B2',
        'figure.autolayout': True,
        'lines.linewidth': 3,
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(rc_extra)


def get_separate_legend(ax: plt.Axes, figsize=(18, 2), ncols=6):
    ax.legend().set_visible(False)
    figlegend = pylab.figure(figsize=figsize)
    handles, labels = ax.get_legend_handles_labels()
    figlegend.legend(handles, labels, loc='center', ncol=ncols)
    return figlegend


def save_all_formats(figure: plt.Figure, save_path: str):
    main_dir = os.path.dirname(save_path)
    name = os.path.basename(save_path)
    save_dir_png = os.path.join(main_dir, "png")
    save_dir_pdf = os.path.join(main_dir, "pdf")
    os.makedirs(save_dir_png, exist_ok=True)
    os.makedirs(save_dir_pdf, exist_ok=True)

    figure.savefig(os.path.join(save_dir_png, f"{name}.png"), bbox_inches='tight')
    figure.savefig(os.path.join(save_dir_pdf, f"{name}.pdf"), bbox_inches='tight')


def equidistant_points(d):
    vk = np.asarray((1, -1), dtype=np.float64).reshape(1, -1)
    for k in range(2, d + 1):
        top_row = np.concatenate((np.ones(1, dtype=np.float64), -(1. / k) * np.ones(k, dtype=np.float64))).reshape(1,
                                                                                                                   -1)
        bottom_right_vk = np.sqrt(1 - (1 / k ** 2)) * vk
        zeros = np.zeros((bottom_right_vk.shape[0], 1))
        bottom_rows = np.concatenate((zeros, bottom_right_vk), axis=1)
        vk = np.concatenate((top_row, bottom_rows), axis=0)
    return vk.T


def generate_data(n, d, K, std_factor: float = 1.0, *, seed):
    assert d > 0
    rng = np.random.default_rng(seed)
    equ_points = equidistant_points(d)
    mu = equ_points[:K]
    std = np.sqrt(d) * std_factor
    points_per_class = (n // K)
    num_points = K * points_per_class
    x = np.repeat(mu, n // K, axis=0) + std * rng.normal(size=(num_points, d))
    x = x - np.mean(x, axis=0)

    # names = [points_per_class * [f"Class {i}"] for i in range(K)]
    names = [points_per_class * [string.ascii_uppercase[i]] for i in range(K)]
    names = sum(names, start=[])
    df = pd.DataFrame(x, columns=[f'{i}' for i in range(d)])
    df['names'] = names
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # return df


def fit_umap(df, seed):
    reducer = umap.UMAP(random_state=seed, min_dist=0.4, spread=0.5)
    embedding = reducer.fit_transform(df.iloc[:, :-1])
    new_df = pd.DataFrame(embedding, columns=['0', '1'])
    new_df['names'] = df['names']
    return new_df


def fit_kmeans(df, reps: int, seed: int):
    labels = df['names']
    num_labels = labels.nunique()
    x = df.iloc[:, :-1]
    seeds = np.random.SeedSequence(seed).generate_state(reps)
    results = {'accuracy': [], 'nmi': []}
    for seed_ in seeds:
        clf = skcluster.KMeans(n_clusters=num_labels, random_state=seed_)
        clf = clf.fit(x)
        accuracy = skmetrics.rand_score(labels, clf.labels_)
        nmi = skmetrics.normalized_mutual_info_score(labels, clf.labels_)
        results['accuracy'].append(accuracy)
        results['nmi'].append(nmi)
    return pd.DataFrame(results)


def concat_and_svd(df, num_repeat):
    x = df.iloc[:, :-1]
    xx = np.concatenate(num_repeat * [x], axis=1)
    # xx = xx - np.mean(xx, axis=0)
    u, s, vh = np.linalg.svd(xx.astype(np.float32), full_matrices=False)
    new_df = pd.DataFrame(u, columns=[f'{i}' for i in range(u.shape[1])])
    new_df['names'] = df['names']
    return new_df


def main(seed=23525):
    setup_matplotlib(fontsize=FONTSIZE, fontsize_legend=FONTSIZE_LEGEND)
    K = 6
    d = K
    n = 200
    save_folder = os.path.join(NEB_ROOT, "results", "figures", "cikm", "cluster_demo")
    df = generate_data(n=n, d=d, K=K, std_factor=0.1, seed=seed)
    res = fit_kmeans(df, 10, seed)
    print(f"Clustering res original: \n", res.mean(), "\n", res.std())

    umap_df = fit_umap(df, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(umap_df, x='0', y='1', hue='names', style='names', s=150,
                    hue_order=list(string.ascii_uppercase[:K]), style_order=list(string.ascii_uppercase[:K]))
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")

    figlegend = get_separate_legend(ax=ax, figsize=(12, 2), ncols=K)
    save_all_formats(fig, os.path.join(save_folder, f"original"))
    plt.close(fig)
    save_all_formats(figlegend, os.path.join(save_folder, f"legend"))
    plt.close(figlegend)

    for num_copy in [2, 3, 5]:
        cat_svd_df = concat_and_svd(df, num_copy)

        res = fit_kmeans(cat_svd_df, 10, seed)
        print(f"Concat {num_copy} and SVD: \n", res.mean(), "\n", res.std())

        umap_cat_svd_df = fit_umap(cat_svd_df, seed=seed)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(umap_cat_svd_df, x='0', y='1', hue='names', style='names', s=400, linewidths=0, alpha=0.8,
                        hue_order=list(string.ascii_uppercase[:K]), style_order=list(string.ascii_uppercase[:K]))
        ax.set_xlabel("UMAP dim 1")
        ax.set_ylabel("UMAP dim 2")
        ax.legend().set_visible(False)
        save_all_formats(fig, os.path.join(save_folder, f"with_{num_copy}_copies"))
        plt.close(fig)


if __name__ == "__main__":
    main()
