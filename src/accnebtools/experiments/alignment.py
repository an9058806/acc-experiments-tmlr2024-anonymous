from typing import Mapping
import numpy as np
import pandas as pd
from numpy.random import Generator

from sklearn.neighbors import KDTree

from accnebtools.data.graph import SimpleGraph
import accnebtools.experiments.utils as utils


class AlignedGraphs:
    g1_num_nodes: int
    g2_num_nodes: int
    g1_to_g2: pd.Series
    g2_to_g1: pd.Series

    def __init__(self, g1_num_nodes, g2_num_nodes, g2_to_g1):
        assert g1_num_nodes >= g2_num_nodes
        assert len(g2_to_g1) == g2_num_nodes

        self.g1_num_nodes = g1_num_nodes
        self.g2_num_nodes = g2_num_nodes

        self.g2_to_g1 = g2_to_g1
        self.g1_to_g2 = pd.Series(g2_to_g1.index, index=g2_to_g1)

    @staticmethod
    def g2_to_merged(g1_num_nodes: int, g2_num_nodes: int):
        return pd.Series(np.arange(g2_num_nodes, dtype=np.int64) + g1_num_nodes)

    @staticmethod
    def merged_to_g2(g1_num_nodes: int, g2_num_nodes: int):
        return pd.Series(np.arange(g2_num_nodes, dtype=np.int64),
                         index=np.arange(g2_num_nodes, dtype=np.int64) + g1_num_nodes)

    @classmethod
    def load_from_file(cls, path: str):
        with open(path, "r") as fp:
            first_line = fp.readline()
        g1_num_nodes, g2_num_nodes = first_line.strip("%\n ").split("::")
        g1_num_nodes = int(g1_num_nodes)
        g2_num_nodes = int(g2_num_nodes)
        g2_to_g1 = pd.read_csv(path, index_col=0, names=["g2_to_g1"], header=None, comment="%")["g2_to_g1"]
        return cls(g1_num_nodes, g2_num_nodes, g2_to_g1)

    def save2file(self, fp):
        fp.write("%" + str(self.g1_num_nodes) + "::" + str(self.g2_num_nodes) + "\n")
        pd.Series(self.g2_to_g1).to_csv(fp, index=True, header=False)


def create_permuted(g: SimpleGraph, rng: Generator):
    g2_to_g1 = pd.Series(rng.permutation(g.num_nodes))
    g1_to_g2 = pd.Series(g2_to_g1.index, index=g2_to_g1)
    new_sources = g1_to_g2[g.edges[:, 0]]
    new_targets = g1_to_g2[g.edges[:, 1]]
    g2_edges = np.stack((new_sources, new_targets), axis=1)
    if g.node_attributes is not None:
        new_node_order = g2_to_g1[range(g.num_nodes)]
        new_node_attributes = g.node_attributes[new_node_order, :]
    else:
        new_node_attributes = None
    new_g = SimpleGraph(num_nodes=g.num_nodes, edges=g2_edges, weights=g.weights, directed=g.directed,
                        node_attributes=new_node_attributes)
    return new_g, AlignedGraphs(g.num_nodes, new_g.num_nodes, g2_to_g1)


def split_embeddings(embeddings: np.ndarray, g2_to_merged: pd.Series):
    g1_num_nodes = embeddings.shape[0] - len(g2_to_merged)
    g1_embeddings = embeddings[:g1_num_nodes, :]
    g2_merged = g2_to_merged[np.arange(len(g2_to_merged), dtype=np.int64)]
    g2_embeddings = embeddings[g2_merged, :]
    return g1_embeddings, g2_embeddings


def calc_topk_similarties(g1_embeddings, g2_embeddings, alpha=50):
    kd_tree = KDTree(g1_embeddings, metric="euclidean")

    dist, ind = kd_tree.query(g2_embeddings, k=alpha)
    similarity = np.exp(-dist)
    return similarity, ind


def get_top_sim(embeddings: np.ndarray, g2_to_merged: pd.Series, alpha=50):
    num_g2_nodes = len(g2_to_merged)
    g1_embeddings, g2_embeddings = split_embeddings(embeddings, g2_to_merged)
    similarity, ind = calc_topk_similarties(g1_embeddings, g2_embeddings, alpha=alpha)
    assert similarity.shape[0] == num_g2_nodes and ind.shape[0] == num_g2_nodes

    res = dict()
    for i in range(num_g2_nodes):
        res[i] = [(s, node) for s, node in zip(similarity[i, :], ind[i, :])]

    return res


def calc_topk_acc_score(y_true: Mapping, top_sim: dict, topk_vals: np.ndarray):
    scores = np.zeros(len(topk_vals), dtype=np.float64)

    for g2_node, y in y_true.items():
        vals = sorted(top_sim[g2_node], key=lambda x: -x[0])
        the_k = np.inf
        for k_, (sim, val) in enumerate(vals):
            if y == val:
                the_k = k_
        scores[the_k < topk_vals] += 1.
    scores = scores / len(y_true)
    return dict(zip(topk_vals, scores))


def eval_topk_sim(pp_emb_generator: utils.PP_EMBS, alignment: AlignedGraphs):
    topk_vals = np.asarray([1, 5, 10], dtype=np.int64)
    all_align_results = {}
    for pp_mode, X in pp_emb_generator:
        if np.isfinite(X).all():
            try:
                top_sim = get_top_sim(X, alignment.g2_to_merged(alignment.g1_num_nodes, alignment.g2_num_nodes),
                                      alpha=np.max(topk_vals))
                res = calc_topk_acc_score(alignment.g2_to_g1, top_sim, topk_vals=topk_vals)
                all_align_results[pp_mode] = res
            except IndexError:
                print(f"Index error produced during alignment")
                continue

    return all_align_results


def get_k_per_node(y_true: Mapping, top_sim: dict):
    k_values_per_node = np.full(len(y_true), fill_value=np.nan)
    for g2_node, y in y_true.items():
        vals = sorted(top_sim[g2_node], key=lambda x: -x[0])
        for k_, (sim, val) in enumerate(vals):
            if y == val:
                k_values_per_node[y] = k_
    return k_values_per_node


def eval_k_nearest_per_node(pp_emb_generator: utils.PP_EMBS, alignment: AlignedGraphs):
    all_align_results = {}
    for pp_mode, X in pp_emb_generator:
        if np.isfinite(X).all():
            try:
                top_sim = get_top_sim(X, alignment.g2_to_merged(alignment.g1_num_nodes, alignment.g2_num_nodes),
                                      alpha=25)
                res = get_k_per_node(alignment.g2_to_g1, top_sim)
                all_align_results[pp_mode] = res
            except IndexError:
                print(f"Index error produced during alignment")
                continue

    return all_align_results
