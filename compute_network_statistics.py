from typing import Union
import os
import json
import numpy as np
import graph_tool as gt
import graph_tool.clustering as gtclust
import graph_tool.topology as gttop
import pandas as pd
import torch
import torch_sparse
import torch_scatter
import accnebtools.utils as nebutils
import accnebtools.experiments.analysis as nebanalysis
import accnebtools.data.graph as dgraph

STATS_PATH = os.path.join(nebutils.NEB_DATAROOT, "data_stats.json")


def get_node_homophily(y: torch.LongTensor, edge_index: Union[torch.Tensor, torch_sparse.SparseTensor],
                       edge_weight: torch.Tensor = None):
    """
    Return the weighted node homophily, according to the weights in the provided adjacency matrix.
    """
    src, dst, edge_weight = get_weighted_edges(edge_index, edge_weight)

    index = src
    mask = (y[src] == y[dst]).float().squeeze() * edge_weight
    per_node_masked_sum = torch_scatter.scatter_sum(mask, index)
    per_node_total_sum = torch_scatter.scatter_sum(edge_weight, index)

    non_zero_mask = per_node_total_sum != 0
    return (per_node_masked_sum[non_zero_mask] / per_node_total_sum[non_zero_mask]).mean()


def get_compatibility_matrix(y, edge_index, edge_weight=None):
    """
    Return the weighted compatibility matrix, according to the weights in the provided adjacency matrix.
    """
    src, dst, edge_weight = get_weighted_edges(edge_index, edge_weight)

    num_classes = torch.unique(y).shape[0]
    H = torch.zeros((num_classes, num_classes))
    for i in range(src.shape[0]):
        y_src = y[src[i]]
        y_dst = y[dst[i]]
        H[y_src, y_dst] += edge_weight[i]

    return torch.nn.functional.normalize(H, p=1)


def get_weighted_edges(edge_index, edge_weight=None):
    """
    Return (src, dst, edge_weight) tuple.
    """
    if isinstance(edge_index, torch_sparse.SparseTensor):
        src, dst, edge_weight = edge_index.coo()
    else:
        src, dst = edge_index
        edge_weight = (
            edge_weight if edge_weight is not None else torch.ones((edge_index.size(1),), device=edge_index.device)
        )

    return src, dst, edge_weight


def compute_all_homophily(graph: dgraph.SimpleGraph, labels: pd.Series):
    y = torch.from_numpy(labels.to_numpy()).to(dtype=torch.long)
    adj = torch_sparse.SparseTensor(
        row=torch.from_numpy(graph.edges[:, 0]),
        col=torch.from_numpy(graph.edges[:, 1]),
        value=torch.ones(graph.num_edges, dtype=torch.float32),
        sparse_sizes=(graph.num_nodes, graph.num_nodes)
    )
    adj_undir = adj.to_symmetric()

    h_u = get_node_homophily(y=y, edge_index=adj_undir).item()
    h_u2 = get_node_homophily(y=y, edge_index=adj_undir @ adj_undir).item()
    h_u_eff = max(h_u, h_u2)

    h_d = get_node_homophily(y=y, edge_index=adj).item()
    h_d_t = get_node_homophily(y=y, edge_index=adj.t()).item()
    h_d_aa = get_node_homophily(y=y, edge_index=adj @ adj).item()
    h_d_a_a_t = get_node_homophily(y=y, edge_index=adj @ adj.t()).item()
    h_d_a_t_a = get_node_homophily(y=y, edge_index=adj.t() @ adj).item()
    h_d_a_t_a_t = get_node_homophily(y=y, edge_index=adj.t() @ adj.t()).item()
    h_d_eff = max(h_d, h_d_t, h_d_aa, h_d_a_a_t, h_d_a_t_a, h_d_a_t_a_t)

    h_eff_gain = (h_d_eff - h_u_eff) / h_u_eff

    out = {
        "h_u": h_u,
        "h_u2": h_u2,
        "h_u_eff": h_u_eff,
        "h_d": h_d,
        "h_d_t": h_d_t,
        "h_d_aa": h_d_aa,
        "h_d_a_a_t": h_d_a_a_t,
        "h_d_a_t_a": h_d_a_t_a,
        "h_d_a_t_a_t": h_d_a_t_a_t,
        "h_d_eff": h_d_eff,
        "h_eff_gain": h_eff_gain
    }
    return out


def compute_all_homophily_partially_labelled(graph: dgraph.SimpleGraph, labels: pd.Series):
    subgraph_nodes = torch.from_numpy(labels.index.to_numpy())
    y = torch.from_numpy(labels.to_numpy()).to(dtype=torch.long)
    adj = torch_sparse.SparseTensor(
        row=torch.from_numpy(graph.edges[:, 0]),
        col=torch.from_numpy(graph.edges[:, 1]),
        value=torch.ones(graph.num_edges, dtype=torch.float32),
        sparse_sizes=(graph.num_nodes, graph.num_nodes)
    )
    adj_undir = adj.to_symmetric()

    adj_u = adj_undir[:, subgraph_nodes][subgraph_nodes]
    h_u = get_node_homophily(y=y, edge_index=adj_u).item()
    adj_u2 = adj_undir @ adj_undir
    adj_u2 = adj_u2[:, subgraph_nodes][subgraph_nodes]
    h_u2 = get_node_homophily(y=y, edge_index=adj_u2).item()
    del adj_u2
    h_u_eff = max(h_u, h_u2)

    adj_d = adj[:, subgraph_nodes][subgraph_nodes]
    h_d = get_node_homophily(y=y, edge_index=adj_d).item()
    h_d_t = get_node_homophily(y=y, edge_index=adj_d.t()).item()
    del adj_d

    aa_d = adj @ adj
    aa_d = aa_d[:, subgraph_nodes][subgraph_nodes]
    h_d_aa = get_node_homophily(y=y, edge_index=aa_d).item()
    del aa_d
    aa_t = adj @ adj.t()
    aa_t = aa_t[:, subgraph_nodes][subgraph_nodes]
    h_d_a_a_t = get_node_homophily(y=y, edge_index=aa_t).item()
    del aa_t
    a_t_a = adj.t() @ adj
    a_t_a = a_t_a[:, subgraph_nodes][subgraph_nodes]
    h_d_a_t_a = get_node_homophily(y=y, edge_index=a_t_a).item()
    del a_t_a

    a_t_a_t = adj.t() @ adj.t()
    a_t_a_t = a_t_a_t[:, subgraph_nodes][subgraph_nodes]
    h_d_a_t_a_t = get_node_homophily(y=y, edge_index=a_t_a_t).item()
    del a_t_a_t

    h_d_eff = max(h_d, h_d_t, h_d_aa, h_d_a_a_t, h_d_a_t_a, h_d_a_t_a_t)

    h_eff_gain = (h_d_eff - h_u_eff) / h_u_eff

    out = {
        "h_u": h_u,
        "h_u2": h_u2,
        "h_u_eff": h_u_eff,
        "h_d": h_d,
        "h_d_t": h_d_t,
        "h_d_aa": h_d_aa,
        "h_d_a_a_t": h_d_a_a_t,
        "h_d_a_t_a": h_d_a_t_a,
        "h_d_a_t_a_t": h_d_a_t_a_t,
        "h_d_eff": h_d_eff,
        "h_eff_gain": h_eff_gain
    }
    return out


def print_summary(graph: dgraph.SimpleGraph, labels: pd.Series, node_labels_type: str):
    data = dict()
    print(f"Num nodes: {graph.num_nodes}")
    print(f"Num edges: {graph.edges.shape[0]}")
    log10_density = np.log10(graph.edges.shape[0]) - 2 * np.log10(graph.num_nodes)
    print(f"Log10 density: {log10_density:.3f}")
    print(f"Directed: {graph.directed}")
    print(f"Weighted: {graph.weights.std() > 0}")
    print(f"Weights std: {graph.weights.std()}")

    data["num_nodes"] = int(graph.num_nodes)
    data["num_edges"] = int(graph.edges.shape[0])
    data["directed"] = graph.directed
    data["Weighted_std"] = graph.weights.std()
    data["log10_density"] = log10_density

    print(f"Attributed: {graph.node_attributes is not None}")
    if graph.node_attributes is not None:
        print(f"Num attributes: {graph.node_attributes.shape[1]}")
        data["num_node_attributes"] = graph.node_attributes.shape[1]
    print(f"Labels: {labels is not None}")
    if labels is not None:
        print("Labels shape", labels.shape)
        data["labels_shape"] = labels.shape
        if node_labels_type in {"multiclass", "binary"} and (len(labels.shape) < 2 or labels.shape[1] == 1):
            print("num classes: ", len(labels.unique()))
            print(labels.value_counts())
            data["num_classes"] = len(labels.unique())
            if labels.shape[0] == graph.num_nodes:
                homophily = compute_all_homophily(graph=graph, labels=labels)
            else:
                homophily = compute_all_homophily_partially_labelled(graph=graph, labels=labels)
            data.update(homophily)

    gt_graph = graph.to_gt_graph()
    out_degs = gt_graph.degree_property_map("out").a
    in_degs = gt_graph.degree_property_map("in").a
    deg_cov = np.cov(np.stack((out_degs, in_degs), axis=0))

    spectral_norm = np.linalg.eigvalsh(deg_cov)[-1]
    sqrt_determinant = np.sqrt(np.linalg.det(deg_cov))
    cov_trace = np.trace(deg_cov)

    mean_out_deg = out_degs.mean()
    mean_in_deg = in_degs.mean()

    std_out_deg = np.std(out_degs)
    std_in_deg = np.std(in_degs)

    median_out_deg = np.median(out_degs)
    median_in_deg = np.median(in_degs)

    c = gtclust.global_clustering(gt_graph)[0]
    comp, hist = gttop.label_components(gt_graph, directed=False)
    comp_dir, hist_dir = gttop.label_components(gt_graph, directed=graph.directed)
    num_comps = len(hist)
    comp_sizes_and_diams = []
    for com_idx, num_memb in enumerate(hist):
        if num_memb > 3:
            member_node = np.argmax(comp.a == com_idx)
            pseudo_diam = float(gttop.pseudo_diameter(gt.GraphView(gt_graph, directed=False), source=member_node)[0])
            comp_sizes_and_diams.append((int(num_memb), pseudo_diam))

    print(f"Mean out deg: {mean_out_deg:.2f}")
    print(f"Mean in deg: {mean_in_deg:.2f}")

    print(f"Median out deg: {median_out_deg:.2f}")
    print(f"Median in deg: {median_in_deg:.2f}")

    print(f"Clustering coeff: {c}")
    print(f"Num weakly connected comps: {num_comps}")
    print(f"Num strongly connected comps: {len(hist_dir)}")
    print(f"pseudo_diam: ", comp_sizes_and_diams)

    data["mean_out_deg"] = mean_out_deg.item()
    data["mean_in_deg"] = mean_in_deg.item()
    data["std_out_deg"] = std_out_deg.item()
    data["std_in_deg"] = std_in_deg.item()
    data["spectral_norm"] = spectral_norm.item()
    data["sqrt_determinant"] = sqrt_determinant.item()
    data["cov_trace"] = cov_trace.item()
    data["median_out_deg"] = median_out_deg.item()
    data["median_in_deg"] = median_in_deg.item()
    data["clustering_coeff"] = c
    data["num_cc"] = int(num_comps)
    data["num_scc"] = len(hist_dir)
    data["pseudo_diam"] = comp_sizes_and_diams

    if graph.num_nodes < 10000:
        out = gttop.shortest_distance(gt_graph)
        out = out.get_2d_array(pos=list(range(graph.num_nodes)))
        out = out[1:]
        out = out[out < graph.num_nodes]
        avg_path_length = np.mean(out).item()
        avg_path_length_se = None
        print(f"Avg. path length: {avg_path_length:.3f}")
    else:
        sample_size = 100
        sources = np.random.choice(graph.num_nodes, size=sample_size, replace=False)
        samples = []
        for src in sources:
            out = gttop.shortest_distance(gt_graph, source=src)
            out = out.a[1:]
            out = out[out < graph.num_nodes]
            samples.append(np.mean(out))
        avg_path_length = np.mean(samples)
        avg_path_length = avg_path_length.item()
        avg_path_length_se = np.std(samples) / np.sqrt(sample_size)
        avg_path_length_se = avg_path_length_se.item()
        print(f"Avg. path length: {avg_path_length:.3f} +/- {avg_path_length_se:.3f}")

    data["avg_path_length"] = avg_path_length
    data["avg_path_length_se"] = avg_path_length_se
    return data


def get_existing_stats():
    stats = dict()
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as fp:
            stats = json.load(fp)
    return stats


if __name__ == "__main__":
    with open(os.path.join(nebutils.NEB_DATAROOT, "data_index.json"), 'r') as fp:
        data_index = json.load(fp)
    all_stats = get_existing_stats()
    for dataname in data_index.keys():
        if dataname in all_stats or dataname == "dblp" or dataname == "pokec":
            continue
        print(f"Calculating stats of graph: {dataname}")
        graph, labels, node_labels_type = nebanalysis.read_graph_and_labels(
            dgraph.DatasetSpec(dataname, force_undirected=False, force_unweighted=False, rm_node_attributes=False,
                               with_self_loops=False))
        all_stats[dataname] = print_summary(graph, labels, node_labels_type)
        print()
        if graph.num_nodes > 10000:
            with open(STATS_PATH, 'w') as fp:
                json.dump(all_stats, fp)

    with open(STATS_PATH, 'w') as fp:
        json.dump(all_stats, fp, indent=2)
