import dataclasses as dc
from functools import cached_property
import os
import json
import numpy as np
import pandas as pd
import graph_tool as gt
from numpy.random import Generator
import torch
import torch_geometric as pyg
import dgl

from accnebtools.data.core_ import read_edges, read_graph_from_npz
import accnebtools.utils as nebutils


@dc.dataclass(frozen=True)
class DatasetSpec:
    data_name: str
    force_undirected: False
    force_unweighted: False
    rm_node_attributes: False
    with_self_loops: False


@dc.dataclass(frozen=True)
class SimpleGraph:
    num_nodes: int
    edges: np.ndarray
    directed: bool = True
    weights: np.ndarray = None
    node_attributes: np.ndarray = None

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def is_undirected(self) -> bool:
        return not self.directed

    @cached_property
    def is_weighted(self) -> bool:
        return self.weights is not None and not np.allclose(self.weights, 1.)

    @property
    def is_node_attributed(self) -> bool:
        return self.node_attributes is not None and self.node_attributes.shape[0] == self.num_nodes

    @cached_property
    def edges_are_canonical(self) -> bool:
        return (not self.directed
                and np.all(self.edges[:, 0] <= self.edges[:, 1])
                and np.unique(self.edges, axis=0).shape[0] == self.edges.shape[0])

    @classmethod
    def load(cls, fp, *, as_canonical_undirected, add_symmetrical_edges=False, remove_self_loops=True,
             with_node_attributes=True, use_weights=True):
        num_nodes, edges, weights, node_attributes, directed = read_graph_from_npz(
            fp,
            as_canonical_undirected=as_canonical_undirected,
            add_symmetrical_edges=add_symmetrical_edges,
            remove_self_loops=remove_self_loops,
            use_weights=use_weights
        )
        if not with_node_attributes:
            node_attributes = None
        return cls(num_nodes=num_nodes, edges=edges, directed=directed,
                   weights=weights, node_attributes=node_attributes)

    @classmethod
    def from_dataset_spec(cls, dataroot: str, dataset_spec: DatasetSpec):
        graph = cls.from_dataset_index(dataroot=dataroot,
                                       data_name=dataset_spec.data_name, as_unweighted=dataset_spec.force_unweighted,
                                       as_undirected=dataset_spec.force_undirected,
                                       remove_self_loops=not dataset_spec.with_self_loops,
                                       with_node_attributes=not dataset_spec.rm_node_attributes)
        return graph

    @classmethod
    def from_dataset_index(cls, dataroot: str, data_name: str, as_unweighted: bool, as_undirected: bool,
                           remove_self_loops=True,
                           with_node_attributes=True):

        index_path = os.path.join(dataroot, "data_index.json")
        with open(index_path, 'r') as fp:
            data_index = json.load(fp)
        if data_name not in data_index:
            raise KeyError(data_name + " was not found in the data index.")
        metadata = data_index[data_name]
        is_weighted = metadata['graph_info']['weights'] and not as_unweighted
        directed = metadata['graph_info']['directed'] and not as_undirected
        graphpath = os.path.join(nebutils.NEB_ROOT, metadata['graphpath'])
        kwargs = {}
        if metadata['file_info']['filetype'] in {'csv', 'tsv', 'edgelist', 'txt', 'parquet'}:
            if "comment_char" in metadata['file_info']:
                kwargs["comment"] = metadata['file_info']['comment_char']
            edges, weights, num_nodes = read_edges(filename=graphpath,
                                                   filetype=metadata['file_info']['filetype'],
                                                   is_weighted=is_weighted,
                                                   as_canonical_undirected=not directed,
                                                   add_symmetrical_edges=not directed,
                                                   num_nodes=metadata['graph_info']['num_nodes'],
                                                   remove_self_loops=remove_self_loops,
                                                   **kwargs
                                                   )
            graph = cls(num_nodes=num_nodes, edges=edges, directed=directed, weights=weights)
        elif metadata['file_info']['filetype'] == 'npz':
            graph = cls.load(graphpath, as_canonical_undirected=not directed,
                             add_symmetrical_edges=not directed, remove_self_loops=remove_self_loops,
                             with_node_attributes=with_node_attributes, use_weights=is_weighted)
        else:
            raise NotImplementedError
        return graph

    @classmethod
    def from_gt_graph(cls, graph: gt.Graph, as_canonical_undirected: bool,
                      add_symmetrical_edges: bool):
        edges = np.asarray([[int(s), int(t)] for s, t in graph.edges()], dtype=np.int64)
        if 'weights' in graph.ep:
            weights = graph.ep['weights'].a
        elif 'weight' in graph.ep:
            weights = graph.ep['weight'].a
        else:
            weights = None
        directed = graph.is_directed() and not as_canonical_undirected
        simple_graph = cls(num_nodes=graph.num_vertices(), edges=edges, weights=weights, directed=directed)
        if as_canonical_undirected or not directed:
            simple_graph = make_edges_canonical(simple_graph)
        if add_symmetrical_edges or not directed:
            simple_graph = add_symmetric_edges(simple_graph)
        return simple_graph

    def to_gt_graph(self):
        this = self
        if not self.directed:
            this = make_edges_canonical(self)
        graph = gt.Graph(directed=this.directed)
        graph.add_vertex(this.num_nodes)
        graph.add_edge_list(this.edges)
        if this.weights is not None and ((len(this.weights.shape) == 1) or this.weights.shape[1] == 1):
            weights_ep = graph.new_ep("double")
            weights_ep.a = this.weights.ravel()
            graph.ep['weights'] = weights_ep
        return graph

    def to_dgl_graph(self):
        edge_index = torch.from_numpy(self.edges.T)
        return dgl.graph(data=(edge_index[0], edge_index[1]), num_nodes=self.num_nodes,
                         idtype=torch.int64, row_sorted=False)

    def to_pyg_graph(self):
        edge_attributes = torch.from_numpy(self.weights).unsqueeze(1) if self.is_weighted else None
        pyg_graph = pyg.data.Data(
            x=torch.from_numpy(self.node_attributes) if self.is_node_attributed else None,
            edge_index=torch.from_numpy(self.edges.T),
            edge_attr=edge_attributes
        )
        return pyg_graph

    def union(self, g2: 'SimpleGraph'):
        return union(self, g2)

    def save_csv(self, fp, sep, with_num_nodes, with_weights):
        data = pd.DataFrame({"source": self.edges[:, 0], "target": self.edges[:, 1]})
        if with_weights:
            weights = self.weights if self.weights is not None else np.ones(self.edges.shape[0])
            weights = pd.DataFrame(weights)
            data = pd.concat((data, weights), axis=1)
        if with_num_nodes:
            if "b" in fp.mode:
                fp.write(f"%{self.num_nodes}\n".encode("utf-8"))
            else:
                fp.write(f"%{self.num_nodes}\n")
        data.to_csv(fp, sep=sep, index=False, header=False)

    def save_npz(self, fp):
        data = {"num_nodes": self.num_nodes, "edge_index": self.edges, "directed": self.directed}
        if self.weights is not None:
            data["edge_attr"] = self.weights
        if self.node_attributes is not None:
            data["x"] = self.node_attributes
        np.savez(fp, **data)


def read_graph_from_spec(dataroot: str, dataset_spec: DatasetSpec):
    return SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)


class EDF:
    """ Empirical distribution function"""
    observations: np.ndarray
    edf: np.ndarray
    max_value: float
    min_value: float

    def __init__(self, observations: np.ndarray, min_value: float, max_value: float = None):
        if max_value is not None and max_value > np.max(observations):
            observations = np.append(observations, max_value)
        assert min_value <= np.min(observations)
        self.observations, counts = np.unique(observations, return_counts=True)
        self.edf = np.cumsum(counts) / len(observations)
        self.max_value = max_value
        self.min_value = min_value

    def sample(self, num_samples: int, rng: Generator):
        uni_samples = rng.uniform(0, 1, size=num_samples)
        sample_index = np.searchsorted(self.edf, uni_samples, side='right')

        zero_indices_mask = sample_index == 0
        num_zero = np.sum(zero_indices_mask)
        in_range_indices = sample_index[~zero_indices_mask]

        start_values = self.observations[in_range_indices - 1]
        end_values = self.observations[in_range_indices]
        values = start_values + rng.uniform(0, 1, size=len(start_values)) * (end_values - start_values)

        min_values = self.min_value + rng.uniform(0, 1, size=num_zero) * (self.observations[0] - self.min_value)

        results = rng.permutation(np.concatenate((values, min_values), axis=0))
        return results


def union(g1: SimpleGraph, g2: SimpleGraph):
    directed = g1.directed or g2.directed
    g2_index2new_map = node_index_map(g1.num_nodes, g2.num_nodes)
    new_edges = np.stack((g2_index2new_map[g2.edges[:, 0]], g2_index2new_map[g2.edges[:, 1]]), axis=1)
    edges = np.concatenate((g1.edges, new_edges), axis=0)
    weights = None
    if g1.weights is not None and g2.weights is not None:
        weights = np.concatenate((g1.weights, g2.weights), axis=0)

    node_attributes = None
    if g1.node_attributes is not None and g2.node_attributes is not None:
        node_attributes = np.concatenate((g1.node_attributes, g2.node_attributes), axis=0)

    return SimpleGraph(num_nodes=g1.num_nodes + g2.num_nodes, edges=edges, directed=directed,
                       weights=weights, node_attributes=node_attributes)


def union_edges_only(g1: SimpleGraph, g2: SimpleGraph):
    assert g1.num_nodes == g2.num_nodes
    assert g1.directed == g2.directed
    edges = np.concatenate((g1.edges, g2.edges), axis=0)
    weights = None
    if g1.weights is not None and g2.weights is not None:
        weights = np.concatenate((g1.weights, g2.weights), axis=0)

    return SimpleGraph(num_nodes=g1.num_nodes, edges=edges, directed=g1.directed,
                       weights=weights, node_attributes=g1.node_attributes)


def add_symmetric_edges(graph: SimpleGraph):
    data = pd.concat((pd.DataFrame({"source": graph.edges[:, 0], "target": graph.edges[:, 1]}),
                      pd.DataFrame({"source": graph.edges[:, 1], "target": graph.edges[:, 0]})))

    if graph.weights is not None:
        weights = pd.DataFrame(graph.weights)
        data = pd.concat((data, weights), axis=1)
    data = data.groupby(["source", "target"], as_index=False).agg("mean")
    edges = data.loc[:, ["source", "target"]].to_numpy().astype(np.int64)
    weights = data.iloc[:, 2:].to_numpy().astype(np.float64) if graph.weights is not None else None
    if weights is not None:
        weights = weights.squeeze()
    return SimpleGraph(num_nodes=graph.num_nodes, edges=edges, directed=graph.directed,
                       weights=weights, node_attributes=graph.node_attributes)


def make_edges_canonical(graph: SimpleGraph):
    assert not graph.directed
    edges = np.sort(graph.edges, axis=1)
    data = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})
    if graph.weights is not None:
        weights = pd.DataFrame(graph.weights)
        data = pd.concat((data, weights), axis=1)
    data = data.groupby(["source", "target"], as_index=False).agg("mean")
    edges = data.loc[:, ["source", "target"]].to_numpy().astype(np.int64)
    weights = data.iloc[:, 2:].to_numpy().astype(np.float64) if graph.weights is not None else None
    if weights is not None:
        weights = weights.squeeze()
    return SimpleGraph(num_nodes=graph.num_nodes, edges=edges, directed=False,
                       weights=weights, node_attributes=graph.node_attributes)


def make_unweighted(graph: SimpleGraph):
    weights = np.ones((len(graph.weights),), dtype=graph.weights.dtype)
    return SimpleGraph(num_nodes=graph.num_nodes, edges=graph.edges, directed=graph.directed,
                       weights=weights, node_attributes=graph.node_attributes)


def add_noise_edges(graph: SimpleGraph, p: float, rng: Generator, exclude_self_loops: bool = True):
    """
    Adds noise edges to the graph. Noise edges are not allowed to previously exist in the graph. For undirected graphs,
    a canonical graph is returned where edges[0,:] <= edges[1,:].
    Apply `add_symmetric_edges` to add the symmetric edges.
    Args:
        graph: Graph to add edges to
        p: Proportion noise edges to existing edges
        rng: Numpy random generator for reproducibility
        exclude_self_loops: If self-loops to be removed from directed graphs
    Returns:
        new_graph: Graph with added noise edges
        actual_p: The measured fraction of noise edges. May be smaller than p if not enough edges could be added.
    """
    if graph.directed:
        return add_noise_edges_directed(graph, p=p, rng=rng, exclude_self_loops=exclude_self_loops)
    else:
        return add_noise_edges_undirected(graph, p=p, rng=rng, exclude_self_loops=exclude_self_loops)


def add_noise_edges_undirected(graph: SimpleGraph, p: float, rng: Generator, exclude_self_loops: bool = True):
    graph = make_edges_canonical(graph)
    edges = np.unique(np.sort(graph.edges, axis=1), axis=0)
    num_noise_edges = int(p * edges.shape[0])
    noise_edges = sample_noise_edges_undirected(edges, num_noise_edges, graph.num_nodes, max_attepts=5, rng=rng,
                                                exclude_self_loops=exclude_self_loops)
    num_noise_edges = noise_edges.shape[0]
    actual_p = num_noise_edges / edges.shape[0]
    edges = np.concatenate((edges, noise_edges), axis=0)
    weights = None
    if graph.weights is not None:
        if graph.weights.ndim > 1:
            raise NotImplementedError("Sampling noise weights not implemented for multiple weights")
        edf = EDF(graph.weights, min_value=0)
        noise_weights = edf.sample(num_noise_edges, rng=rng)
        weights = np.concatenate((graph.weights, noise_weights), axis=0)
    new_graph = SimpleGraph(graph.num_nodes, edges, directed=False, weights=weights,
                            node_attributes=graph.node_attributes)
    new_graph = add_symmetric_edges(new_graph)  # Ensure both directions of new edges are added
    return new_graph, actual_p


def add_noise_edges_directed(graph: SimpleGraph, p: float, rng: Generator, exclude_self_loops: bool = True):
    num_noise_edges = int(p * graph.edges.shape[0])
    noise_edges = sample_noise_edges_directed(graph.edges, num_noise_edges, graph.num_nodes, max_attepts=5, rng=rng,
                                              exclude_self_loops=exclude_self_loops)
    num_noise_edges = noise_edges.shape[0]
    actual_p = num_noise_edges / graph.edges.shape[0]
    edges = np.concatenate((graph.edges, noise_edges), axis=0)
    weights = None
    if graph.weights is not None:
        if graph.weights.ndim > 1:
            raise NotImplementedError("Sampling noise weights not implemented for multiple weights")
        edf = EDF(graph.weights, min_value=0)
        noise_weights = edf.sample(num_noise_edges, rng=rng)
        weights = np.concatenate((graph.weights, noise_weights), axis=0)
    new_graph = SimpleGraph(graph.num_nodes, edges, directed=graph.directed, weights=weights,
                            node_attributes=graph.node_attributes)
    return new_graph, actual_p


def remove_edges_noise(graph: SimpleGraph, p: float, rng: Generator):
    if graph.directed:
        return remove_edges_noise_directed(graph, p=p, rng=rng)
    else:
        return remove_edges_noise_undirected(graph, p=p, rng=rng)


def remove_edges_noise_undirected(graph: SimpleGraph, p: float, rng: Generator):
    graph, actual_p = remove_edges_noise_directed(make_edges_canonical(graph), p=p, rng=rng)
    graph = add_symmetric_edges(graph)
    return graph, actual_p


def remove_edges_noise_directed(graph: SimpleGraph, p: float, rng: Generator):
    num_existing_edges = graph.edges.shape[0]
    num_noise_edges = int(p * num_existing_edges)
    edges_to_remove = rng.choice(num_existing_edges, size=num_noise_edges, replace=False)
    mask = np.ones((num_existing_edges,), dtype=bool)
    mask[edges_to_remove] = 0
    edges = graph.edges[mask, :]
    actual_p = edges_to_remove.shape[0] / num_existing_edges
    weights = None
    if graph.weights is not None:
        weights = graph.weights[mask]
    new_graph = SimpleGraph(graph.num_nodes, edges, directed=graph.directed,
                            weights=weights, node_attributes=graph.node_attributes)
    return new_graph, actual_p


def sample_noise_edges_undirected(current_edges: np.ndarray, num_add: int, num_nodes: int, max_attepts: int,
                                  rng: Generator, exclude_self_loops: bool = True) -> np.ndarray:
    """
    Sample new edges not previously in the graph.

    Args:
        current_edges: Array of current edges, [num_edges x 2]
        num_add: Number of edges to add
        num_nodes: Number of nodes in the graph
        max_attepts: Maximum number of attempt to generate non-existing and unique edges
        rng: Numpy Generator for reproducibility
        exclude_self_loops: Remove generated self-loop edges

    Returns:
        new_edges: New unique edges not previously in the graph
    """
    if exclude_self_loops:
        current_edges = current_edges[current_edges[:, 0] != current_edges[:, 1], :]  # Remove self-loops if exists

    forbidden_to_add = {(s, t) for s, t in np.sort(current_edges, axis=1)}
    edges_to_add = set()
    num_attempts = 0
    while len(edges_to_add) < num_add and num_attempts < max_attepts:
        sources = rng.integers(low=0, high=num_nodes, size=num_add)
        targets = rng.integers(low=0, high=num_nodes, size=num_add)
        new_edges = {(s, t) if s < t else (t, s) for s, t in zip(sources, targets) if s != t or not exclude_self_loops}
        edges_to_add = edges_to_add.union(new_edges - forbidden_to_add)
        num_attempts += 1
    index = rng.permutation(len(edges_to_add))[:num_add]
    new_edges = np.asarray(list(edges_to_add), dtype=np.int64)[index]
    return new_edges


def sample_noise_edges_directed(current_edges: np.ndarray, num_add: int, num_nodes: int, max_attepts: int,
                                rng: Generator, exclude_self_loops: bool = True) -> np.ndarray:
    """
    Sample new edges not previously in the graph.

    Args:
        current_edges: Array of current edges, [num_edges x 2]
        num_add: Number of edges to add
        num_nodes: Number of nodes in the graph
        max_attepts: Maximum number of attempt to generate non-existing and unique edges
        rng: Numpy Generator for reproducibility
        exclude_self_loops: Remove generated self-loop edges

    Returns:
        new_edges: New unique edges not previously in the graph
    """
    if exclude_self_loops:
        current_edges = current_edges[current_edges[:, 0] != current_edges[:, 1], :]  # Remove self-loops if exists

    forbidden_to_add = {(s, t) for s, t in current_edges}
    edges_to_add = set()
    num_attempts = 0
    while len(edges_to_add) < num_add and num_attempts < max_attepts:
        sources = rng.integers(low=0, high=num_nodes, size=num_add)
        targets = rng.integers(low=0, high=num_nodes, size=num_add)
        new_edges = {(s, t) for s, t in zip(sources, targets) if s != t or not exclude_self_loops}
        edges_to_add = edges_to_add.union(new_edges - forbidden_to_add)
        num_attempts += 1
    index = rng.permutation(len(edges_to_add))[:num_add]
    new_edges = np.asarray(list(edges_to_add), dtype=np.int64)[index]
    return new_edges


def node_index_map(g1_num_nodes: int, g2_num_nodes: int):
    return pd.Series(np.arange(g2_num_nodes, dtype=np.int64) + g1_num_nodes)
