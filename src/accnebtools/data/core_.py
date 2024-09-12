import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx


def try_read_num_nodes(filename):
    num_nodes = None
    comment_char = None
    with open(filename) as f:
        first_line = f.readline()
    if first_line[0] in "#%":
        comment_char = first_line[0]
        try:
            num_nodes = int(first_line.strip(comment_char + "\n "))
        except ValueError:
            pass
    return num_nodes, comment_char


def read_nxgraph(filename, filetype, is_weighted, directed, num_nodes=None, remove_self_loops=True, **pd_kwargs):
    edges, weights, num_nodes = read_edges(filename,
                                           filetype,
                                           is_weighted=is_weighted,
                                           directed=directed,
                                           num_nodes=num_nodes,
                                           remove_self_loops=remove_self_loops,
                                           **pd_kwargs)
    return edges2nx(edges, weights, num_nodes=num_nodes, directed=directed)


def edges2nx(edges, weights, num_nodes, directed):
    # type: (np.ndarray, np.ndarray, int, bool) -> nx.Graph
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    graph.add_nodes_from(range(num_nodes))
    edges = [(int(u), int(v), w) for (u, v), w in zip(edges, weights)]
    graph.add_weighted_edges_from(edges)
    return graph


def read_spmat(filename, filetype, is_weighted, directed, num_nodes=None, remove_self_loops=True, **pd_kwargs):
    edges, weights, num_nodes = read_edges(filename,
                                           filetype,
                                           is_weighted=is_weighted,
                                           directed=directed,
                                           num_nodes=num_nodes,
                                           remove_self_loops=remove_self_loops,
                                           **pd_kwargs)
    return edges2spmat(edges, weights, num_nodes=num_nodes, directed=directed)


def edges2spmat(edges, weights=None, num_nodes=None, directed=True):
    if num_nodes is None:
        num_nodes = np.max(edges) + 1

    if weights is None:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    mat = sp.coo_matrix((weights, (edges[:, 1], edges[:, 0])), shape=[num_nodes, num_nodes]).tocsc()

    if not directed:
        mat = mat.maximum(mat.T)

    return mat


def read_edges(filename, filetype, is_weighted,
               as_canonical_undirected, add_symmetrical_edges,
               num_nodes=None, remove_self_loops=True, **pd_kwargs):
    num_nodes_ = None
    comment_char = None
    if filetype in {"csv", "tsv", "edgelist"}:
        num_nodes_, comment_char = try_read_num_nodes(filename)
    num_nodes = num_nodes_ if num_nodes_ is not None else num_nodes

    if "index_col" not in pd_kwargs:
        pd_kwargs["index_col"] = False

    if "header" not in pd_kwargs:
        pd_kwargs["header"] = None

    if "comment" not in pd_kwargs:
        pd_kwargs["comment"] = comment_char

    if filetype in {'csv', 'tsv', 'edgelist', 'txt'}:
        if filetype == "csv":
            pd_kwargs["sep"] = ","
        elif filetype == "tsv" or filetype == "edgelist" or filetype == "twitter_txt":
            pd_kwargs["sep"] = "\s+"

        edges, weights = _read_tex_to_edges(filename,
                                            is_weighted=is_weighted,
                                            as_canonical_undirected=as_canonical_undirected,
                                            add_symmetrical_edges=add_symmetrical_edges,
                                            remove_self_loops=remove_self_loops, **pd_kwargs)
    elif filetype == 'parquet':
        edges, weights = _read_parquet_to_edges(filename,
                                                is_weighted=is_weighted,
                                                as_canonical_undirected=as_canonical_undirected,
                                                add_symmetrical_edges=add_symmetrical_edges,
                                                remove_self_loops=remove_self_loops)
    else:
        raise NotImplementedError

    if num_nodes is None:
        num_nodes = np.max(edges) + 1

    return edges, weights, num_nodes


def _read_parquet_to_edges(filename, is_weighted, as_canonical_undirected, add_symmetrical_edges,
                           remove_self_loops=True, **pd_kwargs):
    df = pd.read_parquet(filename, **pd_kwargs)
    return _post_load(df, is_weighted=is_weighted, as_canonical_undirected=as_canonical_undirected,
                      add_symmetrical_edges=add_symmetrical_edges,
                      remove_self_loops=remove_self_loops)


def _read_tex_to_edges(filename, is_weighted, as_canonical_undirected, add_symmetrical_edges,
                       remove_self_loops=True, **pd_kwargs):
    df = pd.read_csv(filename, **pd_kwargs)
    return _post_load(df, is_weighted=is_weighted, as_canonical_undirected=as_canonical_undirected,
                      add_symmetrical_edges=add_symmetrical_edges,
                      remove_self_loops=remove_self_loops)


def fix_edge_orientation(edges):
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    return edges


def read_graph_from_npz(fp, as_canonical_undirected, add_symmetrical_edges=False, remove_self_loops=True, use_weights=True):
    with np.load(fp, allow_pickle=False) as data:
        # Get either 'edge_index' or 'edges_index' from data.
        if 'edge_index' in data:
            edges = data['edge_index']
        elif 'edges_index' in data:
            edges = data['edges_index']
        else:
            edges = None
        edges = fix_edge_orientation(edges)
        node_attributes = data['x'] if 'x' in data else None
        num_nodes = data['num_nodes'] if 'num_nodes' in data else None
        if num_nodes is None and node_attributes is None:
            num_nodes = edges.max() + 1
        elif num_nodes is None:
            num_nodes = node_attributes.shape[0]
        else:
            num_nodes = num_nodes.item()
        weights = data['edge_attr'] if 'edge_attr' in data and use_weights else None
        directed = data['directed'] if 'directed' in data else False
        try:
            directed = directed.item()
        except AttributeError:
            pass
        directed = not as_canonical_undirected and directed

    data = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})
    if weights is not None:
        weights = pd.DataFrame(weights)
        data = pd.concat((data, weights), axis=1)
    edges, weights = _post_load(data, is_weighted=weights is not None,
                                as_canonical_undirected=not directed,
                                add_symmetrical_edges=add_symmetrical_edges,
                                remove_self_loops=remove_self_loops)

    return num_nodes, edges, weights, node_attributes, directed


def _post_load(df, is_weighted, as_canonical_undirected, add_symmetrical_edges, remove_self_loops=True):
    if remove_self_loops:
        df = df.loc[df.iloc[:, 0] != df.iloc[:, 1], :]
    edges = df.iloc[:, [0, 1]].to_numpy().astype(np.int64)
    if is_weighted and df.shape[1] > 2:
        weights = df.iloc[:, 2:].to_numpy().astype(np.float64)
        weights = weights.squeeze(axis=1) if weights.ndim > 1 and weights.shape[1] == 1 else weights
    else:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    if as_canonical_undirected:
        edges = np.sort(edges, axis=1)
        df = pd.DataFrame({'source': edges[:, 0], 'target': edges[:, 1]})
        weights_df = pd.DataFrame(weights)
        df = pd.concat((df, weights_df), axis=1)
        df = df.groupby(['source', 'target'], as_index=False).agg('mean')
        edges = df.loc[:, ['source', 'target']].to_numpy().astype(np.int64)
        weights = df.iloc[:, 2:].to_numpy().astype(np.float64)
        weights = weights.squeeze(axis=1) if weights.ndim > 1 and weights.shape[1] == 1 else weights

        if add_symmetrical_edges:
            sym_edges = np.stack((edges[:, 1], edges[:, 0]), axis=1)
            edges = np.concatenate((edges, sym_edges), axis=0)
            weights = np.concatenate((weights, weights), axis=0)

    return edges, weights


def read_flow_graph_from_npz(fp, as_canonical_undirected, use_weights_as_flow=False, remove_self_loops=True):
    with np.load(fp, allow_pickle=False) as data:
        # Get either 'edge_index' or 'edges_index' from data.
        if 'edge_index' in data:
            edges = data['edge_index']
        elif 'edges_index' in data:
            edges = data['edges_index']
        else:
            edges = None
        if edges is None:
            raise ValueError("Edges do not exist.")
        edges = fix_edge_orientation(edges)
        node_attributes = data['x'] if 'x' in data else None
        num_nodes = data['num_nodes'] if 'num_nodes' in data else None
        if num_nodes is None and node_attributes is None:
            num_nodes = edges.max() + 1
        elif num_nodes is None:
            num_nodes = node_attributes.shape[0]
        else:
            num_nodes = num_nodes.item()
        if use_weights_as_flow:
            flow = data['edge_attr'] if 'edge_attr' in data else None
        else:
            flow = data['flow'] if 'flow' in data else None
        if flow is None:
            raise ValueError("Flow does not exist.")
        weights = data['edge_attr'] if 'edge_attr' in data else np.ones(edges.shape[0], dtype=np.float32)
        directed = data['directed'] if 'directed' in data else True
        try:
            directed = directed.item()
        except AttributeError:
            pass
        directed = not as_canonical_undirected and directed

        counts = data['counts'] if 'counts' in data else np.ones(edges.shape[0], dtype=np.float32)

    data = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1], "flow": flow, "counts": counts})

    weights = pd.DataFrame(weights)
    data = pd.concat((data, weights), axis=1)
    edges, flow, counts, weights = _post_load_flow(data, has_counts=True, has_weights=True,
                                                   as_canonical_undirected=not directed,
                                                   remove_self_loops=remove_self_loops)

    return num_nodes, edges, flow, counts, weights, node_attributes, directed


def _post_load_flow(df, has_counts, has_weights, as_canonical_undirected, remove_self_loops=True):
    if remove_self_loops:
        df = df.loc[df.iloc[:, 0] != df.iloc[:, 1], :]
    edges = df.iloc[:, [0, 1]].to_numpy().astype(np.int64)
    flow = df.iloc[:, 2].to_numpy().astype(np.float64)
    if has_counts:
        counts = df.iloc[:, 3].to_numpy().astype(np.int64)
    else:
        counts = np.ones(edges.shape[0], dtype=np.float64)
    if has_weights:
        weights = df.iloc[:, 3 + int(has_counts):].to_numpy().astype(np.float64)
        weights = weights.squeeze(axis=1) if weights.ndim > 1 and weights.shape[1] == 1 else weights
    else:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    if as_canonical_undirected:
        non_canonical_order = edges[:, 0] > edges[:, 1]
        edges[non_canonical_order, :] = edges[non_canonical_order, ::-1]
        flow[non_canonical_order] = -flow[non_canonical_order]
        df = pd.DataFrame({'source': edges[:, 0], 'target': edges[:, 1], 'flow': flow,
                           'counts': counts})
        df = df.groupby(['source', 'target'], as_index=False, sort=True).agg('sum')
        edges = df.loc[:, ['source', 'target']].to_numpy().astype(np.int64)
        flow = df.iloc[:, 2].to_numpy().astype(np.float64)
        counts = df.iloc[:, 3].to_numpy().astype(np.int64)

        df = pd.DataFrame({'source': edges[:, 0], 'target': edges[:, 1]})
        weights_df = pd.DataFrame(weights)
        df = pd.concat((df, weights_df), axis=1)
        df = df.groupby(['source', 'target'], as_index=False, sort=True).agg('mean')

        weights = df.iloc[:, 2:].to_numpy().astype(np.float64)
        weights = weights.squeeze(axis=1) if weights.ndim > 1 and weights.shape[1] == 1 else weights

    return edges, flow, counts, weights


def read_flow_edges(filename, filetype, as_canonical_undirected, num_nodes=None, remove_self_loops=True, **pd_kwargs):
    num_nodes_ = None
    comment_char = None
    if filetype in {"csv", "tsv", "edgelist"}:
        num_nodes_, comment_char = try_read_num_nodes(filename)
    num_nodes = num_nodes_ if num_nodes_ is not None else num_nodes

    if "index_col" not in pd_kwargs:
        pd_kwargs["index_col"] = False

    if "header" not in pd_kwargs:
        pd_kwargs["header"] = None

    if "comment" not in pd_kwargs:
        pd_kwargs["comment"] = comment_char

    if filetype in {'csv', 'tsv', 'edgelist', 'txt'}:
        if filetype == "csv":
            pd_kwargs["sep"] = ","
        elif filetype == "tsv" or filetype == "edgelist" or filetype == "twitter_txt":
            pd_kwargs["sep"] = "\s+"

        df = pd.read_csv(filename, **pd_kwargs)
        edges, flow, counts, weights = _post_load_flow(df, has_counts=False, has_weights=False,
                                                       as_canonical_undirected=as_canonical_undirected,
                                                       remove_self_loops=remove_self_loops)

    elif filetype == 'parquet':
        df = pd.read_parquet(filename, **pd_kwargs)
        edges, flow, counts, weights = _post_load_flow(df, has_counts=False, has_weights=False,
                                                       as_canonical_undirected=as_canonical_undirected,
                                                       remove_self_loops=remove_self_loops)
    else:
        raise NotImplementedError

    if num_nodes is None:
        num_nodes = np.max(edges) + 1

    return edges, flow, counts, weights, num_nodes
