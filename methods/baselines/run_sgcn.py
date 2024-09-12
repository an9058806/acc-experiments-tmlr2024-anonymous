import os
import sys
import time
import json
import random
import numpy as np
from typing import Optional

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))

import accnebtools.argsfromconfig as parsing
import accnebtools.data.core_ as datacore

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

import structfeatures.main as stf


def gcn_norm(  # noqa: F811
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: Optional[int] = None,
        improved: bool = False,
        add_self_loops: bool = True,
        dtype: Optional[torch.dtype] = None,
):
    """Fixed version of where both in and out degrees are used for normalization."""
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        in_deg = torch_sparse.sum(adj_t, dim=1)
        out_deg = torch_sparse.sum(adj_t, dim=0)
        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        out_deg_inv_sqrt = out_deg.pow_(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0.)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, in_deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, out_deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        out_deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        in_deg = scatter(value, row, 0, dim_size=num_nodes, reduce='sum')
        out_deg_inv_sqrt = out_deg.pow_(-0.5)
        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)
        value = in_deg_inv_sqrt[row] * value * out_deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    src_nodes, dst_nodes = edge_index[0], edge_index[1]
    # row, col = edge_index[0], edge_index[1]
    # idx = col if flow == 'source_to_target' else row
    out_deg = scatter(edge_weight, src_nodes, dim=0, dim_size=num_nodes, reduce='sum')
    in_deg = scatter(edge_weight, dst_nodes, dim=0, dim_size=num_nodes, reduce='sum')
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)
    edge_weight = in_deg_inv_sqrt[dst_nodes] * edge_weight * out_deg_inv_sqrt[src_nodes]

    return edge_index, edge_weight


class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_x: Optional[Tensor]

    def __init__(self, K: int = 1, cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(K={self.K})')


def compute_embeddings(input_file, output_path, as_undirected, weighted, node_attributed, args, metadata_path=None):
    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file, as_canonical_undirected=as_undirected, add_symmetrical_edges=as_undirected, remove_self_loops=True
    )

    if not weighted:
        weights = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not node_attributed or node_attributes is None:
        node_attributes = None
    else:
        node_attributes = np.atleast_2d(node_attributes)

    bf_params = stf.BaseFeatureParams(
        use_weights=weighted,
        use_node_attributes=node_attributed and node_attributes is not None,
        as_undirected=not directed,
        use_degree=True,
        use_lcc=args.use_lcc,
        use_egonet_edge_counts=False,
        use_legacy_egonet_edge_counts=False
    )

    start = time.perf_counter()
    if args.initialize_with_ones:
        base_features = np.ones((num_nodes, 1), dtype=np.float32)
    else:
        _edge_index, weights, node_attributes = stf.prepare_inputs(edge_index=edges, num_nodes=num_nodes,
                                                                   bf_params=bf_params,
                                                                   weights=weights,
                                                                   node_attributes=node_attributes)
        base_features, _ = stf.get_structural_initial_features(edge_index=_edge_index, num_nodes=num_nodes,
                                                               bf_params=bf_params,
                                                               weights=weights, node_attributes=node_attributes)
        if args.standardize:
            std = np.std(base_features, axis=0)
            std[std == 0] = 1
            base_features = (base_features - np.mean(base_features, axis=0)) / std

    encoder = SGConv(K=args.num_layers)

    encoder = encoder.to(device)
    x = torch.from_numpy(base_features).to(device)

    edges = torch.from_numpy(edges.T).contiguous().to(device)

    with torch.no_grad():
        embeddings = encoder(x=x, edge_index=edges, edge_weight=weights).detach()
        if directed:
            embeddings_rev = encoder(x=x, edge_index=edges.flip(0), edge_weight=weights).detach()
            embeddings = 0.5 * embeddings + 0.5 * embeddings_rev
        embeddings = embeddings.cpu().numpy()
    duration = time.perf_counter() - start
    meta_data = vars(args)
    meta_data["duration"] = duration
    np.save(output_path, embeddings, allow_pickle=False)
    if metadata_path is not None:
        with open(metadata_path, 'w') as fp:
            json.dump(meta_data, fp)


def main():
    name = "sgcn"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", 'sgcn')
    parser.description = f"{name}: Simplified Graph Convolutional Neural Network."
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config = json.load(fp)
    else:
        config = vars(args)

    if config['seed'] is not None:
        np.random.seed(config['seed'])
        random.seed(config['seed'])

    stf_cache_dir = f"/tmp/gnn_stf_cache/cpu_{os.cpu_count()}/"
    os.makedirs(stf_cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = stf_cache_dir

    compute_embeddings(input_file=config['input_file'],
                       output_path=config['output_file'],
                       metadata_path=config['metadata'],
                       as_undirected=config['undirected'],
                       weighted=config['weighted'],
                       node_attributed=config['node_attributed'],
                       args=args,
                       )


if __name__ == "__main__":
    main()
