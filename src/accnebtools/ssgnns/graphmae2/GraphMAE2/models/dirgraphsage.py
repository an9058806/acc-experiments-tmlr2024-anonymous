import torch
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from ..utils import create_activation, create_norm


class DirGraphSAGE(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 activation,
                 feat_drop,
                 norm,
                 encoding=False,
                 ):
        super(DirGraphSAGE, self).__init__()
        self.out_dim = out_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.sage_layers = nn.ModuleList()
        self.activation = activation

        last_activation = create_activation(activation) if encoding else None
        last_norm = norm if encoding else None

        hidden_in = in_dim
        hidden_out = out_dim

        if num_layers == 1:
            self.sage_layers.append(DirSAGEConv(hidden_in, hidden_out, 'mean', feat_drop, norm=last_norm))
        else:
            # input projection (no residual)
            self.sage_layers.append(DirSAGEConv(
                hidden_in, num_hidden, 'mean', feat_drop,
                activation=create_activation(activation), norm=norm))
            # hidden layers

            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.sage_layers.append(DirSAGEConv(
                    num_hidden, activation=create_activation(activation), norm=norm))

            # output projection
            self.sage_layers.append(DirSAGEConv(
                num_hidden, hidden_out, 'mean',
                feat_drop, activation=last_activation, norm=last_norm))
        self.head = nn.Identity()

    def forward(self, g, inputs):
        h = inputs

        for l in range(self.num_layers):
            h = self.sage_layers[l](g, h)

        if self.head is not None:
            return self.head(h)
        else:
            return h

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.is_pretraining = False
        self.head = nn.Linear(self.out_dim, num_classes)


class DirSAGEConv(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)

        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{(l+1)})

    If a weight tensor on each edge is provided, the aggregation becomes:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{e_{ji} h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that :math:`e_{ji}` is broadcastable with :math:`h_j^{l}`.

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.

        SAGEConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer applies on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    """

    def __init__(
            self,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=True,
            norm=None,
            activation=None,
    ):
        super(DirSAGEConv, self).__init__()
        valid_aggre_types = {"mean"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(3 * self._in_dst_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward_one_dir(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                # graph.srcdata["h"] = (
                #     self.fc_neigh(feat_src) if lin_before_mp else feat_src
                # )
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                # if not lin_before_mp:
                #     h_neigh = self.fc_neigh(h_neigh)
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            # if self._aggre_type == "gcn":
            #     rst = h_neigh
            #     # add bias manually for GCN
            #     if self.bias is not None:
            #         rst = rst + self.bias
            # else:
            #     rst = self.fc_self(h_self) + h_neigh

            # # activation
            # if self.activation is not None:
            #     rst = self.activation(rst)
            # # normalization
            # if self.norm is not None:
            #     rst = self.norm(rst)
            return h_self, h_neigh

    def forward(self, graph, feat):
        h_self, fwd_neigh = self.forward_one_dir(graph, feat)
        _, bwd_neigh = self.forward_one_dir(graph.reverse(), feat)
        torch.cat((h_self, fwd_neigh, bwd_neigh), dim=0)
        rst = self.fc_self(torch.cat((h_self, fwd_neigh, bwd_neigh), dim=1))
        if self.activation:
            rst = self.activation(rst)
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
