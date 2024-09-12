from typing import Final, Tuple, Union, Optional

import torch_geometric as pyg
from torch_geometric.nn.conv import (
    MessagePassing,
    SAGEConv,
)


class GraphSAGE(pyg.nn.models.basic_gnn.BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            use_dir_wrapper: bool,
            out_channels: Optional[int] = None,
            **kwargs,
    ):
        self.use_dir_wrapper = use_dir_wrapper
        super(GraphSAGE, self).__init__(in_channels=in_channels, hidden_channels=hidden_channels,
                                        num_layers=num_layers, out_channels=out_channels, **kwargs)

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        conv = SAGEConv(in_channels, out_channels, **kwargs)
        if self.use_dir_wrapper:
            conv = pyg.nn.conv.DirGNNConv(conv)
        return conv

    def forward(self, data, **kwargs):
        return super().forward(data.x, data.edge_index)
