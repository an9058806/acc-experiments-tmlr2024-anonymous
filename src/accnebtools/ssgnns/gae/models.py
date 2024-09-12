import torch
import torch.nn as nn
from torch_geometric.nn import DirGNNConv, Sequential
from ..bgrl.bgrl.gcn_fixed import GCNConv


class Encoder(nn.Module):

    def __init__(self, n_in, n_h, num_layers: int, directed: bool):
        super(Encoder, self).__init__()
        layers = []
        for l in range(num_layers):
            gcn = GCNConv(in_channels=n_in if l == 0 else n_h, out_channels=n_h)
            if directed:
                gcn = DirGNNConv(gcn)
            layers.append((gcn, 'x, edge_index -> x'))
            if l < num_layers - 1:
                layers.append(nn.PReLU())

        self.gcn = Sequential('x, edge_index', layers)

    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)
