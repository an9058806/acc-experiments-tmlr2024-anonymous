import torch
import torch.nn as nn
from torch_geometric.nn import DirGNNConv, Sequential
from ..bgrl.bgrl.gcn_fixed import GCNConv


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits


class DGI(nn.Module):
    def __init__(self, n_in, n_h, num_layers: int, directed: bool):
        super(DGI, self).__init__()
        layers = []
        for i in range(num_layers):
            gcn = GCNConv(in_channels=n_in if i == 0 else n_h, out_channels=n_h)
            if directed:
                gcn = DirGNNConv(gcn)
            layers.append(
                (gcn, 'x, edge_index -> x')
            )
            layers.append(nn.PReLU())

        self.gcn = Sequential('x, edge_index', layers)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, x1, x2, edge_index, msk, samp_bias1, samp_bias2):

        h_1 = self.gcn(x1, edge_index)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(x2, edge_index)

        # ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return h_1, h_2, c

    # Detach the return variables
    def embed(self, x, edge_index, msk):
        h_1 = self.gcn(x, edge_index)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


# Applies an average on seq, of shape (nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)
