import dataclasses as dc
import torch
import torch.nn as nn

import accnebtools.data.graph as dgraph
from accnebtools.ssgnns.utils import SSGNNTrainer, get_features

from .models import DGI


@dc.dataclass(frozen=True)
class Parameters:
    num_epochs: int = 100
    lr: float = 0.001
    wd: float = 0.0
    num_layers: int = 2
    dimensions: int = 512
    batch_size: int = 512
    add_degree: bool = True
    add_lcc: bool = True
    standardize: bool = True


class DGITrainer(SSGNNTrainer):

    def __init__(self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device):
        enhanced_features = get_features(graph, add_degree=params.add_degree, add_lcc=params.add_lcc,
                                         standardize=params.standardize)
        self.data = graph.to_pyg_graph()
        self.data.x = enhanced_features
        self.num_nodes = graph.num_nodes

        self.params = params
        self.batch_size = params.batch_size
        self.model = DGI(enhanced_features.shape[1], params.dimensions, num_layers=params.num_layers,
                         directed=graph.directed)
        self.b_xent = nn.BCEWithLogitsLoss()

        self.data = self.data.to(device)
        self.model = self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.wd)
        self.device = device

    def step(self, step):
        self.model.train()
        self.optimizer.zero_grad()

        idx = torch.randperm(self.num_nodes, device=self.device)
        shuf_fts = self.data.x[idx, :]

        lbl_1 = torch.ones((self.batch_size,), device=self.device)
        lbl_2 = torch.zeros((self.batch_size,), device=self.device)
        # lbl = torch.cat((lbl_1, lbl_2), 0)

        h_1, h_2, c = self.model(self.data.x, shuf_fts, self.data.edge_index, None, None, None)

        total_loss = 0.
        for i in range(0, self.num_nodes, self.batch_size):
            start = i
            end = min(i + self.batch_size, self.num_nodes)
            real_batch_size = end - start
            logits = self.model.disc(c, h_1[start:end], h_2[start:end], None, None)
            lbl = torch.cat((lbl_1[:real_batch_size], lbl_2[:real_batch_size]), 0)
            loss = self.b_xent(logits, lbl)
            if end == self.num_nodes:
                loss.backward(retain_graph=False)
            else:
                loss.backward(retain_graph=True)
            total_loss += loss.item()
        self.optimizer.step()

        return total_loss

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        embeds, _ = self.model.embed(self.data.x, self.data.edge_index, None)
        return embeds
