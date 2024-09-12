import dataclasses as dc
import torch
import torch.nn as nn

import torch_geometric.transforms as T
import accnebtools.data.graph as dgraph
from accnebtools.ssgnns.utils import SSGNNTrainer, get_features

from torch_geometric.nn import GAE
from .models import Encoder


@dc.dataclass(frozen=True)
class Parameters:
    num_epochs: int = 200
    lr: float = 1e-3
    wd: float = 0.0
    num_layers: int = 2
    dimensions: int = 512
    neg_sampling_ratio: int = 1
    add_degree: bool = True
    add_lcc: bool = True
    standardize: bool = True


class GAETrainer(SSGNNTrainer):

    def __init__(self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device):
        enhanced_features = get_features(graph, add_degree=params.add_degree, add_lcc=params.add_lcc,
                                         standardize=params.standardize)
        self.sample_links_transform = T.Compose([
            T.RandomLinkSplit(num_val=0., num_test=0., is_undirected=graph.is_undirected,
                              split_labels=True, add_negative_train_samples=True,
                              neg_sampling_ratio=params.neg_sampling_ratio),
        ])
        self.data = graph.to_pyg_graph()
        self.data.x = enhanced_features
        self.num_nodes = graph.num_nodes

        self.params = params
        self.encoder = Encoder(enhanced_features.shape[1], params.dimensions, num_layers=params.num_layers,
                               directed=graph.directed).to(device)
        self.model = GAE(self.encoder).to(device)
        self.b_xent = nn.BCEWithLogitsLoss()

        self.data = self.data.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.wd)
        self.device = device

    def step(self, step):
        self.model.train()
        self.optimizer.zero_grad()
        train_data, _, _ = self.sample_links_transform(self.data)
        z = self.model.encode(train_data.x, train_data.edge_index)
        loss = self.model.recon_loss(
            z,
            pos_edge_index=train_data.pos_edge_label_index,
            neg_edge_index=train_data.neg_edge_label_index
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        embeds = self.encoder(self.data.x, self.data.edge_index)
        return embeds
