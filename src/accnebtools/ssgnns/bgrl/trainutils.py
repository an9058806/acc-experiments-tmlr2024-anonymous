import dataclasses as dc
from typing import Optional
import copy

import torch
import torch.nn.functional as thfun

import accnebtools.data.graph as dgraph
from accnebtools.ssgnns.utils import SSGNNTrainer, get_features

from .bgrl.bgrl import BGRL, MLP_Predictor, get_graph_drop_transform, CosineDecayScheduler
from .bgrl.gcn_fixed import FixedGCN
from .gs_model import GraphSAGE


@dc.dataclass(frozen=True)
class Parameters:
    num_epochs: int = 10000
    lr: float = 1e-5
    wd: float = 1e-5
    mm: float = 0.99
    graph_encoder_layer: Optional[tuple[int, ...]] = (512, 256)
    drop_feat_p_1: float = 0.25
    drop_feat_p_2: float = 0.25
    drop_edge_p_1: float = 0.25
    drop_edge_p_2: float = 0.25
    predictor_hidden_size: int = 512
    encoder: str = 'gcn'
    add_degree: bool = True
    add_lcc: bool = True
    standardize: bool = True


class BGRLTrainer(SSGNNTrainer):

    def __init__(self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device):
        enhanced_features = get_features(graph, add_degree=params.add_degree, add_lcc=params.add_lcc,
                                         standardize=params.standardize)
        self.data = graph.to_pyg_graph()
        self.data.x = enhanced_features

        self.params = params
        self.transform_1 = get_graph_drop_transform(drop_edge_p=params.drop_edge_p_1, drop_feat_p=params.drop_feat_p_1)
        self.transform_2 = get_graph_drop_transform(drop_edge_p=params.drop_edge_p_2, drop_feat_p=params.drop_feat_p_2)

        input_size, representation_size = self.data.x.size(1), params.graph_encoder_layer[-1]
        if params.encoder == "gcn":
            encoder = FixedGCN([input_size] + list(params.graph_encoder_layer), directed=graph.directed,
                               batchnorm=True)  # 512, 256, 128
        elif params.encoder == "graphsage":
            encoder = GraphSAGE(in_channels=input_size, out_channels=representation_size,
                                hidden_channels=representation_size, num_layers=len(params.graph_encoder_layer),
                                use_dir_wrapper=graph.directed, norm='batch', act='prelu')
        else:
            raise NotImplementedError(f"Encoder '{params.encoder}' not implemented.")
        predictor = MLP_Predictor(representation_size, representation_size, hidden_size=params.predictor_hidden_size)
        self.model = BGRL(encoder, predictor).to(device)
        self.data = self.data.to(device)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.trainable_parameters(), lr=params.lr, weight_decay=params.wd)
        self.lr_scheduler = CosineDecayScheduler(params.lr, params.num_epochs // 10, params.num_epochs)
        self.mm_scheduler = CosineDecayScheduler(1 - params.mm, 0, params.num_epochs)

        self.device = device

    def step(self, step):
        self.model.train()

        # update learning rate
        lr = self.lr_scheduler.get(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - self.mm_scheduler.get(step)

        # forward
        self.optimizer.zero_grad()

        x1, x2 = self.transform_1(self.data), self.transform_2(self.data)

        q1, y2 = self.model(x1, x2)
        q2, y1 = self.model(x2, x1)

        loss = 2 - thfun.cosine_similarity(q1, y2.detach(), dim=-1).mean() - thfun.cosine_similarity(q2, y1.detach(),
                                                                                                     dim=-1).mean()
        loss.backward()

        # update online network
        self.optimizer.step()
        # update target network
        self.model.update_target_network(mm)

        return loss.item()

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        tmp_encoder = copy.deepcopy(self.model.online_encoder).eval()
        embeds = tmp_encoder(self.data)
        return embeds

    @torch.no_grad()
    def get_embeddings_for_graph(self, graph: dgraph.SimpleGraph) -> torch.Tensor:
        enhanced_features = get_features(graph, add_degree=self.params.add_degree, add_lcc=self.params.add_lcc,
                                         standardize=self.params.standardize)
        data = graph.to_pyg_graph()
        data.x = enhanced_features
        tmp_encoder = copy.deepcopy(self.model.online_encoder).eval()
        embeds = tmp_encoder(data.to(self.device))
        return embeds
