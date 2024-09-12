import dataclasses as dc
import torch

import accnebtools.data.graph as dgraph
from accnebtools.ssgnns.utils import SSGNNTrainer, get_features

import torch.nn.functional as F

from .GREET.utils import augmentation, generate_random_node_pairs
from .GREET.data_loader import get_structural_encoding
from .GREET.model import GCL, Edge_Discriminator


@dc.dataclass(frozen=True)
class AugmentParams:
    maskfeat_rate_1: float = 0.1
    maskfeat_rate_2: float = 0.5
    dropedge_rate_1: float = 0.5
    dropedge_rate_2: float = 0.1
    sparse: bool = True


@dc.dataclass(frozen=True)
class DiscParams:
    margin_hom: float = 0.5
    margin_het: float = 0.5


@dc.dataclass(frozen=True)
class Parameters:
    aug_params: AugmentParams
    disc_params: DiscParams
    lr_gcl: float = 0.001
    lr_disc: float = 0.001
    cl_rounds: int = 2
    wd: float = 0.0
    dropout: float = 0.5
    alpha: float = 0.1
    nlayers_enc: int = 2
    nlayers_proj: int = 1
    emb_dim: int = 512
    proj_dim: int = 512
    cl_batch_size: int = 0
    num_neighbours: int = 20  # number of neighbors of knn augmentation
    sparse: bool = True
    add_degree: bool = True
    add_lcc: bool = True
    standardize: bool = True


def train_cl(cl_model, discriminator, optimizer_cl, features, str_encodings, edges, aug_params: AugmentParams):
    cl_model.train()
    discriminator.eval()

    adj_1, adj_2, weights_lp, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
    features_1, adj_1, features_2, adj_2 = augmentation(features, adj_1, features, adj_2, aug_params, cl_model.training)
    cl_loss = cl_model(features_1, adj_1, features_2, adj_2)

    optimizer_cl.zero_grad()
    cl_loss.backward()
    optimizer_cl.step()

    return cl_loss.item()


def train_discriminator(cl_model, discriminator, optimizer_disc, features, str_encodings, edges, disc_params):
    cl_model.eval()
    discriminator.train()

    adj_1, adj_2, weights_lp, weights_hp = discriminator(torch.cat((features, str_encodings), 1), edges)
    rand_np = generate_random_node_pairs(features.shape[0], edges.shape[1]).to(features.device)
    psu_label = torch.ones(edges.shape[1]).to(features.device)

    embedding = cl_model.get_embedding(features, adj_1, adj_2)
    edge_emb_sim = F.cosine_similarity(embedding[edges[0]], embedding[edges[1]])

    rnp_emb_sim_lp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_lp = F.margin_ranking_loss(edge_emb_sim, rnp_emb_sim_lp, psu_label, margin=disc_params.margin_hom,
                                    reduction='none')
    loss_lp *= torch.relu(weights_lp - 0.5)

    rnp_emb_sim_hp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim, psu_label, margin=disc_params.margin_het,
                                    reduction='none')
    loss_hp *= torch.relu(weights_hp - 0.5)

    rank_loss = (loss_lp.mean() + loss_hp.mean()) / 2

    optimizer_disc.zero_grad()
    rank_loss.backward()
    optimizer_disc.step()

    return rank_loss.item()


class GREETTrainer(SSGNNTrainer):

    def __init__(self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device):
        self.params = params
        self.aug_params = params.aug_params
        self.disc_params = params.disc_params
        self.edges = torch.from_numpy(graph.edges.T)
        self.num_nodes = graph.num_nodes
        self.feat = get_features(graph, add_degree=params.add_degree, add_lcc=params.add_lcc,
                                 standardize=params.standardize)
        nfeats = self.feat.shape[1]

        self.str_encodings = get_structural_encoding(self.edges, self.num_nodes)
        self.cl_model = GCL(nlayers=params.nlayers_enc, nlayers_proj=params.nlayers_proj, in_dim=nfeats,
                            emb_dim=params.emb_dim // 2,
                            proj_dim=params.proj_dim // 2, dropout=params.dropout, sparse=params.sparse,
                            batch_size=params.cl_batch_size)
        self.cl_model.set_mask_knn(self.feat.cpu(), k=params.num_neighbours, dataset="")
        self.discriminator = Edge_Discriminator(self.num_nodes, nfeats + self.str_encodings.shape[1], params.alpha,
                                                params.sparse)

        self.feat = self.feat.to(device)
        self.edges = self.edges.to(device)
        self.cl_model = self.cl_model.to(device)
        self.discriminator = self.discriminator.to(device)
        self.str_encodings = self.str_encodings.to(device)

        self.optimizer_cl = torch.optim.Adam(self.cl_model.parameters(),
                                             lr=params.lr_gcl, weight_decay=params.wd)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=params.lr_disc, weight_decay=params.wd)

        self.device = device

    def step(self, step: int):
        total_loss = 0.
        for _ in range(self.params.cl_rounds):
            cl_loss = train_cl(self.cl_model, self.discriminator, self.optimizer_cl,
                               self.feat, self.str_encodings, self.edges, self.aug_params)
            total_loss += cl_loss
        rank_loss = train_discriminator(self.cl_model, self.discriminator, self.optimizer_discriminator,
                                        self.feat, self.str_encodings, self.edges, self.disc_params)
        return total_loss / self.params.cl_rounds

    def get_embeddings(self) -> torch.Tensor:
        self.cl_model.eval()
        self.discriminator.eval()
        adj_1, adj_2, _, _ = self.discriminator(torch.cat((self.feat, self.str_encodings), 1), self.edges)
        embedding = self.cl_model.get_embedding(self.feat, adj_1, adj_2)
        return embedding
