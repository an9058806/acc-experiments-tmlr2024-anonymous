import dataclasses as dc
from accnebtools.algs.utils import EmbeddingAlgSpec, AlgGraphSupport


@dc.dataclass(frozen=True)
class NodeAttributesOnlyAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=False,
        weighted=False,
        node_attributed=False,
    )
    name: str = 'node_attribute_only'
    path: str = ''
    env_name: str = ''


@dc.dataclass(frozen=True)
class AccAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=True,
        node_attributed=True,
    )
    max_steps: int = 2  # Maximum number of embedding expansion iterations (L)
    normalization: str = 'none'  # Either 'sphere' or 'std'
    subtract_mean: bool = True  # Centralize features in every step.
    dtype32: bool = True  # Use single precision
    use_degree: bool = True  # Use the degree as initial feature.
    use_lcc: bool = True  # Use the local clustering coefficient features
    min_add_dim: int = 2  # Minimum number of dimensions of added aggregated features.
    dimensions: int = 512  # Final embedding dimension.
    sv_thresholding: str = 'rtol'  # Mode for determining the rank. Either 'none', 'tol', 'rtol', 'rank' or 'stwhp'.
    theta: float = 1e-8  # Tolerance for determining rank.
    return_us: bool = False  # Return U@S from rSVD rather than X @ Vh.t().
    use_rsvd: bool = False  # Use randomized SVD instead of full SVD
    name: str = 'acc'
    path: str = 'methods/acc_and_pcapass/run_acc.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class PcapassAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=True,
        node_attributed=True,
    )
    max_steps: int = 2  # Maximum number of embedding expansion iterations (L)
    normalization: str = 'none'  # Either 'sphere' or 'std'
    subtract_mean: bool = True  # Centralize features in every step.
    dtype32: bool = True  # Use single precision
    use_degree: bool = True  # Use the degree as initial feature.
    use_lcc: bool = True  # Use the local clustering coefficient features
    dimensions: int = 512  # Only used for svd and kmeans pruning methods. The maximum number of embeddings.
    sv_thresholding: str = 'none'  # Mode for determining the rank. Either 'none', 'tol', 'rtol', 'rank' or 'stwhp'.
    theta: float = 0.0  # Tolerance for determining rank.
    return_us: bool = False  # Return U @ S from svd compression rather than X @ V.t()
    use_rsvd: bool = False  # Use randomized SVD instead of full SVD
    name: str = 'pcapass'
    path: str = 'methods/acc_and_pcapass/run_pcapass.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class BgrlAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 10000  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-5  # Learning rate.
    wd: float = 1e-5  # Weight decay.
    mm: float = 0.99  # Momentum
    dfp1: float = 0.25  # feature drop out
    dfp2: float = 0.25  # feature drop out
    dep1: float = 0.25  # edge drop out
    dep2: float = 0.25  # edge drop out
    encoder: str = 'gcn'  # Which encoder to use.
    name: str = 'bgrl'
    path: str = 'methods/baselines/run_bgrl.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class Bgrl_gsAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 10000  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-5  # Learning rate.
    wd: float = 1e-5  # Weight decay.
    mm: float = 0.99  # Momentum
    dfp1: float = 0.25  # feature drop out
    dfp2: float = 0.25  # feature drop out
    dep1: float = 0.25  # edge drop out
    dep2: float = 0.25  # edge drop out
    encoder: str = 'graphsage'  # Which encoder to use.
    name: str = 'bgrl_gs'
    path: str = 'methods/baselines/run_bgrl.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class CcassgAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 100  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 0.0  # Weight decay.
    lambd: float = 1e-3  # Lambda loss parameter.
    dfr: float = 0.2  # feature drop out ratio.
    der: float = 0.2  # edge drop out ratio.
    dep1: float = 0.25  # feature drop out 1.
    name: str = 'ccassg'
    path: str = 'methods/baselines/run_ccassg.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class GraphmaeAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 1000  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 1e-4  # Weight decay.
    num_heads: int = 4  # Number of heads.
    mask_rate: float = 0.5  # Mask ratio.
    replace_rate: float = 0.05  # Replacement rate.
    alpha_l: int = 3  # Loss exponent parameter.
    name: str = 'graphmae'
    path: str = 'methods/baselines/run_graphmae.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class Graphmae2Alg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 1000  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 1e-4  # Weight decay.
    num_heads: int = 8  # Number of heads.
    mask_rate: float = 0.5  # Mask ratio.
    replace_rate: float = 0.05  # Replacement rate.
    alpha_l: int = 3  # Loss exponent parameter.
    lam: float = 1.0  # Loss weighting parameter.
    encoder: str = 'gat'  # Which encoder to use.
    name: str = 'graphmae2'
    path: str = 'methods/baselines/run_graphmae2.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class Graphmae2_gsAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 1000  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 1e-4  # Weight decay.
    num_heads: int = 8  # Number of heads.
    mask_rate: float = 0.5  # Mask ratio.
    replace_rate: float = 0.05  # Replacement rate.
    alpha_l: int = 3  # Loss exponent parameter.
    lam: float = 1.0  # Loss weighting parameter.
    encoder: str = 'graphsage'  # Which encoder to use.
    name: str = 'graphmae2_gs'
    path: str = 'methods/baselines/run_graphmae2.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class MvgrlAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 3000  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 0.001  # Learning rate.
    wd: float = 0.0  # Weight decay.
    sample_size: int = 2000  # Sample size.
    batch_size: int = 4  # Batch size.
    patience: int = 200  # Early stopping patience.
    name: str = 'mvgrl'
    path: str = 'methods/baselines/run_mvgrl.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class GreetAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 400  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 0.0  # Weight decay.
    dropout: float = 0.5  # Dropout
    num_neighbours: int = 20  # Number of neighbours for K-NN graph
    dfr1: float = 0.1  # edge drop out ratio.
    dfr2: float = 0.5  # edge drop out ratio.
    der1: float = 0.5  # edge drop out ratio.
    der2: float = 0.1  # edge drop out ratio.
    cl_rounds: int = 2  # Rounds before updating discriminator.
    margin_hom: float = 0.5  # HOM margin
    margin_het: float = 0.5  # HET margin
    alpha: float = 0.1  # Alpha
    name: str = 'greet'
    path: str = 'methods/baselines/run_greet.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class SpgclAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 500  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 0.001  # Learning rate.
    wd: float = 0.0001  # Weight decay.
    subg_num_hops: int = 4  # Num hops when extracting subgraph
    name: str = 'spgcl'
    path: str = 'methods/baselines/run_spgcl.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class DgiAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 100  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 0.0  # Weight decay.
    name: str = 'dgi'
    path: str = 'methods/baselines/run_dgi.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class GaeAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    num_epochs: int = 200  # Number of training epochs
    num_layers: int = 2  # Number of layers.
    dimensions: int = 512  # Embedding dimension.
    add_degree: bool = True  # Use the node degree features
    add_lcc: bool = True  # Use the local clustering coefficient features
    standardize: bool = True  # Standardize the input attributes.
    use_cpu: bool = False  # Force use CPU.
    lr: float = 1e-3  # Learning rate.
    wd: float = 0.0  # Weight decay.
    name: str = 'gae'
    path: str = 'methods/baselines/run_gae.py'
    env_name: str = 'acc_neb_env'


@dc.dataclass(frozen=True)
class SgcnAlg(EmbeddingAlgSpec):
    graph_support: AlgGraphSupport = AlgGraphSupport(
        directed=True,
        weighted=False,
        node_attributed=True,
    )
    standardize: bool = True  # Standardize in input features
    num_layers: int = 2  # Number of layers.
    use_lcc: bool = True  # Use the local clustering coefficient features
    initialize_with_ones: bool = False  # Use unity features instead of degrees and lcc
    name: str = 'sgcn'
    path: str = 'methods/baselines/run_sgcn.py'
    env_name: str = 'acc_neb_env'
