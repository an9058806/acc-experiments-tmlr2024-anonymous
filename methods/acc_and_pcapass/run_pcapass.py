import os
import time
import json
import random
import numpy as np

import accnebtools.argsfromconfig as parsing
import accnebtools.data.core_ as datacore
import torch
import accmp.pcapass as pcapass
import accmp.transforms as accmptrns
import accmp.preprocessing as preproc

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))


def compute_embeddings(input_file, output_path, as_undirected, weighted, node_attributed, args, metadata_path=None):
    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file, as_canonical_undirected=as_undirected, add_symmetrical_edges=as_undirected, remove_self_loops=True)
    directed = not as_undirected and directed
    if not weighted:
        weights = None
    if not node_attributed or node_attributes is None:
        node_attributes = None
    else:
        node_attributes = np.atleast_2d(node_attributes)

    dtype = np.float32 if args.dtype32 else np.float64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    init_params = preproc.InitFeaturesWeightsParams(
        use_degree=args.use_degree,
        use_log1p_degree=False,
        use_lcc=args.use_lcc,
        use_weights=weights is not None,
        use_node_attributes=node_attributes is not None,
        as_undirected=not directed,
        dtype=dtype
    )
    norm = None if args.normalization == 'none' else args.normalization

    feature_normalization = accmptrns.FeatureNormalization(
        mode=norm,
        subtract_mean=args.subtract_mean,
        before_prune=True,
        before_propagate=False
    )

    instance_normalization = accmptrns.FeatureNormalization(mode=None, subtract_mean=False,
                                                            before_prune=False, before_propagate=False)

    params = pcapass.PCAPassParams(
        max_steps=args.max_steps,
        initial_feature_standardization=accmptrns.FeatureNormalization(mode='std', subtract_mean=True,
                                                                       before_prune=False, before_propagate=False),
        mp_feature_normalization=feature_normalization,
        mp_instance_normalization=instance_normalization,
        init_params=init_params,
        decomposed_layers=1,
        max_dim=args.dimensions,
        return_us=args.return_us,
        use_rsvd=args.use_rsvd,
        sv_thresholding=args.sv_thresholding,
        theta=args.theta,
        normalized_weights=True,
    )

    start = time.perf_counter()
    embeddings = pcapass.pcapass_embeddings(
        edge_index=edges.T, num_nodes=num_nodes, directed_conv=directed, params=params, weights=weights,
        node_attributes=node_attributes, device=device)
    duration = time.perf_counter() - start

    meta_data = vars(args)
    meta_data["duration"] = duration
    meta_data["acp_steps"] = args.max_steps

    np.save(output_path, embeddings, allow_pickle=False)
    if metadata_path is not None:
        with open(metadata_path, 'w') as fp:
            json.dump(meta_data, fp)


def main():
    name = "Directed PCAPass with rank determining."
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", 'pcapass')
    parser.description = f"{name}: Aggregate, concatenate, prune with PCA"
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config = json.load(fp)
    else:
        config = vars(args)

    if config['seed'] is not None:
        np.random.seed(config['seed'])
        random.seed(config['seed'])

    acp_cache_dir = f"/tmp/accmp_cache/cpu_{os.cpu_count()}/"
    os.makedirs(acp_cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = acp_cache_dir

    compute_embeddings(input_file=config['input_file'],
                       output_path=config['output_file'],
                       metadata_path=config['metadata'],
                       as_undirected=config['undirected'],
                       weighted=config['weighted'],
                       node_attributed=config['node_attributed'],
                       args=args
                       )


if __name__ == "__main__":
    main()
