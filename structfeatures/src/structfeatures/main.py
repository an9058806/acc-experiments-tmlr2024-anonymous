import dataclasses as dc

import numpy as np
import structfeatures.features as abf


# @dc.dataclass(frozen=True, kw_only=True)  # This would be nice here, but kw_only requires python 3.10
@dc.dataclass(frozen=True)
class BaseFeatureParams:
    use_weights: bool
    use_node_attributes: bool
    as_undirected: bool
    use_degree: bool
    use_lcc: bool
    use_egonet_edge_counts: bool
    use_legacy_egonet_edge_counts: bool
    dtype: type = np.float32


def prepare_inputs(edge_index: np.ndarray, num_nodes: int, bf_params: BaseFeatureParams,
                   weights: np.ndarray = None, node_attributes: np.ndarray = None):
    edge_index = edge_index.astype(np.int64, copy=False)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T

    if bf_params.use_node_attributes and node_attributes is not None:
        assert num_nodes == node_attributes.shape[0]
        if node_attributes.ndim == 1:
            node_attributes = node_attributes.reshape(-1, 1)
        node_attributes = node_attributes.astype(bf_params.dtype, copy=False)
    else:
        node_attributes = None

    if bf_params.use_weights and weights is not None:
        assert edge_index.shape[1] == weights.shape[0]
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
    else:
        weights = None
    return edge_index, weights, node_attributes


def get_structural_initial_features(edge_index: np.ndarray, num_nodes: int, bf_params: BaseFeatureParams,
                                    weights: np.ndarray = None, node_attributes: np.ndarray = None):
    initial_features = []
    initial_feature_names = []
    if bf_params.use_degree:
        deg_features, feature_names = abf.degree_features(edge_index=edge_index, num_nodes=num_nodes,
                                                          as_undirected=bf_params.as_undirected,
                                                          weights=weights, dtype=bf_params.dtype)
        initial_features.append(deg_features)
        initial_feature_names.extend(feature_names)

    if bf_params.use_lcc:
        lcc_features, feature_names = abf.local_clustering_coefficients_features(
            edge_index=edge_index, num_nodes=num_nodes,
            as_undirected=bf_params.as_undirected,
            weights=weights, dtype=bf_params.dtype
        )
        initial_features.append(lcc_features)
        initial_feature_names.extend(feature_names)

    if bf_params.use_legacy_egonet_edge_counts:
        lcy_egn_features, feature_names = abf.legacy_egonet_edge_features(
            edge_index=edge_index, num_nodes=num_nodes,
            as_undirected=bf_params.as_undirected,
            weights=weights, dtype=bf_params.dtype)
        initial_features.append(lcy_egn_features)
        initial_feature_names.extend(feature_names)

    if node_attributes is not None:
        initial_features.append(node_attributes)
        initial_feature_names.extend([f"na{i}" for i in range(node_attributes.shape[1])])

    initial_features = np.concatenate(initial_features, axis=1)
    return initial_features, initial_feature_names
