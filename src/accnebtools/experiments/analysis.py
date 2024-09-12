import os
import json

import numpy as np
import pandas as pd

import accnebtools.utils as nebutils
import accnebtools.data.graph as dgraphs
import accnebtools.experiments.classification as nodeclassification


def read_graph_and_labels(dataset_spec: dgraphs.DatasetSpec):
    index_path = os.path.join(nebutils.NEB_DATAROOT, 'data_index.json')
    data_graph = dgraphs.SimpleGraph.from_dataset_spec(dataset_spec=dataset_spec, dataroot=nebutils.NEB_DATAROOT)
    node_labels_type = None

    with open(index_path, 'r') as fp:
        dataset_info = json.load(fp)[dataset_spec.data_name]

    if dataset_spec.data_name == "dgraphfin":
        with open(index_path, 'r') as fp:
            dataset_info = json.load(fp)["dgraphfin"]
        node_labels_file = os.path.join(dataset_info["datapath"], dataset_info["node_labels_file"])
        data = np.load(node_labels_file)
        train_labels = pd.Series(data['node_labels'][data['train_mask']], index=data['train_mask'])
        val_labels = pd.Series(data['node_labels'][data['valid_mask']], index=data['valid_mask'])
        test_labels = pd.Series(data['node_labels'][data['test_mask']], index=data['test_mask'])
        node_labels = pd.concat([train_labels, val_labels, test_labels], axis=0)
        node_labels_type = dataset_info["node_labels"]
    elif "node_labels_file" in dataset_info:
        node_labels, node_labels_type = nodeclassification.read_node_labels(dataroot=nebutils.NEB_DATAROOT,
                                                                            dataset=dataset_spec.data_name)
    else:
        node_labels = None
    return data_graph, node_labels, node_labels_type
