import os
from typing import Dict
import shutil
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np

import scipy
import accnebtools.data.graph as dgraph
import torch_geometric as pyg
from ogb.nodeproppred import NodePropPredDataset
from accnebtools.utils import NEB_DATAROOT

DEFAULT_COMMENT_CHAR = "%"


class MetadataError(ValueError):
    pass


# @dc.dataclass(frozen=True)
# class DatasetMetadata:
#     name: str
#     graphpath: str
#     datapath: str
#     file_info: Dict
#     graph_info: Dict


def get_data_savepath(data_root: str, metadata_path: str):
    relpath_to_metadata_inrepo = os.path.dirname(os.path.abspath(metadata_path))[len(NEB_DATAROOT) + 1:]
    path_to_data_save_location = os.path.abspath(os.path.join(data_root, relpath_to_metadata_inrepo))
    return path_to_data_save_location


def graph_file_exists(data_root: str, metadata: Dict, metadata_path: str):
    datapath = get_data_savepath(data_root, metadata_path)
    graphpath = os.path.join(datapath, metadata['graph_file'])
    return os.path.exists(graphpath)


def copy_inrepo_data(data_root, metadata_path):
    datafolder = get_data_savepath(data_root, metadata_path)
    try:
        shutil.copytree(os.path.dirname(metadata_path), datafolder, dirs_exist_ok=False)
    except FileExistsError as e:
        shutil.rmtree(datafolder)
        shutil.copytree(os.path.dirname(metadata_path), datafolder, dirs_exist_ok=False)


def download_data_if_required(data_root, metadata, metadata_path, force_download: bool,
                              snap_patents_mat_file_path=None):
    graph_exists_dst = graph_file_exists(data_root, metadata, metadata_path)
    graph_exists_neb_root = graph_file_exists(NEB_DATAROOT, metadata, metadata_path)
    if graph_exists_dst and not force_download:
        return
    elif graph_exists_neb_root and ('inrepo' in metadata_path or not force_download):
        copy_inrepo_data(data_root, metadata_path)
    elif "incloud" in metadata_path:
        download_and_process_dataset(data_root, metadata, metadata_path,
                                     snap_patents_mat_file_path=snap_patents_mat_file_path)
    else:
        print(f"No existing graph or download possibility for '{metadata['name']}'")


def download_and_process_dataset(data_root, metadata, metadata_path, snap_patents_mat_file_path=None):
    if metadata['name'] in {'chameleon', 'squirrel'}:
        download_chameleon_and_squirrel(data_root, metadata, metadata_path)
    elif metadata['name'] == 'roman_empire':
        download_roman_empire(data_root, metadata, metadata_path)
    elif metadata['name'] == 'arxiv_year':
        download_and_process_arxiv_year(data_root, metadata, metadata_path)
    elif metadata['name'] == 'snap_patents' and snap_patents_mat_file_path:
        process_snap_patents(snap_patents_mat_file_path, data_root, metadata, metadata_path)
    elif metadata['name'] == 'snap_patents':
        print("No Snap parents file path provided, skipping.")
    else:
        raise NotImplementedError(f"No download implementation for dataset '{metadata['name']}'.")


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def download_and_process_arxiv_year(data_root, metadata, metadata_path):
    dataset = NodePropPredDataset(root="/tmp/tmp_ogb", name='ogbn-arxiv')
    year_label = even_quantile_labels(dataset.graph['node_year'].flatten(), 5, verbose=True)
    graph = dgraph.SimpleGraph(
        num_nodes=dataset.graph['node_feat'].shape[0],
        edges=dataset.graph['edge_index'].T,
        node_attributes=dataset.graph['node_feat'],
        directed=True
    )

    path = get_data_savepath(data_root=data_root, metadata_path=metadata_path)
    os.makedirs(path, exist_ok=True)
    graph.save_npz(os.path.join(path, metadata['graph_file']))
    pd.Series(year_label).to_json(os.path.join(path, metadata['node_labels_file']))


def process_snap_patents(path_to_mat_file, data_root, metadata, metadata_path):
    data = scipy.io.loadmat(path_to_mat_file)
    labels = even_quantile_labels(data['years'].flatten(), 5)
    x = data['node_feat'].toarray().astype(np.float32)

    graph = dgraph.SimpleGraph(
        num_nodes=x.shape[0],
        edges=data['edge_index'].T.astype(np.int64),
        node_attributes=x,
        directed=True,
    )

    path = get_data_savepath(data_root=data_root, metadata_path=metadata_path)
    os.makedirs(path, exist_ok=True)
    graph.save_npz(os.path.join(path, metadata['graph_file']))
    pd.Series(labels).to_json(os.path.join(path, metadata['node_labels_file']))


def pyg_dataset_to_graph(dataset):
    graph = dgraph.SimpleGraph(
        num_nodes=dataset.x.shape[0],
        edges=dataset.edge_index.T.numpy().astype(np.int64),
        node_attributes=dataset.x.numpy(),
        directed=dataset[0].is_directed()
    )
    return graph


def download_roman_empire(data_root, metadata, metadata_path):
    url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/roman_empire.npz"
    folder = "/tmp/tmp_pyg_data_hetero"
    os.makedirs(folder, exist_ok=True)
    tmp_path = os.path.join(folder, "roman_empire.npz")
    r = requests.get(url, stream=True)
    total_length = r.headers.get('content-length')
    total_length = int(total_length) // 4096 if total_length is not None else total_length
    with open(tmp_path, 'wb') as f:
        if total_length is None:
            f.write(r.content)
        else:
            for chunk in tqdm(r.iter_content(chunk_size=4096), total=total_length):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    data = np.load(tmp_path)
    graph = dgraph.SimpleGraph(
        num_nodes=data['node_features'].shape[0],
        edges=data['edges'].astype(np.int64),
        node_attributes=data['node_features'],
        directed=True,
    )

    path = get_data_savepath(data_root=data_root, metadata_path=metadata_path)
    os.makedirs(path, exist_ok=True)
    graph.save_npz(os.path.join(path, metadata['graph_file']))
    pd.Series(data['node_labels']).to_json(os.path.join(path, metadata['node_labels_file']))


def download_chameleon_and_squirrel(data_root, metadata, metadata_path):
    dataset = pyg.datasets.WikipediaNetwork(root="/tmp/tmp_pyg_data_hetero", name=metadata['name'])
    graph = pyg_dataset_to_graph(dataset)
    path = get_data_savepath(data_root=data_root, metadata_path=metadata_path)
    os.makedirs(path, exist_ok=True)
    graph.save_npz(os.path.join(path, metadata['graph_file']))
    pd.Series(dataset.y).to_json(os.path.join(path, metadata['node_labels_file']))


def verify_graph(data_root: str, metadata: Dict, metadata_path: str):
    datapath = get_data_savepath(data_root, metadata_path)
    graphpath = os.path.join(datapath, metadata['graph_file'])
    if not os.path.exists(graphpath):
        raise FileNotFoundError(f"Could not find graph data {graphpath} for {metadata_path}")


def verify_filetype_info(metadata: Dict, metadata_path: str):
    if 'filetype' not in metadata['file_info']:
        raise MetadataError(f"Filetype not avaiable in {metadata_path}")


def verify_graph_info(metadata: Dict, metadata_path: str):
    if 'num_nodes' not in metadata['graph_info']:
        raise MetadataError(f"'num_nodes' not in {metadata_path}")
    if 'weights' not in metadata['graph_info']:
        raise MetadataError(f"'weights' info not in {metadata_path}")
    if 'directed' not in metadata['graph_info']:
        raise MetadataError(f"'directed' info not in {metadata_path}")


def verify_metadata(data_root: str, metadata: Dict, methadata_path: str):
    if 'name' not in metadata:
        raise MetadataError(f"'name' not in {methadata_path}")
    verify_graph(data_root, metadata, methadata_path)
    verify_filetype_info(metadata, methadata_path)
    verify_graph_info(metadata, methadata_path)
