import argparse
import os
from glob import glob
import json
import yaml
from accnebtools.utils import NEB_DATAROOT
import accnebtools.data.datasets as datasets


def index_and_download_datasets(data_root: str, force_download: bool = False, snap_patents_path: str = None):
    if not data_root:
        data_root = NEB_DATAROOT
        print(f"No data root provided. Using '{data_root}'.")
    os.makedirs(data_root, exist_ok=True)
    available_metadata_paths = glob(os.path.join(NEB_DATAROOT, "**/metadata.yml"), recursive=True)
    print(f"{len(available_metadata_paths)} datasets with metadata located")
    data_index = {}

    for metadata_path in available_metadata_paths:
        try:
            with open(metadata_path, 'r') as fp:
                metadata = yaml.safe_load(fp)
            datasets.download_data_if_required(data_root, metadata, metadata_path, force_download=force_download,
                                               snap_patents_mat_file_path=snap_patents_path)
            # print(metadata)
            datasets.verify_metadata(data_root, metadata, metadata_path)
            if metadata['name'] in data_index:
                raise datasets.MetadataError(f"Name in {metadata_path} already used by "
                                             f"{data_index[metadata['name']]['datapath']} dataset.")
        except FileNotFoundError as e:
            # print(e)
            continue
        except datasets.MetadataError as e:
            print(e)
            continue

        datapath = datasets.get_data_savepath(data_root, metadata_path)
        graphpath = os.path.join(datapath, metadata['graph_file'])
        metadata["datapath"] = datapath
        metadata["graphpath"] = graphpath
        data_index[metadata['name']] = metadata

    with open(os.path.join(data_root, "data_index.json"), 'w') as fp:
        json.dump(data_index, fp, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default=None,
                        help="Where to store the data. Default it store store the data inside the "
                             "'data' folder in this repository.")
    parser.add_argument("--force-download", action="store_true", default=False,
                        help="Force redownload and overwrite existting node classification datasets.")
    parser.add_argument("--snap-patents-path", type=str, default="",
                        help="Provide path to downloaded snap_patents.mat file from here "
                             "https://github.com/CUAI/Non-Homophily-Large-Scale?tab=readme-ov-file#dataset-preparation")
    args = parser.parse_args()
    index_and_download_datasets(data_root=args.dataroot, force_download=args.force_download,
                                snap_patents_path=args.snap_patents_path)
