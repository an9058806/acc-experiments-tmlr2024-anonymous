from typing import Sequence, Dict, List, Tuple
import argparse
import os
import json

import accnebtools.algs.utils as algutils
from accnebtools.utils import make_result_folder, NEB_DATAROOT
from accnebtools.data.graph import DatasetSpec


class AlgFilter:
    failed_algs: Dict[str, List[List[str]]]
    max_strikes: int

    def __init__(self, max_strikes: int):
        self.failed_algs = dict()
        self.max_strikes = max_strikes

    def update(self, alg: algutils.EmbeddingAlg, alg_output: algutils.EmbAlgOutputs):
        if alg_output.outcome != "completed":
            alg_key = self.alg2key(alg)
            message = [alg_output.outcome] + alg_output.error_out
            if alg_key in self.failed_algs:
                self.failed_algs[alg_key].append(message)
            else:
                self.failed_algs[alg_key] = [message]

    def filter(self, algs: Sequence[algutils.EmbeddingAlg]):
        filtered_algs = []
        for alg in algs:
            alg_key = self.alg2key(alg)
            if alg_key not in self.failed_algs or len(self.failed_algs[alg_key]) <= self.max_strikes:
                filtered_algs.append(alg)
        return filtered_algs

    @staticmethod
    def alg2key(alg: algutils.EmbeddingAlg):
        return f"{alg.alg_hash()} ({alg.spec.name})"

    def reset(self):
        self.failed_algs = dict()

    def write(self, path):
        with open(path, 'w') as fp:
            json.dump(self.failed_algs, fp, indent=2)


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, help="Where the data is stored.", default=NEB_DATAROOT)
    parser.add_argument("--dataset", type=str, help="Which dataset to use.", default="")
    parser.add_argument("--methods", nargs="+", help="Which sets of models to run.", default=["defaults"])
    parser.add_argument("--weighted", type=int, default=0, help="Use weights")
    parser.add_argument("--undirected", type=int, default=0, help="Remove directions.")
    parser.add_argument("--node-attributed", type=int, default=0, help="Use node attributes if available.")
    parser.add_argument("--dims", type=int, default=None, help="Embedding dimensionality.")
    parser.add_argument("--pp-mode", type=str, default="all", help="Which preprocessing to use.")

    parser.add_argument("--resultdir", type=str, default="./local_results",
                        help="Path to file where results will be saved.")
    parser.add_argument("--tempdir", type=str, default="/tmp",
                        help="Path to directory where temporary files will be saved.")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="When to timeout embedding methods.")
    parser.add_argument("--seed", type=int, default=1535523,
                        help="The random seed to use.")
    parser.add_argument("--debug", action="store_true", help="Use debug settings.")
    return parser


def setup_experiment(experiment_name, args) -> Tuple[str, DatasetSpec, argparse.Namespace]:
    if args.debug:
        args.resultdir = "./debug"
        args.tempdir = "./debug"
        # args.methods = ["debug"]

    os.makedirs(args.resultdir, exist_ok=True)
    os.makedirs(args.tempdir, exist_ok=True)

    dataset_spec = DatasetSpec(data_name=args.dataset,
                               force_undirected=bool(args.undirected),
                               force_unweighted=not bool(args.weighted),
                               rm_node_attributes=not bool(args.node_attributed),
                               with_self_loops=False
                               )

    methods_str = "_".join(sorted(args.methods))
    file_name = (
            "".join(sorted(args.methods)) +
            f"{'_undir' if args.undirected else ''}" +
            f"{'_weighted' if args.weighted else ''}" +
            f"{'_node_attributed' if hasattr(args, 'node_attributed') and args.node_attributed else ''}" +
            ".json"
    )
    results_path = make_result_folder(args.resultdir, experiment_name, args.dataset, methods_str, file_name)
    with open(os.path.join(os.path.dirname(results_path), "run_inputs.json"), 'w') as fp:
        json.dump(vars(args), fp, indent=2)

    return results_path, dataset_spec, args
