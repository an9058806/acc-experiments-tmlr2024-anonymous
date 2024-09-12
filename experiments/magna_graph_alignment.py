import copy
import os
import json
import dataclasses as dc
from typing import Sequence
import time

import pandas as pd
import tqdm.auto as tqdm
import numpy as np

import accnebtools.data.graph as dgraphs
from accnebtools.data.graph import SimpleGraph, DatasetSpec
import accnebtools.algs.preconfigs as embalgsets
import accnebtools.algs.utils as algutils
import accnebtools.experiments.alignment as alignment
import accnebtools.experiments.utils as utils
import common


def load_alignment_graph(dataroot: str, noise_level: int):
    index_path = os.path.join(dataroot, "data_index.json")
    with open(index_path, 'r') as fp:
        dataset_info = json.load(fp)["magna"]
    g1_graph = SimpleGraph.from_dataset_index(dataroot=dataroot, data_name='magna', as_undirected=True,
                                              as_unweighted=True)
    g2_path = os.path.join(dataset_info['datapath'], dataset_info['alignment_graphs_dir'], f'graph+{noise_level}e.npz')
    g2_graph = SimpleGraph.load(g2_path, as_canonical_undirected=True, add_symmetrical_edges=True, use_weights=False)
    merged_graph = dgraphs.union(g1_graph, g2_graph)
    align_obj_path = os.path.join(dataset_info['datapath'], dataset_info['alignment_graphs_dir'],
                                  f'graph+{noise_level}e_align_obj.csv')
    alignment_objective = alignment.AlignedGraphs.load_from_file(align_obj_path)
    noise_p = float(noise_level) / 100.
    return merged_graph, alignment_objective, noise_p


def get_evaluation(embeddings, align_obj, noise_p, alg, alg_output, pp_mode: str, alg_seed: int):
    all_results = []
    data = {"noise_p": noise_p, "alg_seed": alg_seed}
    if embeddings is not None:
        if np.isfinite(embeddings).all():
            data.update(dc.asdict(alg_output))
            pp_embeddings = utils.pp_embeddings_generator(embeddings, pp_modes=pp_mode.split("::"))
            start = time.time()
            align_results = alignment.eval_topk_sim(pp_embeddings, align_obj)
            alignment_duration = time.time() - start
            data["alignment_duration"] = alignment_duration
            for pp_mode_, results in align_results.items():
                entry = copy.deepcopy(data)
                entry["pp_mode"] = pp_mode_
                data["alignment_duration"] = alignment_duration
                entry.update({f"k@{k}": val for k, val in results.items()})
                all_results.append(entry)
            data["kemb"] = embeddings.shape[1]
        else:
            alg_output = dc.replace(alg_output, outcome="nan_embeddings")
            data.update(dc.asdict(alg_output))
            all_results.append(data)
    else:
        print(f"Outcome {alg_output.outcome} for {alg_output.name}.")
        all_results.append(data)
    return all_results


def run_eval(dataroot: str, dataset_spec: DatasetSpec, alg_specs: Sequence[algutils.EmbeddingAlgSpec],
             seed: int, noise_levels: Sequence[int],
             pp_mode: str = 'all',
             tempdir: str = "./", results_path: str = None,
             timeout: int = 3600,
             debug: bool = False):
    if debug:
        noise_levels = [5]

    all_results = []

    data_graph = SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)
    algs = algutils.EmbeddingAlg.specs2algs(alg_specs=alg_specs, graph=data_graph, gc_mode='alg_compatible')

    alg_filter = common.AlgFilter(max_strikes=0)
    for noise_level in tqdm.tqdm(noise_levels):

        algs_to_run = alg_filter.filter(algs)
        graph, align_obj, noise_p = load_alignment_graph(
            dataroot=dataroot,
            noise_level=noise_level
        )
        emb_generator = algutils.generate_embeddings_from_subprocesses(
            graph,
            algs_to_run,
            tempdir=tempdir,
            seed=seed,
            timeout=timeout
        )
        for alg, embeddings, alg_output in tqdm.tqdm(emb_generator, total=len(algs_to_run)):
            alg_filter.update(alg, alg_output)
            eval_results = get_evaluation(embeddings=embeddings,
                                          align_obj=align_obj, noise_p=noise_p,
                                          alg=alg, alg_output=alg_output,
                                          pp_mode=pp_mode, alg_seed=seed)
            all_results.extend(eval_results)
        if results_path:
            alg_filter.write(results_path[:-5] + "_failed_algs.json")
            pd.DataFrame(all_results).to_json(results_path, indent=2, orient="records")
    return all_results


def main():
    experiment_name = "magna_graph_alignment"
    parser = common.get_common_parser()
    parser.add_argument("--noise-level", type=str, default="full", help="Amount of edge noise to use.")
    args = parser.parse_args()

    if args.noise_level == "full":
        noise_levels = [5, 10, 15, 20, 25]
    else:
        noise_levels = [int(args.noise_level)]

    experiment_name = f"{experiment_name}_{args.noise_level}"
    args.dataset = 'magna'

    results_path, _, args = common.setup_experiment(experiment_name, args)
    dataset_spec = DatasetSpec(data_name=args.dataset,
                               force_undirected=True,
                               force_unweighted=True,
                               rm_node_attributes=True,
                               with_self_loops=False
                               )
    algs = embalgsets.get_algs(args.methods, emb_dims=args.dims)

    results = run_eval(dataroot=args.dataroot,
                       dataset_spec=dataset_spec, alg_specs=algs, tempdir=args.tempdir,
                       results_path=results_path, timeout=args.timeout,
                       seed=args.seed, debug=args.debug,
                       noise_levels=noise_levels, pp_mode=args.pp_mode)
    pd.DataFrame(results).to_json(results_path, indent=2, orient="records")


if __name__ == "__main__":
    main()
