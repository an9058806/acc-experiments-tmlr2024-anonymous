import copy
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


def make_alignment_graph(graph: SimpleGraph, noise_model: str, p: float, rng: np.random.Generator):
    graph2, alignment_objective = alignment.create_permuted(graph, rng=rng)
    if noise_model.lower() == "add":
        graph2, actual_p = dgraphs.add_noise_edges(graph2, p=p, rng=rng)
    elif noise_model.lower() == "remove":
        graph2, actual_p = dgraphs.remove_edges_noise(graph2, p=p, rng=rng)
    else:
        raise NotImplementedError(f"Noise model {noise_model} is not found.")

    merged_graph = dgraphs.union(graph, graph2)
    return merged_graph, alignment_objective, actual_p


def get_evaluation(embeddings, align_obj, noise_p, noise_p_actual, alg_name, alg_output, pp_modes: str, rep: int,
                   alg_seed: int):
    all_results = []
    data = {"noise_p": noise_p, "noise_p_actual": noise_p_actual, "rep": rep, "alg_seed": alg_seed}
    if embeddings is not None and alg_output.outcome == "completed":
        if np.isfinite(embeddings).all():
            data.update(dc.asdict(alg_output))
            del data['feature_descriptions']
            pp_embeddings = utils.pp_embeddings_generator(embeddings, pp_modes=pp_modes.split("::"))
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
             seed: int, noise_levels: Sequence[float],
             num_reps: int = 5, pp_mode: str = 'all',
             tempdir: str = "./", results_path: str = None,
             timeout: int = 3600,
             debug: bool = False):
    if debug:
        num_reps = 1
        noise_levels = [0.01]

    all_results = []
    rng = np.random.default_rng(seed)
    seed_spawners = np.random.SeedSequence(seed).spawn(len(noise_levels))

    data_graph = SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)
    algs = algutils.EmbeddingAlg.specs2algs(alg_specs=alg_specs, graph=data_graph, gc_mode='alg_compatible')

    alg_filter = common.AlgFilter(max_strikes=0)
    for noise_p, ss in zip(tqdm.tqdm(noise_levels), seed_spawners):
        spawned_seeds = ss.generate_state(num_reps)
        for rep, algs_seed in zip(tqdm.trange(num_reps), spawned_seeds):
            algs_to_run = alg_filter.filter(algs)
            graph, align_obj, noise_p_actual = make_alignment_graph(
                data_graph,
                noise_model="remove",
                p=noise_p,
                rng=rng
            )
            emb_generator = algutils.generate_embeddings_from_subprocesses(
                graph,
                algs_to_run,
                tempdir=tempdir,
                seed=algs_seed,
                timeout=timeout
            )
            for alg, embeddings, alg_output in tqdm.tqdm(emb_generator, total=len(algs_to_run)):
                alg_filter.update(alg, alg_output)
                eval_results = get_evaluation(embeddings=embeddings,
                                              align_obj=align_obj, noise_p=noise_p, noise_p_actual=noise_p_actual,
                                              alg_name=alg.spec.name, alg_output=alg_output, pp_modes=pp_mode,
                                              rep=rep, alg_seed=algs_seed)
                all_results.extend(eval_results)
        if results_path:
            alg_filter.write(results_path[:-5] + "_failed_algs.json")
            pd.DataFrame(all_results).to_json(results_path, indent=2, orient="records")
    return all_results


def main():
    experiment_name = "graph_alignment"
    parser = common.get_common_parser()
    parser.add_argument("--noisemodel", type=str, help="Which method to add noise to the graphs",
                        default="remove")
    parser.add_argument("--noise-p", type=str, default="full",
                        help="Amount of edge noise to use.")
    parser.add_argument("--num-reps", type=int, default=5,
                        help="Num repeats")
    args = parser.parse_args()

    if args.noise_p == "low":
        noise_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    elif args.noise_p == "high":
        noise_levels = [0.075, 0.1, 0.15, 0.2, 0.25]
    elif args.noise_p == "full":
        noise_levels = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
    elif args.noise_p == "selected":
        noise_levels = [0.05, 0.1, 0.2]
    else:
        noise_levels = [float(args.noise_p)]

    experiment_name = f"{experiment_name}_{args.noise_p}"

    results_path, dataset_spec, args = common.setup_experiment(experiment_name, args)
    algs = embalgsets.get_algs(args.methods, emb_dims=args.dims)

    results = run_eval(dataroot=args.dataroot,
                       dataset_spec=dataset_spec, alg_specs=algs,
                       tempdir=args.tempdir, results_path=results_path, timeout=args.timeout,
                       seed=args.seed, debug=args.debug,
                       noise_levels=noise_levels, num_reps=args.num_reps, pp_mode=args.pp_mode)
    pd.DataFrame(results).to_json(results_path, indent=2, orient="records")


if __name__ == "__main__":
    main()
