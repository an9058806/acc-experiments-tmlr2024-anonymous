import copy
import dataclasses as dc
import os.path
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

from graph_alignment import make_alignment_graph
from magna_graph_alignment import load_alignment_graph


def separate_merged_graphs(g: dgraphs.SimpleGraph, align_obj: alignment.AlignedGraphs):
    g1_edges_mask = (g.edges[:, 0] < align_obj.g1_num_nodes) | (g.edges[:, 1] < align_obj.g1_num_nodes)
    g1_edges = g.edges[g1_edges_mask]

    g2_edges = g.edges[~g1_edges_mask]
    g2_edges[:, 0] = align_obj.g2_to_g1.loc[g2_edges[:, 0] - align_obj.g1_num_nodes].to_numpy()
    g2_edges[:, 1] = align_obj.g2_to_g1.loc[g2_edges[:, 1] - align_obj.g1_num_nodes].to_numpy()

    g1 = dgraphs.SimpleGraph(num_nodes=align_obj.g1_num_nodes, edges=g1_edges)
    g2 = dgraphs.SimpleGraph(num_nodes=align_obj.g2_num_nodes, edges=g2_edges)
    return g1, g2


def separate_merged_embeddings(embeddings: np.ndarray, align_obj: alignment.AlignedGraphs):
    g2_to_merged = align_obj.g2_to_merged(align_obj.g1_num_nodes, align_obj.g2_num_nodes)
    clean_embeddings, noisy_graph_embeddings = alignment.split_embeddings(embeddings, g2_to_merged)
    g2_index = align_obj.g1_to_g2[np.arange(align_obj.g2_num_nodes, dtype=int)]
    noisy_graph_embeddings = noisy_graph_embeddings[g2_index, :]
    return clean_embeddings, noisy_graph_embeddings


def get_evaluation(results_path, embeddings, align_obj, noise_p, noise_p_actual, alg, alg_output, pp_mode: str):
    all_results = []
    data = {"noise_p": noise_p, "noise_p_actual": noise_p_actual}
    align_results = None
    if embeddings is not None:
        if np.isfinite(embeddings).all():
            data.update(dc.asdict(alg_output))
            start = time.time()
            align_results = get_align_results_and_save(
                results_path=results_path,
                embeddings=embeddings,
                align_obj=align_obj,
                alg=alg,
                alg_output=alg_output,
                pp_mode=pp_mode,
                noise_p=noise_p
            )
            alignment_duration = time.time() - start
            data["alignment_duration"] = alignment_duration
            for pp_mode_, (k_distances, accuracy) in align_results.items():
                entry = copy.deepcopy(data)
                entry["pp_mode"] = pp_mode_
                data["alignment_duration"] = alignment_duration
                entry["k@1"] = accuracy
                entry["k_distances"] = k_distances
                all_results.append(entry)
            data["kemb"] = embeddings.shape[1]
        else:
            alg_output = dc.replace(alg_output, outcome="nan_embeddings")
            data.update(dc.asdict(alg_output))
            all_results.append(data)
    else:
        print(f"Outcome {alg_output.outcome} for {alg_output.name}.")
        all_results.append(data)
    return all_results, align_results


def get_align_results_and_save(results_path, embeddings: np.ndarray, align_obj, alg, alg_output, pp_mode: str,
                               noise_p: float):
    results_dir = os.path.dirname(results_path)
    folder = os.path.join(results_dir, f"noise_p_{noise_p:.2f}")
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, f"{alg.spec.name}_raw_embeddings.npz"),
             embeddings=embeddings
             )
    pp_embeddings = utils.pp_embeddings_generator(embeddings, pp_modes=pp_mode.split("::"))

    all_results = {}
    for pp_mode, X in pp_embeddings:
        if np.isfinite(X).all():
            g2_to_merged = align_obj.g2_to_merged(align_obj.g1_num_nodes, align_obj.g2_num_nodes)
            try:
                top_sim = alignment.get_top_sim(X, g2_to_merged, alpha=25)
                accuracy_res = alignment.calc_topk_acc_score(align_obj.g2_to_g1, top_sim, topk_vals=np.asarray([1]))
                res = alignment.get_k_per_node(align_obj.g2_to_g1, top_sim)
                all_results[pp_mode] = (res, accuracy_res[1])
            except IndexError:
                print(f"Index error produced during alignment")
                continue
            clean_embeddings, noisy_graph_embeddings = separate_merged_embeddings(X, align_obj)
            folder = os.path.join(results_dir, f"noise_p_{noise_p:.2f}", f"{alg.spec.name}_{pp_mode}")
            os.makedirs(folder, exist_ok=True)
            np.savez(os.path.join(folder, f"embeddings_and_k_per_node.npz"),
                     clean_embeddings=clean_embeddings,
                     noisy_embeddings=noisy_graph_embeddings,
                     k_per_node=res
                     )
    return all_results


def run_eval(dataroot, dataset_spec: DatasetSpec, alg_specs: Sequence[algutils.EmbeddingAlgSpec],
             seed: int, noise_p: float,
             pp_mode: str = 'all', tempdir: str = "./", results_path: str = None, timeout: int = 3600,
             debug: bool = False):
    results_dir = os.path.dirname(results_path)

    all_results = []
    rng = np.random.default_rng(seed)

    if dataset_spec.data_name == 'magna':
        graph, align_obj, noise_p_actual = load_alignment_graph(dataroot=dataroot, noise_level=int(100 * noise_p))
    else:
        data_graph = SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)
        graph, align_obj, noise_p_actual = make_alignment_graph(
            data_graph,
            noise_model="remove",
            p=noise_p,
            rng=rng
        )
    algs = algutils.EmbeddingAlg.specs2algs(alg_specs=alg_specs, graph=graph, gc_mode='alg_compatible')

    alg_filter = common.AlgFilter(max_strikes=2)
    algs_to_run = alg_filter.filter(algs)

    folder = os.path.join(results_dir, f"noise_p_{noise_p:.2f}")
    os.makedirs(folder, exist_ok=True)
    g1, g2 = separate_merged_graphs(graph, align_obj)
    g1.save_npz(os.path.join(folder, "clean_graph.npz"))
    g2.save_npz(os.path.join(folder, "noisy_graph.npz"))

    emb_generator = algutils.generate_embeddings_from_subprocesses(
        graph,
        algs_to_run,
        tempdir=tempdir,
        seed=seed,
        timeout=timeout
    )
    for alg, embeddings, alg_output in tqdm.tqdm(emb_generator, total=len(algs_to_run)):
        alg_filter.update(alg, alg_output)
        eval_results, align_results = get_evaluation(
            results_path=results_path,
            embeddings=embeddings, align_obj=align_obj, noise_p=noise_p,
            noise_p_actual=noise_p_actual, alg=alg,
            alg_output=alg_output, pp_mode=pp_mode
        )
        all_results.extend(eval_results)
    if results_path:
        alg_filter.write(results_path[:-5] + "_failed_algs.json")
        pd.DataFrame(all_results).to_json(results_path, lines=True, orient="records")
    return all_results


def main():
    experiment_name = "ga_node_analysis"
    parser = common.get_common_parser()
    parser.add_argument("--noise-p", type=float, default=0.15,
                        help="Amount of edge noise to use.")
    parser.add_argument("--num-reps", type=int, default=5,
                        help="Num repeats")
    args = parser.parse_args()

    experiment_name = f"{experiment_name}_{args.noise_p}"

    results_path, dataset_spec, args = common.setup_experiment(experiment_name, args)
    algs = embalgsets.get_algs(args.methods, emb_dims=args.dims)

    results = run_eval(dataroot=args.dataroot, dataset_spec=dataset_spec, alg_specs=algs,
                       tempdir=args.tempdir, results_path=results_path, timeout=args.timeout,
                       seed=args.seed, debug=args.debug,
                       noise_p=args.noise_p, pp_mode=args.pp_mode)
    pd.DataFrame(results).to_json(results_path, lines=True, orient="records")


if __name__ == "__main__":
    main()
