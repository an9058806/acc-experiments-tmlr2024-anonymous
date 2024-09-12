import dataclasses as dc
from typing import Sequence
import copy
import pickle

import pandas as pd
import tqdm
import numpy as np

import accnebtools.data.graph as dgraphs
import accnebtools.algs.preconfigs as embalgsets
import accnebtools.algs.utils as algutils
import common


def run_eval(dataroot: str, dataset_spec: dgraphs.DatasetSpec, alg_specs: Sequence[algutils.EmbeddingAlgSpec], *,
             seed: int, num_reps: int = 1, tempdir: str = "./", timeout: int = 3600, debug: bool = False, ):
    all_results = []
    singular_values = []
    seed_spawners = np.random.SeedSequence(seed)
    seeds = seed_spawners.generate_state(num_reps)
    alg_filter = common.AlgFilter(max_strikes=0)

    data_graph = dgraphs.SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)

    algs = algutils.EmbeddingAlg.specs2algs(alg_specs=alg_specs, graph=data_graph, gc_mode="alg_compatible")

    for rep, alg_seed in zip(tqdm.trange(num_reps), seeds):
        algs_to_run = alg_filter.filter(algs)
        emb_generator = algutils.generate_embeddings_from_subprocesses(data_graph, algs_to_run,
                                                                       tempdir=tempdir,
                                                                       seed=alg_seed,
                                                                       timeout=timeout)
        for alg, embeddings, alg_output in tqdm.tqdm(emb_generator, total=len(algs_to_run)):
            alg_filter.update(alg, alg_output)

            data = {"rep": rep, "alg_seed": alg_seed, "agg_steps": getattr(alg, 'max_steps', None)}
            data.update(dc.asdict(alg_output))
            del data['feature_descriptions']

            if embeddings is not None and not np.isnan(np.sum(embeddings)) and alg_output.outcome == "completed":
                if np.isfinite(embeddings).all():
                    data["kemb"] = embeddings.shape[1]
                    svs = np.linalg.svd(embeddings, compute_uv=False, hermitian=False)
                    sv_entry = copy.deepcopy(data)
                    sv_entry['singular_values'] = svs
                    singular_values.append(sv_entry)
                else:
                    alg_output = dc.replace(alg_output, outcome="nan_embeddings")
                    data.update(dc.asdict(alg_output))
                    all_results.append(data)
            else:
                print(f"Outcome {alg_output.outcome} for {alg_output.name}.")
                all_results.append(data)

    return all_results, singular_values, alg_filter


def main():
    experiment_name = "embedding_info_analysis"
    parser = common.get_common_parser()
    parser.add_argument("--num-reps", type=int, default=1, help="Number of times to extract embeddings.")

    args = parser.parse_args()
    results_path, dataset_spec, args = common.setup_experiment(experiment_name, args)

    alg_specs = embalgsets.get_algs(args.methods, emb_dims=args.dims)
    results, singular_values, alg_filter = run_eval(dataroot=args.dataroot,
                                                    dataset_spec=dataset_spec, alg_specs=alg_specs, seed=args.seed,
                                                    tempdir=args.tempdir, timeout=args.timeout, num_reps=args.num_reps,
                                                    debug=args.debug)
    pd.DataFrame(results).to_json(results_path, indent=2, orient="records")
    with open(results_path[:-5] + "_singular_values.pkl", 'wb') as fp:
        pickle.dump(singular_values, fp)
    alg_filter.write(results_path[:-5] + "_failed_algs.json")


if __name__ == "__main__":
    main()
