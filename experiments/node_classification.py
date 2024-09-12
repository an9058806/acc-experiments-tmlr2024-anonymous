import dataclasses as dc
from typing import Sequence
import warnings
import copy

import pandas as pd
import tqdm
import numpy as np
from sklearn.exceptions import ConvergenceWarning

import accnebtools.data.graph as dgraphs
import accnebtools.algs.preconfigs as embalgsets
import accnebtools.algs.utils as algutils
import accnebtools.experiments.classification as nodeclassification
import accnebtools.experiments.pt_sgd_log_reg as sgd_classification
import common


def evaluate_node_classification(embeddings: np.ndarray, pp_modes, alg: algutils.EmbeddingAlg, labels: pd.Series,
                                 node_labels_type: str, seed: int, model_name: str, stratified: bool):
    feature2node_ratio = embeddings.shape[1] / embeddings.shape[0]
    n_reps = 3 if embeddings.shape[0] < 1e6 else 1
    if node_labels_type == "multiclass" and model_name == "log_reg_sgd":

        evaluators = [sgd_classification.PTSGDMultiClassEvaluator(random_state=seed, with_train_eval=False,
                                                                  n_repeats=n_reps, n_splits=5, train_ratio=None,
                                                                  stratified=stratified)]
    elif node_labels_type == "multiclass":

        evaluators = [nodeclassification.MultiClassEvaluator(random_state=seed, with_train_eval=False,
                                                             n_repeats=n_reps, n_splits=5, train_ratio=None,
                                                             stratified=stratified)]
    else:
        raise NotImplementedError(f"Evaluation not implemented for '{node_labels_type}'.")

    with warnings.catch_warnings():
        # Ignore convergence warnings caused by identical embedding vectors
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        all_results = []

        model, model_name = nodeclassification.get_default_classification_model(
            node_labels_type,
            feature2node_ratio=feature2node_ratio,
            random_state=seed,
            model_name=model_name
        )

        for evaluator in tqdm.tqdm(evaluators):
            results = nodeclassification.pp_and_cv_evaluate(
                embeddings=embeddings, model=model, pp_modes=pp_modes, evaluator=evaluator,
                alg_name=alg.spec.name, y=labels,
                scores_name=f"{model_name}::{node_labels_type}")
            all_results += results
    return all_results


def get_evaluation(embeddings, node_labels, node_labels_type, alg, alg_output, pp_mode, rep: int,
                   alg_seed: int, classifier: str, stratified: bool):
    all_results = []
    data = {"rep": rep, "alg_seed": alg_seed}
    data.update(dc.asdict(alg_output))
    del data['feature_descriptions']
    clf_models = classifier.split("::")

    if embeddings is not None and not np.isnan(np.sum(embeddings)) and alg_output.outcome == "completed":
        if np.isfinite(embeddings).all():
            data["kemb"] = embeddings.shape[1]
            for i, clf_model_name in enumerate(clf_models):
                eval_results = evaluate_node_classification(embeddings=embeddings, pp_modes=pp_mode.split("::"),
                                                            alg=alg, labels=node_labels,
                                                            node_labels_type=node_labels_type,
                                                            seed=alg_seed, model_name=clf_model_name,
                                                            stratified=stratified)
                for res in eval_results:
                    entry = copy.deepcopy(data)
                    entry.update(dc.asdict(res))
                    all_results.append(entry)

        else:
            alg_output = dc.replace(alg_output, outcome="nan_embeddings")
            data.update(dc.asdict(alg_output))
            all_results.append(data)
    else:
        print(f"Outcome {alg_output.outcome} for {alg_output.name}.")
        all_results.append(data)
    return all_results


def run_eval(dataroot: str, dataset_spec: dgraphs.DatasetSpec, alg_specs: Sequence[algutils.EmbeddingAlgSpec], *,
             seed: int, num_reps: int = 1, pp_mode: str = 'all', tempdir: str = "./", timeout: int = 3600,
             debug: bool = False, classifier: str, stratified: bool):
    all_results = []
    all_fi_results = dict()

    seed_spawners = np.random.SeedSequence(seed)
    seeds = seed_spawners.generate_state(num_reps)
    alg_filter = common.AlgFilter(max_strikes=0)

    data_graph = dgraphs.SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)
    node_labels, node_labels_type = nodeclassification.read_node_labels(dataroot=dataroot,
                                                                        dataset=dataset_spec.data_name)

    algs = algutils.EmbeddingAlg.specs2algs(alg_specs=alg_specs, graph=data_graph, gc_mode="alg_compatible")

    for rep, alg_seed in zip(tqdm.trange(num_reps), seeds):
        algs_to_run = alg_filter.filter(algs)
        emb_generator = algutils.generate_embeddings_from_subprocesses(data_graph, algs_to_run,
                                                                       tempdir=tempdir,
                                                                       seed=alg_seed,
                                                                       timeout=timeout)
        for alg, embeddings, alg_output in tqdm.tqdm(emb_generator, total=len(algs_to_run)):
            alg_filter.update(alg, alg_output)
            eval_results = get_evaluation(embeddings=embeddings, node_labels=node_labels,
                                          node_labels_type=node_labels_type, alg=alg,
                                          alg_output=alg_output, pp_mode=pp_mode,
                                          rep=rep, alg_seed=alg_seed,
                                          classifier=classifier, stratified=stratified)
            all_results.extend(eval_results)

    return all_results, alg_filter, all_fi_results


def main():
    experiment_name = "node_classification"
    parser = common.get_common_parser()
    parser.add_argument("--num-reps", type=int, default=1, help="Number of times to extract embeddings.")
    parser.add_argument("--clf", type=str, default='log_reg', help="Which classifier to use.")
    parser.add_argument("--stratified", action='store_true', help="Use stratified splits.")

    args = parser.parse_args()
    results_path, dataset_spec, args = common.setup_experiment(experiment_name, args)

    alg_specs = embalgsets.get_algs(args.methods, emb_dims=args.dims)
    results, alg_filter, all_fi_results = run_eval(dataroot=args.dataroot,
                                                   dataset_spec=dataset_spec, alg_specs=alg_specs, seed=args.seed,
                                                   tempdir=args.tempdir, timeout=args.timeout, num_reps=args.num_reps,
                                                   pp_mode=args.pp_mode, debug=args.debug,
                                                   classifier=args.clf, stratified=args.stratified)
    pd.DataFrame(results).to_json(results_path, indent=2, orient="records")
    if len(all_fi_results) > 0:
        np.savez(results_path[:-5] + "_fi_results.npz", **all_fi_results)
    alg_filter.write(results_path[:-5] + "_failed_algs.json")


if __name__ == "__main__":
    main()
