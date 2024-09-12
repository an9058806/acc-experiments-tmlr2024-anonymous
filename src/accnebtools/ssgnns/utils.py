import abc
import warnings
import dataclasses as dc
from typing import List, Dict
import time
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
import torch

import accnebtools.data.graph as dgraph
import accnebtools.experiments.classification as nodeclassification
import accnebtools.experiments.alignment as alignment
import accnebtools.experiments.utils as exprutils
import accnebtools.experiments.pt_sgd_log_reg as ssngnnlrcls
import accnebtools.algs.utils as algutils


class SSGNNTrainer(abc.ABC):

    @abc.abstractmethod
    def step(self, step: int):
        raise NotImplementedError

    @abc.abstractmethod
    def get_embeddings(self) -> torch.Tensor:
        raise NotImplementedError

    def get_embeddings_for_graph(self, graph: dgraph.SimpleGraph) -> torch.Tensor:
        raise NotImplementedError


def get_features(graph: dgraph.SimpleGraph, add_degree: bool, add_lcc: bool, standardize: bool):
    features = algutils.get_features(graph=graph, add_degree=add_degree, add_lcc=add_lcc)

    if standardize:
        std = np.std(features, axis=0)
        std[std == 0] = 1
        features = (features - np.mean(features, axis=0)) / std

    return torch.from_numpy(features).to(dtype=torch.float32)


def train_ss_gnn_without_eval(model_trainer: SSGNNTrainer, num_epochs: int, verbose: bool):
    loss_history = []
    for epoch in range(num_epochs):
        loss = model_trainer.step(step=epoch)
        loss_history.append(loss)
        if verbose:
            print(f'Epoch={epoch:03d}, loss={loss:.4f}')
    return model_trainer, loss_history


class TestEvalCallback:
    name: str = "abc"

    def evaluate(self, embeddings: torch.Tensor) -> List[Dict]:
        raise NotImplementedError

    def __call__(self, embeddings: torch.Tensor) -> List[Dict]:
        return self.evaluate(embeddings)


class EvalClassificationCallback(TestEvalCallback):
    name: str = "nc"

    def __init__(self, labels: pd.Series, node_labels_type: str, pp_modes, seed: int,
                 test_ratio: float = 0.2, y_train_test: tuple = None):
        self.labels = labels
        self.pp_modes = pp_modes
        self.seed = seed
        self.y_train = self.y_test = None
        if y_train_test is not None:
            self.y_train = labels[y_train_test[0]]
            self.y_test = labels[y_train_test[1]]
        n_splits = int(1. / test_ratio)

        if node_labels_type == "multiclass":

            self.evaluator = nodeclassification.MultiClassEvaluator(random_state=seed, with_train_eval=False,
                                                                    n_repeats=3, n_splits=n_splits,
                                                                    train_ratio=1. - test_ratio)
        else:
            raise NotImplementedError(f"Evaluation not implemented for '{node_labels_type}'.")

        self.model = self.get_classification_model(node_labels_type, random_state=seed)

    @staticmethod
    def get_classification_model(node_labels_type: str, random_state: int):
        C = 100.0
        if node_labels_type == "multilabel":
            model = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    random_state=random_state,
                    multi_class="ovr"),
                n_jobs=-1)
        else:
            model = HistGradientBoostingClassifier(class_weight='balanced', random_state=random_state)
        return model

    @torch.no_grad()
    def evaluate(self, embeddings: torch.Tensor) -> List[Dict]:
        with warnings.catch_warnings():
            # Ignore convergence warnings caused by identical embedding vectors
            warnings.simplefilter('ignore', category=ConvergenceWarning)

            if self.y_test is None:
                results = nodeclassification.pp_and_cv_evaluate(
                    embeddings=embeddings.cpu().numpy(), model=self.model, pp_modes=self.pp_modes,
                    evaluator=self.evaluator,
                    alg_name="ssgnn", y=self.labels,
                    scores_name="")
            else:
                results = nodeclassification.pp_and_evaluate(
                    embeddings=embeddings.cpu().numpy(), model=self.model, pp_modes=self.pp_modes,
                    evaluator=self.evaluator,
                    alg_name="ssgnn", y_train=self.y_train, y_test=self.y_test, scores_name="")
        results = [dc.asdict(s) for s in results]
        return results


class EvalClassificationPTLRCallback(TestEvalCallback):
    name: str = "nc_ptlr"

    def __init__(self, labels: pd.Series, *, seed: int, lr: float = 0.01, weight_decay: float = 1e-4,
                 max_epoch: int = 300, test_ratio: float = 0.2, n_repeats: int = 3):
        self.labels = labels
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.test_ratio = test_ratio
        self.n_repeats = n_repeats

    def evaluate(self, embeddings: torch.Tensor) -> List[Dict]:
        results = ssngnnlrcls.cv_evaluate(
            embeddings=embeddings.detach(),
            labels=self.labels, lr=self.lr, weight_decay=self.weight_decay,
            max_epoch=self.max_epoch,
            test_ratio=self.test_ratio, seed=self.seed, n_repeats=self.n_repeats,
            mute=True
        )

        results = [dc.asdict(s) for s in results]
        return results


class EvalNetworkAlignmentCallback(TestEvalCallback):
    name: str = "na"

    def __init__(self, true_graph: dgraph.SimpleGraph, pp_modes, seed: int):
        self.pp_modes = pp_modes
        alignment_data = []
        rng = np.random.default_rng(seed)
        for rep in range(3):
            graph2, alignment_objective = alignment.create_permuted(true_graph, rng=rng)
            graph2, actual_p = dgraph.remove_edges_noise(graph2, p=0.15, rng=rng)
            alignment_graph = dgraph.union(true_graph, graph2)
            alignment_data.append((alignment_graph, alignment_objective))
        self.alignment_data = alignment_data

    def get_eval_graphs(self):
        for align_data in self.alignment_data:
            yield align_data

    @torch.no_grad()
    def evaluate_alignment(self, embeddings: torch.Tensor, align_obj) -> List[Dict]:
        results = []
        pp_embeddings = exprutils.pp_embeddings_generator(embeddings.cpu().numpy(),
                                                          pp_modes=self.pp_modes)
        start = time.perf_counter()
        align_results = alignment.eval_topk_sim(pp_embeddings, align_obj)
        duration = time.perf_counter() - start
        for pp_mode_, results in align_results.items():
            entry = dict(pp_mode=pp_mode_, alignment_duration=duration)
            entry.update({f"k@{k}": val for k, val in results.items()})
            results.append(entry)

        return results


def get_evaluation(model_trainer: SSGNNTrainer, eval_cb: TestEvalCallback, epoch: int):
    emb = None
    if isinstance(eval_cb, EvalNetworkAlignmentCallback):
        scores = []
        for graph, align_obj in eval_cb.get_eval_graphs():
            emb = model_trainer.get_embeddings_for_graph(graph)
            scores += eval_cb.evaluate_alignment(embeddings=emb, align_obj=align_obj)
    else:
        emb = model_trainer.get_embeddings()
        scores = eval_cb.evaluate(emb)

    for score in scores:
        score["epoch"] = epoch
        if emb is not None:
            score["emb_dim"] = emb.shape[1]
    return scores


def train_ss_gnn(model_trainer: SSGNNTrainer, num_epochs: int, eval_callbacks: List[TestEvalCallback],
                 eval_every: int = 10, verbose: bool = False, get_slopes_callback=None):
    loss_history = []
    train_times = []
    score_histories = {eval_cb.name: [] for eval_cb in eval_callbacks}
    slopes_history = []

    for epoch in range(num_epochs):
        if epoch % eval_every == 0:
            if verbose:
                print("Evaluating...")
            if get_slopes_callback is not None:
                slopes_history.append({"epoch": epoch, "slopes": get_slopes_callback(model_trainer)})
            for eval_cb in eval_callbacks:
                start = time.perf_counter()
                score_histories[eval_cb.name] += get_evaluation(model_trainer, eval_cb, epoch=epoch)
                duration = time.perf_counter() - start
                if verbose:
                    print(f"Eval {eval_cb.name} done after {duration} sec.")
        start = time.perf_counter()
        loss = model_trainer.step(step=epoch)
        train_times.append(time.perf_counter() - start)
        loss_history.append(loss)

        if verbose:
            print(f'Epoch={epoch:03d}, loss={loss:.4f}')

    final_scores = dict()
    for eval_cb in eval_callbacks:
        final_scores[eval_cb.name] = get_evaluation(model_trainer, eval_cb, epoch=num_epochs)

    # Do final evaluation
    return final_scores, score_histories, loss_history, train_times, slopes_history
