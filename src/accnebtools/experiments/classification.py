from typing import Sequence, Union, List, Iterable, Tuple, Optional
import os
import json
import dataclasses as dc
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit, RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer

import accnebtools.experiments.utils as utils

Classifier = Union[LogisticRegression, HistGradientBoostingClassifier]
PP_MODEL = Iterable[Tuple[utils.PP_MODE, Union[Classifier, MultiOutputClassifier]]]


def read_node_labels(dataroot: str, dataset):
    with open(os.path.join(dataroot, 'data_index.json'), 'r') as fp:
        dataset_info = json.load(fp)[dataset]
    if "node_labels" not in dataset_info:
        raise KeyError("Metadata for '{dataset}' does not contain node_labels.")
    node_labels_type = dataset_info["node_labels"]

    node_labels_file = os.path.join(dataset_info["datapath"], dataset_info["node_labels_file"])
    if node_labels_type == "multilabel":
        node_labels = pd.DataFrame(np.load(node_labels_file))
    else:
        node_labels = pd.read_json(node_labels_file, typ='series')
    return node_labels, node_labels_type


@dc.dataclass(frozen=True)
class MulticlassClassificationScores(utils.PerformanceScores):
    accuracy: float
    macro_f1: float
    micro_f1: float
    train_ratio: Optional[float]

    @classmethod
    def create(cls, scores_name, model, labels, preds, train_ratio):
        acc = skmetrics.accuracy_score(y_true=labels, y_pred=preds)
        macro_f1 = skmetrics.f1_score(y_true=labels, y_pred=preds, average='macro')
        micro_f1 = skmetrics.f1_score(y_true=labels, y_pred=preds, average='micro')
        return cls(scores_name=scores_name,
                   model_name=model.__class__.__name__,
                   model_repr=repr(model),
                   accuracy=acc,
                   macro_f1=macro_f1,
                   micro_f1=micro_f1,
                   train_ratio=train_ratio)


class MultiClassEvaluator(utils.Evaluator):
    def evaluate(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                 model: Classifier, scores_name: str) -> List['MulticlassClassificationScores']:
        classification_model = model.fit(X_train, y_train)

        y_pred_test = classification_model.predict(X_test)
        test_scores = MulticlassClassificationScores.create(
            scores_name=scores_name + "::test",
            model=classification_model,
            labels=y_test,
            preds=y_pred_test,
            train_ratio=self.train_ratio
        )
        scores = [test_scores]
        if self.with_train_eval:
            y_pred_train = classification_model.predict(X_train)
            train_scores = MulticlassClassificationScores.create(
                scores_name=scores_name + "::train",
                model=classification_model,
                labels=y_train,
                preds=y_pred_train,
                train_ratio=self.train_ratio
            )
            scores.append(train_scores)
        return scores

    def cv_evaluate(self, X: np.ndarray, y: pd.Series, model: Classifier, scores_name: str
                    ) -> List['MulticlassClassificationScores']:

        y = y.loc[~y.isnull()]

        if self.train_ratio is None:
            if self.stratified:
                rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats,
                                               random_state=self.random_state)
            else:
                rskf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats,
                                     random_state=self.random_state)
        else:
            rskf = ShuffleSplit(n_splits=self.n_splits * self.n_repeats, test_size=1 - self.train_ratio,
                                random_state=self.random_state)
        all_scores = []

        for train_index, test_index in rskf.split(np.empty_like(y), y):
            node_train_index = y.index[train_index]
            node_test_index = y.index[test_index]
            X_train, X_test = X[node_train_index, :], X[node_test_index, :]
            y_train, y_test = y.loc[node_train_index], y.loc[node_test_index]

            all_scores += self.evaluate(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model=model,
                                        scores_name=scores_name)
        return all_scores


def get_default_classification_model(node_labels_type: str, model_name: str, feature2node_ratio: float,
                                     random_state: int):
    if node_labels_type not in {"multiclass"}:
        raise NotImplementedError(f"Default model not implemented for {node_labels_type}.")

    if model_name == "grad_boost":
        model = HistGradientBoostingClassifier(class_weight='balanced', random_state=random_state)
    elif model_name == "log_reg":
        model = LogisticRegression(class_weight='balanced', random_state=random_state)
    elif model_name == "log_reg_sgd" and node_labels_type == "multiclass":
        # Model is created internally in the Evaluator
        model = FunctionTransformer()
    elif model_name == "linear":
        if feature2node_ratio > 1:
            model = LinearSVC(class_weight='balanced', random_state=random_state, dual='auto')
        else:
            model = LogisticRegression(class_weight='balanced', random_state=random_state)
    else:
        raise NotImplementedError(f"Model name {model_name} unknown")

    return model, model_name


ClassificationEval = Union[MultiClassEvaluator]


def pp_and_cv_evaluate(embeddings: np.ndarray,
                       model: Union[Classifier, MultiOutputClassifier],
                       pp_modes: Sequence[str],
                       evaluator: utils.Evaluator,
                       alg_name: str,
                       y: Union[pd.Series, pd.DataFrame],
                       scores_name: str):
    scores = []
    pp_model_generator = pp_pipeline_generator(model=model, pp_modes=pp_modes, )

    for pp_mode, model_pipeline in pp_model_generator:
        scores += evaluator.cv_evaluate(
            X=embeddings, y=y, model=model_pipeline, scores_name=scores_name + f"::{pp_mode}"
        )

    return scores


def pp_pipeline_generator(model: Union[Classifier, MultiOutputClassifier], pp_modes: Sequence[str]):
    assert not isinstance(pp_modes, str)

    remaped_pp_modes = []
    for pp_mode in pp_modes:
        if pp_mode == "all":
            remaped_pp_modes += ['none', 'standardize', 'whiten']
        else:
            remaped_pp_modes.append(pp_mode)
    pp_modes = sorted(list(set(remaped_pp_modes)))

    for pp_mode in pp_modes:
        pipeline = get_pp_pipeline(model, pp_mode)
        yield pp_mode, pipeline


def get_pp_pipeline(model: Union[Classifier, MultiOutputClassifier], pp_mode: str):
    if pp_mode is None or pp_mode == 'none':
        pipeline = Pipeline([('model', model)])
    elif pp_mode == 'standardize':
        pipeline = Pipeline([('preprocess', StandardScaler()), ('model', model)])
    elif pp_mode == 'whiten':
        pipeline = Pipeline([('preprocess', PCA(whiten=True)), ('model', model)])
    else:
        raise ValueError(f"Unknown pp_mode '{pp_mode}'")
    return pipeline


def pp_and_evaluate(embeddings: np.ndarray,
                    model: Union[Classifier, MultiOutputClassifier],
                    pp_modes: Sequence[str],
                    evaluator: utils.Evaluator,
                    alg_name: str,
                    y_train: pd.Series, y_test: pd.Series,
                    scores_name: str
                    ):
    scores = []

    pp_model_generator = pp_pipeline_generator(model=model, pp_modes=pp_modes)
    X_train, X_test = embeddings[y_train.index, :], embeddings[y_test.index, :]
    for pp_mode, model_pipeline in pp_model_generator:
        scores += evaluator.evaluate(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model=model_pipeline,
            scores_name=scores_name + f"::{pp_mode}"
        )

    return scores
