import copy
from typing import List, Optional, Union

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit

import torch
import torch.nn as nn

import accnebtools.experiments.classification as nodeclassification
from accnebtools.experiments.classification import MulticlassClassificationScores, Pipeline
import accnebtools.experiments.utils as utils

TArray = Union[torch.Tensor, np.ndarray]


def to_tensor(array: TArray):
    if not isinstance(array, torch.Tensor):
        array = torch.from_numpy(array)
    if torch.cuda.is_available():
        array = array.to(torch.device('cuda'))
    return array


class PTSGDMultiClassEvaluator(utils.Evaluator):

    def __init__(self, *, lr: float = 0.01, wd: float = 1e-4, max_epoch: int = 300,
                 random_state: int, with_train_eval: bool, train_ratio: Optional[float],
                 n_splits: int = 5, n_repeats: int = 3, stratified: bool = True):
        super().__init__(random_state=random_state, with_train_eval=with_train_eval, n_splits=n_splits,
                         n_repeats=n_repeats, train_ratio=train_ratio, stratified=stratified)
        self.lr = lr
        self.wd = wd
        self.max_epoch = max_epoch

    def evaluate(self, X_train: TArray, X_test: TArray, y_train: TArray, y_test: TArray, pre_processing: Pipeline,
                 scores_name: str) -> List[MulticlassClassificationScores]:
        X_train = pre_processing.fit_transform(X_train)
        X_test = pre_processing.transform(X_test)

        X_train = to_tensor(X_train)
        X_test = to_tensor(X_test)
        y_train = to_tensor(y_train)
        y_test = to_tensor(y_test)
        lr_model = get_log_reg_model(X_train, labels=y_train, lr=self.lr,
                                     weight_decay=self.wd, max_epoch=self.max_epoch, mute=True)
        with torch.no_grad():
            lr_model.eval()
            pred = lr_model(X_test)
            y_true = y_test.squeeze().long()
            preds = pred.max(1)[1].type_as(y_true)
            test_scores = nodeclassification.MulticlassClassificationScores.create(
                scores_name=scores_name + "::test",
                model=lr_model,
                labels=y_true.cpu().numpy(),
                preds=preds.cpu().numpy(),
                train_ratio=self.train_ratio
            )
            scores = [test_scores]

            if self.with_train_eval:
                lr_model.eval()
                pred = lr_model(X_train)
                y_true = y_train.squeeze().long()
                preds = pred.max(1)[1].type_as(y_true)
                train_scores = nodeclassification.MulticlassClassificationScores.create(
                    scores_name=scores_name + "::train",
                    model=lr_model,
                    labels=y_true.cpu().numpy(),
                    preds=preds.cpu().numpy(),
                    train_ratio=self.train_ratio
                )
                scores.append(train_scores)

        return scores

    def cv_evaluate(self, X: np.ndarray, y: pd.Series, model: Pipeline, scores_name: str
                    ) -> List['MulticlassClassificationScores']:

        y = y.loc[~y.isnull()]
        labels, unique_labels = pd.factorize(y)
        labels = torch.from_numpy(labels).to(torch.long)

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

        for train_index, test_index in rskf.split(np.empty_like(labels), labels):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = labels[train_index], labels[test_index]
            all_scores += self.evaluate(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                        pre_processing=model, scores_name=scores_name)

        return all_scores


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        logits = self.linear(x)
        return logits


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def cv_evaluate(embeddings: torch.Tensor, labels: pd.Series, *, lr: float, weight_decay: float,
                max_epoch: int, test_ratio: float, seed: int, n_repeats: int = 3, mute=False):
    all_scores = []
    labels, unique_labels = pd.factorize(labels)
    labels = torch.from_numpy(labels).to(torch.long)
    n_splits = int(1. / test_ratio)

    rskf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for train_index, test_index in rskf.split(np.empty_like(labels), labels):
        X_train, X_test = embeddings[train_index, :], embeddings[test_index, :]
        y_train, y_test = labels[train_index], labels[test_index]
        lr_model = get_log_reg_model(X_train, labels=y_train.to(embeddings.device), lr=lr, weight_decay=weight_decay,
                                     max_epoch=max_epoch, mute=mute)
        with torch.no_grad():
            lr_model.eval()
            pred = lr_model(X_test)
            y_true = y_test.squeeze().long()
            preds = pred.max(1)[1].type_as(y_true)
            scores = nodeclassification.MulticlassClassificationScores.create(
                scores_name="lr_gpu",
                model=lr_model,
                labels=y_true.cpu().numpy(),
                preds=preds.cpu().numpy(),
                train_ratio=0.9 * (n_splits - 1) / float(n_splits)
            )
            all_scores.append(scores)
        # with torch.no_grad():
        #     test_acc = eval_forward(test_loader, test_label)

    return all_scores


def get_log_reg_model(embeddings: torch.Tensor, labels: torch.Tensor,
                      lr, weight_decay, max_epoch, mute=False):
    criterion = torch.nn.CrossEntropyLoss()
    train_emb, val_emb, train_label, val_label = train_test_split(embeddings, labels, test_size=0.1)

    best_val_acc = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    encoder = LogisticRegression(train_emb.shape[1], int(train_label.max().item() + 1))
    encoder = encoder.to(embeddings.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in epoch_iter:
        encoder.train()
        optimizer.zero_grad()
        pred = encoder(train_emb)
        loss = criterion(pred, train_label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            encoder.eval()
            pred = encoder(val_emb)
            val_acc = accuracy(pred, val_label)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(encoder)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc:.4f}")

    return best_model
