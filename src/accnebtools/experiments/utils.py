import abc
import dataclasses as dc
from typing import Sequence, List, Literal, Iterable, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dc.dataclass(frozen=True)
class PerformanceScores:
    scores_name: str
    model_name: str
    model_repr: str


class Evaluator:

    def __init__(self, random_state: int, with_train_eval: bool, train_ratio: Optional[float],
                 n_splits: int = 5, n_repeats: int = 3, stratified: bool = True
                 ):
        self.random_state = random_state
        self.with_train_eval = with_train_eval
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.train_ratio = train_ratio
        self.stratified = stratified

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> List['PerformanceScores']:
        pass

    @abc.abstractmethod
    def cv_evaluate(self, *args, **kwargs) -> List['PerformanceScores']:
        pass


PP_MODE = Literal['none', 'standardize', 'whiten']
PP_EMBS = Iterable[Tuple[PP_MODE, np.ndarray]]


def preprocess(embeddings: np.ndarray, pp_mode: PP_MODE):
    if pp_mode is None or pp_mode == 'none':
        pass
    elif pp_mode == 'standardize':
        embeddings = StandardScaler().fit_transform(embeddings)
    elif pp_mode == 'whiten':
        embeddings = PCA(whiten=True).fit_transform(embeddings)
    else:
        raise NotImplementedError(f"pp_mode '{pp_mode}' not implemented.")
    return embeddings


def pp_embeddings_generator(embeddings: np.ndarray, pp_modes: Sequence[str]):
    assert not isinstance(pp_modes, str)

    remaped_pp_modes = []
    for pp_mode in pp_modes:
        if pp_mode == "all":
            remaped_pp_modes += ['none', 'standardize', 'whiten']
        else:
            remaped_pp_modes.append(pp_mode)
    pp_modes = sorted(list(set(remaped_pp_modes)))

    for pp_mode in pp_modes:
        X = preprocess(embeddings, pp_mode=pp_mode)
        yield pp_mode, X
