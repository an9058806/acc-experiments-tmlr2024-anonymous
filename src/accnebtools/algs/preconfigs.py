from typing import List, Collection, Dict
import inspect
import accnebtools.algs.algorithms as embalgs


def get_algs(method_sets: List[str], emb_dims: List[int] = None, num_epochs: int = None):
    methods = []
    if isinstance(method_sets, str):
        method_sets = [method_sets]
    if emb_dims is None or isinstance(emb_dims, int):
        emb_dims = [emb_dims]
    for method_set in method_sets:
        for emb_dim in emb_dims:
            methods += _get_algs(method_set, emb_dim=emb_dim, num_epochs=num_epochs)
    return sorted(list(set(methods)), key=lambda x: x.name)


def _get_algs(method_set: str, emb_dim: int = None, num_epochs: int = None):
    if method_set == "acc_ga_steps_and_dims":
        methods = acc_ga_steps_and_dims()
    elif method_set == "acc_nc_steps_and_dims":
        methods = acc_nc_steps_and_dims()
    elif method_set == "sgcn_steps":
        methods = sgcn_steps()
    elif method_set == "acc_vs_pcapass_k4_k10":
        methods = acc_vs_pcapass_k4_k10(max_dim=emb_dim)
    elif method_set == "acc_pca_rtol_sweep":
        methods = acc_rtol_sweep(max_dim=emb_dim, num_steps=10)
    elif method_set == "acc_pca_rtol_sweep4":
        methods = acc_rtol_sweep(max_dim=emb_dim, num_steps=4)
    elif method_set == "acc_pca_rtol_sweep12":
        methods = acc_rtol_sweep(max_dim=emb_dim, num_steps=12)
    elif method_set == "pcapass_rtol_sweep":
        methods = pcapass_rtol_sweep(max_dim=emb_dim, num_steps=10)
    elif method_set == "pcapass_rtol_sweep4":
        methods = pcapass_rtol_sweep(max_dim=emb_dim, num_steps=4)
    elif method_set == "detailed_ga_analysis":
        methods = detailed_ga_analysis(max_dim=emb_dim)
    elif method_set == "detailed_ga_analysis12":
        methods = detailed_ga_analysis12(max_dim=emb_dim)
    elif method_set == "acc_pcapass_sv_spectra":
        methods = acc_pcapass_sv_spectra(max_dim=emb_dim)
    else:
        methods = get_alg_by_name({method_set}, emb_dim=emb_dim, num_epochs=num_epochs)
        if len(methods) == 0:
            raise NotImplementedError(f"Method set {method_set} not implemented.")

    return methods


def acc_ga_steps_and_dims():
    methods = []
    for steps in [0, 1, 2, 3, 4, 6, 8, 10, 12]:
        for max_dim in [4, 8, 16, 32, 64, 128, 256, 512]:
            methods.append(
                embalgs.AccAlg(
                    max_steps=steps, dimensions=max_dim
                )
            )
    return methods


def acc_nc_steps_and_dims():
    methods = []
    for steps in [0, 1, 2, 3, 4, 6, 8, 10, 12]:
        for max_dim in [4, 8, 16, 32, 64, 128, 256, 512]:
            methods.append(
                embalgs.AccAlg(
                    max_steps=steps, dimensions=max_dim
                )
            )
    return methods


def sgcn_steps():
    methods = []
    for steps in [0, 1, 2, 3, 4, 6, 8, 10, 12]:
        methods.append(
            embalgs.SgcnAlg(num_layers=steps)
        )
    return methods


def acc_vs_pcapass_k4_k10(max_dim: int):
    methods = [
        embalgs.AccAlg(max_steps=4, dimensions=max_dim, name="acc4"),
        embalgs.AccAlg(max_steps=10, dimensions=max_dim, name="acc10"),
        embalgs.PcapassAlg(max_steps=4, dimensions=max_dim, name="pcapass4"),
        embalgs.PcapassAlg(max_steps=10, dimensions=max_dim, name="pcapass10")
    ]
    return methods


def acc_rtol_sweep(max_dim: int, num_steps: int):
    methods = []
    tols = [0., 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.2, 0.5]
    for tol in tols:
        methods.append(
            embalgs.AccAlg(sv_thresholding='rtol', theta=tol, max_steps=num_steps, dimensions=max_dim)
        )
    return methods


def pcapass_rtol_sweep(max_dim: int, num_steps: int):
    methods = []
    tols = [0., 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.2, 0.5]
    for tol in tols:
        methods.append(
            embalgs.PcapassAlg(sv_thresholding='rtol', theta=tol, max_steps=num_steps, dimensions=max_dim)
        )
    return methods


def detailed_ga_analysis(max_dim: int):
    methods = [
        embalgs.AccAlg(sv_thresholding='none', max_steps=2, dimensions=max_dim, name="acc_2"),
        embalgs.AccAlg(sv_thresholding='none', max_steps=4, dimensions=max_dim, name="acc_4"),
        embalgs.AccAlg(sv_thresholding='none', max_steps=6, dimensions=max_dim, name="acc_6"),
        embalgs.PcapassAlg(sv_thresholding='none', max_steps=2, dimensions=max_dim, name="pcapass_2"),
        embalgs.PcapassAlg(sv_thresholding='none', max_steps=4, dimensions=max_dim, name="pcapass_4"),
        embalgs.PcapassAlg(sv_thresholding='none', max_steps=6, dimensions=max_dim, name="pcapass_6")
    ]
    return methods


def detailed_ga_analysis12(max_dim: int):
    methods = []
    methods.append(
        embalgs.AccAlg(sv_thresholding='none', max_steps=12, dimensions=max_dim, name="acc_12")
    )
    return methods


def acc_pcapass_sv_spectra(max_dim: int):
    methods = []
    for steps in [0, 1, 2, 3, 4, 6, 8, 10]:
        methods.append(
            embalgs.PcapassAlg(
                dimensions=max_dim,
                max_steps=steps,
                sv_thresholding='none',
                theta=0.0
            )
        )
        methods.append(
            embalgs.AccAlg(
                max_steps=steps,
                dimensions=max_dim,
                sv_thresholding='none',
                theta=0.0
            )
        )
    return methods


def get_alg_by_name(names: Collection[str], emb_dim: int = None, num_epochs: int = None, alg_kwargs: Dict = None):
    if alg_kwargs is None:
        alg_kwargs = dict()
    all_algs = inspect.getmembers(embalgs, inspect.isclass)
    out_algs = []
    for cls_name, alg in all_algs:
        if is_alg_valid(alg) and alg.name in names:
            alg = instantiate_alg(alg, dimensions=emb_dim, num_epochs=num_epochs, **alg_kwargs)
            out_algs.append(alg)
    return out_algs


def is_alg_valid(alg):
    if alg.__name__ in ("EmbeddingAlg", "EmbeddingAlgSpec", "AlgGraphSupport"):
        return False
    if hasattr(alg, 'disabled') and alg.disabled:
        return False
    return True


def instantiate_alg(alg, dimensions: int = None, num_epochs: int = None, **kwargs):
    if hasattr(alg, 'dimensions'):
        dimensions = alg.dimensions if dimensions is None else dimensions
        kwargs['dimensions'] = dimensions

    if hasattr(alg, 'num_epochs'):
        num_epochs = alg.num_epochs if num_epochs is None else num_epochs
        kwargs['num_epochs'] = num_epochs

    alg = alg(**kwargs)
    return alg
