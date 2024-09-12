import os

if "STRUCTFEATURES_USE_NUMBA_CACHE" not in os.environ or len(os.environ["STRUCTFEATURES_USE_NUMBA_CACHE"]) != 1:
    os.environ["STRUCTFEATURES_USE_NUMBA_CACHE"] = "1"