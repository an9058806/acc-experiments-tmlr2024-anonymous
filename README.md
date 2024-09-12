# ACC experiments framework

This repository contains the graph experiments and benchmarking framework used for the paper "Full-Rank Unsupervised Node Embeddings for Directed
Graphs via Message Aggregation" submitted to TLMR 2024. See usage instructions below.


## Overview

This library is an excerpt from larger unpublished library for benchmarking node embedding models. The core functionality of 
the library is to compartmentalize node embedding models by running them in different conda environments.
However, for reproducing the results from the paper, only a single conda environment is needed, meaning that some
of the functionalities of this library are redundant. Still, we keep them for transparency and reproducibility purposes.

The structrure of the library is as follows
```
acc-experiments-tmlr2024
│ README.md
│ index_and_download_datasets.py  # See usage below
│ required_packages.txt  # Lists the required packages without version numbers
│ basic_environment_cpu.yml  # Conda environment file without CUDA support
│ basic_environment_cu118.yml  # Conda environment file using CUDA=11.8
│
└─── data/  # Contains the datasets
└─── experiments/  # Contains the main function scripts for running the experiments
└─── methods/  # Contains the main functions for calling the node embedding models, including ACC and PCAPass
└─── run_scripts/  # Contain bash scripts for reproducing the results of the paper, see below
└─── src/
│   └─── accnebtools
│       └─── alg/
│       │    │ algorithms.py  # Contains the default hyperparameters for all models
│       │    │ preconfigs.py  # Defines sets of models used by the scripts in run_scripts/
│       │    │ utils.py       # Contains the core pipelining of the library
│       └─── data/  # Data and graph processing
│       └─── experiments/  # Code for embedding-based graph alignment and node classification
│       └─── ssgnns/  # Implementations of used self-supervised graph neural networks
└─── structfeatures/  # Seperate library used to compute node degrees and local clustering coefficients from edge lists



```

## Installation

### Install dependencies

You first need to have a [conda](https://docs.anaconda.com/miniconda/) installed on your system.
It is possible that [mamba](https://mamba.readthedocs.io/en/latest/index.html) will also work, but this has not been tested.

Then you need to install the packages listed in [the requirements file](required_packages.txt) into an conda environment 
named 'acc_neb_env'. You can do this conveniently by first using the provided [environment files](basic_environment_cu118.yml):
```commandline
conda env create --file basic_environment_${CUDA}.yml
```
where `${CUDA}` is either `cu118` for GPU, or `cpu` for only CPU support.
Then, activate the environment, and install the following dependencies via pip:
```commandline
conda activate acc_neb_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/${CUDA}
pip install torch_geometric
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/${CUDA}/repo.html
pip install ogb==1.3.6
```

To run [clustering_rank_deficiency_noise_demo.py](experiments/clustering_rank_deficiency_noise_demo.py) and 
[squirrel_analysis.py](experiments/squirrel_analysis.py), you also need to install [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) and [umap](https://umap-learn.readthedocs.io/en/latest/).


### Install framework
To install the framework, while standing in the top repo directory, run
```commandline
pip install -e structfeatures/
pip install -e .
```

### Install ACC
Clone ACC from [https://github.com/an9058806/acc-mp-anonymous](https://github.com/an9058806/acc-mp-anonymous) into a suitable location, and then install ACC using
```commandline
pip install -e <path/to/ACC-repo>
```


## Datasets setup

All the graph alignment datasets are available under `data/inrepo`. To index these datasets and to download the node 
classification datasets, please run
```commandline
python index_and_download_datasets.py
```

Snap Patents has to be downloaded manually from here [LINK](https://github.com/CUAI/Non-Homophily-Large-Scale?tab=readme-ov-file#dataset-preparation).
Once this is done, you can prepare and index it by providing the path to the downloaded .mat file to the dataset preparation script:
```commandline
python index_and_download_datasets.py --snap-patents-path <path/to/snap_patents.mat>
```

## Usage

After the installation and dataset setup, the experimental results from the paper can be reproduced using the scripts under
the [run_scripts](run_scripts) directory.

- For the clustering results in Figure 2, run [run_scripts/clustering_and_rank_deficiency_demo.sh](run_scripts/clustering_and_rank_deficiency_demo.sh)
- For the singular value and graph alignment displacement correlation results in Figures 3c, 3d and 9f, run [run_scripts/graph_alignment_analysis.sh](run_scripts/graph_alignment_analysis.sh)
- For the singular value spectra results used in Figures 4 and 7, run [run_scripts/singular_value_spectra.sh](run_scripts/singular_value_spectra.sh)
- For the ACC and PCAPass comparison in Figures 5, 6, and 8, run [run_scripts/graph_alignment_acc_vs_pcapass](run_scripts/graph_alignment_acc_vs_pcapass.sh)
- For the node classification results in Table 1 and 3, run [run_scripts/node_classification.sh](run_scripts/node_classification.sh)
- For the effect of the number of message-passing iterations in Figure 9a-9e, 10, and 11, run [run_scripts/acc_steps_and_max_dim.sh](run_scripts/acc_steps_and_max_dim.sh)
- For the analysis of the Squirrel dataset, run [run_scripts/squirrel_analysis.sh](run_scripts/squirrel_analysis.sh)


## Troubleshooting

You may encounder errors along the lines of
```
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found.
```
or
```commandline
OSError: libcusparse.so.11: cannot open shared object file: No such file or directory.
```
Both of these can often be fixed by updating the LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
You can set up the 'acc_neb_env' conda environment to always use this env var using the following command:
```commandline
conda env config vars set LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib
```
If you are running the code via PyCharm, you might also need to set this path inside the IDE.


In some setup, for example in Google Cloud, `conda` is not a valid command for interacting with conda from subprocesses
from within python. I'm not sure why this is, but a fix is to do the following replacement on Line 178 in [src/accnebtools/algs/utils.py](src/accnebtools/algs/utils.py).
```diff
- conda_command = f"conda run -n {alg.spec.env_name}".split()
+ conda_command = f"__conda_exe run -n {alg.spec.env_name}".split()
```

