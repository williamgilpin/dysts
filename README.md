## Main installation

git checkout reservoir_computing
git submodule init
git submodule update

To run the experiments and get the rank over all dynamical systems, run: 

python rc_chaos/Methods/Models/esn/esn_rc_dyst_copy.py 



## Dysts Installation

Install from PyPI

    pip install dysts

To obtain the latest version, including new features and bug fixes, download and install the project repository directly from GitHub

    git clone https://github.com/williamgilpin/dysts
    cd dysts
    pip install -I . 

Test that everything is working

    python -m unittest

Alternatively, to use this as a regular package without downloading the full repository, install directly from GitHub

    pip install git+git://github.com/williamgilpin/dysts

The key dependencies are

+ Python 3+
+ numpy
+ scipy
+ pandas
+ sdeint (optional, but required for stochastic dynamics)
+ numba (optional, but speeds up generation of trajectories)

These additional optional dependencies are needed to reproduce some portions of this repository, such as benchmarking experiments and estimation of invariant properties of each dynamical system:

+ nolds (used for calculating the correlation dimension)
+ darts (used for forecasting benchmarks)
+ sktime (used for classification benchmarks)
+ tsfresh (used for statistical quantity extraction)
+ pytorch (used for neural network benchmarks)


## Benchmarks

The benchmarks reported in our preprint can be found in [`benchmarks`](benchmarks/). An overview of the contents of the directory can be found in [`BENCHMARKS.md`](benchmarks/BENCHMARKS.md), while individual task areas are summarized in corresponding Jupyter Notebooks within the top level of the directory.

## Contents

+ Code to generate benchmark forecasting and training experiments are included in [`benchmarks`](benchmarks/)
+ Pre-computed time series with training and test partitions are included in [`data`](dysts/data/)
+ The raw definitions metadata for all chaotic systems are included in the database file [`chaotic_attractors`](dysts/data/chaotic_attractors.json). The Python implementations of differential equations can be found in [`the flows module`](dysts/flows.py)
