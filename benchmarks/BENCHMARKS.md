## Benchmark experiments

Summaries of the different benchmark experiments. To re-run experiments reported in our manuscript, open the corresponding Jupyter Notebook file for a step-by-step guide

# Forecasting

See `figure_forecasting_benchmarks_figures.ipynb` for an overview

+ `find_hyperparameters.py` computes hyperparameters across all forecasting models separately for each dynamical system
+ `compute_benchmarks.py` uses the best hyperparameters to train and score a models on the test data.

# Importance sampling 

See `figure_importance_sampling.ipynb` for an overview

+ `importance_sampling.py` uses importance sampling to improve training on an LSTM forecasting model

# Tranfer learning

See `figure_transfer_learning.ipynb` for an overview

+ `surrogate_transfer_learning.py` computes the transfer learning benchmark on the UCR database
+ `sweep_surrogate_transfer_learning.py` recalculates the transfer learning results for different numbers of dynamical systems
+ `sweep_surrogate_transfer_learning.py` recalculates the transfer learning results for different numbers of dynamical systems
+ `random_surrogate_transfer_learning.py` calculates a baseline with random timescales
+ `baseline_transfer_learning.py` calculates a baseline on the raw UCR time series 

# Symbolic regression

See `figure_symbolic_regression_benchmark.ipynb` for an overview

+ `symbolic_regression_benchmarks.py` calculates all of the benchmarks