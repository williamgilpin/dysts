import warnings
import os, json, gzip
import numpy as np
import pandas as pd
import pkg_resources

## Check for optional time series featurizer
try:
    from tsfresh import extract_features
    from tsfresh import select_features
    from tsfresh.utilities.dataframe_functions import impute
except ImportError:
    _has_tsfresh = False
else:
    _has_tsfresh = True


## Check for optional datasets
try:
    from dysts_data.dataloader import get_datapath
except ImportError:
    _has_data = False
else:
    _has_data = True

from .utils import *


class TimeSeriesDataset:
    """A data structure for handling time series. Reads and writes from JSON
    
    Assumes that all time series have the same number of timepoints. In the case
    of multivariate series with different numbers of features / dimensions, the
    missing dimensionality is padded
    
    
    Args:
        data (dict, optional): A time series dataset, consisting of a dictionary
            with index given by top-level names of different time series.
        datapath (string, optional): The path to a JSON dataset. Defaults to creating
            an empty dataset object
    
    Attributes:
        rank (list): The original length of each time series
        is_multivariate (bool): Whether the time series dataset is multivariate
        names (list): The names of the systems in the dataset
    
    """

    def __init__(self, datapath=None, data=None):
        if datapath is not None:
            ext = os.path.splitext(datapath)[1]
            if ext == ".json":
                with open(datapath, "r") as file:
                    self.dataset = json.load(file)
            elif ext == ".gz":
                with gzip.open(datapath, 'rt', encoding="utf-8") as file:
                    self.dataset = json.load(file)
            else:
                raise ValueError("Unrecognized file extension")
        if data is not None:
            self.dataset = data

        self.rank = np.max(
            np.array(
                [
                    len(np.squeeze(np.array(self.dataset[item]["values"])).shape)
                    for item in self.dataset
                ]
            )
        )
        self.is_multivariate = self.rank > 1

        if self.is_multivariate:
            self.max_d = np.max(
                [
                    np.array(self.dataset[item]["values"]).shape[-1]
                    for item in self.dataset
                ]
            )
        else:
            self.max_d = 1

        for key in self.dataset:
            self.dataset[key]["values"] = np.array(self.dataset[key]["values"])
        self.names = np.array(list(self.dataset.keys()))

    def get_rowvalues(self, key):
        """Retrieve the value of a field from each row"""
        return np.array([self.dataset[item][key] for item in self.dataset])

    def __getitem__(self, key):
        return self.dataset[key]

    def __setitem__(self, key, val):
        self.dataset[key] = val

    def trim_series(self, start_val, stop_val):
        """
        Trims a time series in-place. This is not reversible without re-initializing
        """
        for key in self.dataset:
            self.dataset[key]["values"] = self.dataset[key]["values"][
                start_val:stop_val
            ]
            self.dataset[key]["time"] = self.dataset[key]["time"][start_val:stop_val]

    def to_pandas(self, standardize=True, filling="constant"):
        """
        Convert a data dictionary to a DataFrame
        Draws values from self.data (dict), a dictionary with index given by top-level keys
        
        Args:
            standardize (bool): whether to standardize each dataset before flattening
            filling (str): For multivariate time series of varying demensionality, values to pad 
                in empty columns.
        Returns:
            data_pd (pd.DataFrame): a 2D dataset consisting of indexed time series
        """
        data = self.dataset
        all_names = self.names
        all_times = np.vstack([data[item]["time"] for item in data])

        #         all_values = np.vstack([data[item]["values"] for item in data])
        all_values = self.to_array(standardize=standardize)
        # self.max_d

        if self.is_multivariate:
            all_indices = (
                np.arange(all_times.shape[0])[:, None]
                * np.ones(all_times.shape[1])[None, :]
            ).astype(int)
            all_values = np.squeeze(np.reshape(all_values, (-1, self.max_d)))
            all_indices, all_times = [
                np.reshape(item, (-1, 1)) for item in (all_indices, all_times)
            ]
            all_data = np.hstack([all_indices, all_times, all_values]).T
            column_names = ["id", "time"] + [f"values_{i}" for i in range(self.max_d)]
        else:
            all_indices = (
                np.arange(all_times.shape[0])[:, None]
                * np.ones(all_times.shape[1])[None, :]
            ).astype(int)
            all_data = np.vstack(
                [np.ravel(item) for item in (all_indices, all_times, all_values)]
            )
            column_names = ["id", "time", "values"]

        all_names = all_names[all_indices]
        data_pd = pd.DataFrame(
            all_data.T, columns=column_names, index=np.ravel(all_names)
        )
        return data_pd

    def to_array(self, standardize=False):
        """
        Return the values of a time series as a matrix of shape (B, T, D) or (B, T)
        """
        data = self.dataset
        # all_values = np.vstack([data[item]["values"].T for item in data])
        all_values = np.squeeze(
            np.dstack(
                [pad_axis(np.array(data[key]["values"]), 10, axis=-1).T for key in data]
            ).T
        )
        if standardize:
            scale = np.std(all_values, axis=1, keepdims=True)
            scale[scale == 0] = 1
            all_values = (
                all_values - np.mean(all_values, axis=1, keepdims=True)
            ) / scale
        return all_values

    def dump(self, path):
        with open(path, "w") as file:
            json.dump(self.dataset, file, indent=4)

def featurize_timeseries(dataset):
    """
    Extract features from a TimeSeriesDataset

    Args:
        dataset (TimeSeriesDataset): a dataset of time series

    Returns:
        extracted_features (pd.DataFrame): a dataframe of extracted features
    """
    if not _has_tsfresh:
        raise ImportError("Install the package tsfresh in order to use this function.")
    all_datasets = dataset.to_pandas(standardize=True)
    names = np.unique(dataset.to_pandas().index.to_list())
    extracted_features = extract_features(
        all_datasets, column_id="id", column_sort="time"
    )
    extracted_features = extracted_features.dropna(axis=1)  # drop nans
    extracted_features = extracted_features.loc[
        :, extracted_features.std() > 0.0
    ]  # drop nonvarying
    extracted_features -= extracted_features.mean()  # subtract featurewise mean
    extracted_features = extracted_features.set_index(names)
    return extracted_features


def convert_json_to_gzip(fpath, encoding="utf-8", delete_original=False):
    """
    Convert a json file to a gzip file in a format that can be easily read by the
    `dysts` package. By default, the gzip file will be saved with the same name and
    in the same directory as the json file, but with a ".gz" extension.

    Args:
        fpath (str): Path to the json file to be converted
        encoding (str): Encoding to use when writing the gzip file
        delete_original (bool): Whether to delete the original json file after
            conversion. Default is False.

    Returns:
        None
    
    """
    if os.path.splitext(fpath)[1] == ".gz":
        warnings.warn("File already gzipped, exiting without conversion")
        return None
    
    with open(fpath, 'r') as file:
        data = json.load(file)
        
    with gzip.open(fpath + ".gz", 'wt', encoding=encoding) as file:
        json.dump(data, file, indent=4)

    if delete_original:
        os.remove(fpath)

import zipfile
def load_json(fpath):
    """
    Load either a raw, zipped, or gzipped json file.

    Args:
        fpath (str): Path to the json file to be loaded

    Returns:
        data (dict): Dictionary containing the data from the json file
    """
    if os.path.splitext(fpath)[1] == ".json":
        with open(fpath, 'r') as file:
            data = json.load(file)
        return data
    elif os.path.splitext(fpath)[1] == ".gz":
        with gzip.open(fpath, 'rt') as file:
            data = json.load(file)
        return data
    elif os.path.splitext(fpath)[1] == ".zip":
        with zipfile.ZipFile(fpath, 'r') as file:
            data = json.load(file.open(file.namelist()[0]))
        return data
    else:
        raise ValueError("File must be a json or gzipped json file")

def load_file(filename):
    """Locate and import from the module data directory"""
    if not _has_data:
        warnings.warn(
                    "Data module not found. To use precomputed datasets, "+ \
                        "please install the external data repository "+ \
                            "\npip install git+https://github.com/williamgilpin/dysts_data"
        )
    # base_path = pkg_resources.resource_filename("dysts", "data")
    base_path = get_datapath()
    data_path = os.path.join(base_path, filename)
    dataset = TimeSeriesDataset(data_path)
    return dataset


def load_dataset(
    subsets="train", univariate=True, granularity="fine", data_format="object", noise=False, 
    split_fraction=5/6,
    **kwargs
):
    """
    Load dynamics from continuous dynamical systems. 
    
    Args:
        subsets ("train" | "train_val" | "test" | "test_val" | "train_all" | "test_all"): Which dataset to draw.
            Train and train val correspond to the same time series split 5/6 of the way through,
            while "test" and "test_val" both represent a trajectory emanating from a different
            initial condition, split 5/6 of the way though. "train_all" and "test_all" return full trajectories
            without splits
        univariate (bool): Whether to use one coordinate, or all for each system.
        granularity ("course" | "fine"): Whether to use fine or coarsely-spaced samples
        data_format ("object" | "numpy" | "pandas"): The format to return
        noise (bool): Whether to include stochastic forcing
        split_fraction (float): The fraction of the time series to hold out as a test/val partition
        kwargs (dict): keyword arguments passed to the data formatter
    
    Returns:
        dataset (TimeSeriesDataset): A collection of time series datasets
    """
    period = 12
    granval = {"coarse": 15, "fine": 100}[granularity]

    dataset_name = subsets.split("_")[0]

    if univariate:
        data_path = f"{dataset_name}_univariate__pts_per_period_{granval}__periods_{period}.json"
    else:
        data_path = f"{dataset_name}_multivariate__pts_per_period_{granval}__periods_{period}.json"
    
    if noise:
        name_parts = list(os.path.splitext(data_path))
        data_path = "".join(name_parts[:-1] + ["_noise"] + [name_parts[-1]])
    
    ## append a .gz extension if it's not there already
    if os.path.splitext(data_path)[1] != ".gz":
        data_path += ".gz"
    dataset = load_file(data_path)
    
    split_point = int(split_fraction * period) * granval
    if subsets == "train":
        dataset.trim_series(0, split_point)
    if subsets == "test":
        dataset.trim_series(0, split_point)
    if "val" in subsets:
        dataset.trim_series(split_point, -1)

    if data_format == "object":
        return dataset
    if data_format == "pandas":
        return dataset.to_pandas(**kwargs)
    if data_format == "numpy":
        return dataset.to_array(**kwargs)
    else:
        raise ValueError("Return format not recognized.")
        return None


# def load_continuous(subsets="train", univariate=True, granularity="fine"):
#     """
#     Load dynamics from continuous dynamical systems

#     Args:
#         subsets ("train" | "val" | "test" | "test_val"): Which dataset to draw.
#         univariate (bool): Whether to use one coordinate, or all for each system.
#         granularity ("course" | "fine"): Whether to use fine or coarsely-spaced samples

#     Returns:
#         dataset (TimeSeriesDataset): A collection of time series dataset
#     """
#     period = {"train": "10", "test": "2", "val": "2"}[subsets]
#     granval = {"coarse": "15", "fine": "100"}[granularity]
#     split_point = 5/6*period
#     # self.trim_series(split_point, -1)
#     if univariate:
#         dataset = load_file(f"{subsets}_univariate__pts_per_period_{granval}__periods_{period}.json")
#     else:
#         dataset = load_file(f"{subsets}_multivariate__pts_per_period_{granval}__periods_{period}.json")
#     return dataset

# def load_discrete(subsets="all"):
#     """Load dynamics from discrete-time dynamical systems
#     """
#     dataset = load_file("dataset_univariate__pts_per_period_100__periods_10.json")
#     return dataset

# def make_dataset(random_state=None):
#     pass
