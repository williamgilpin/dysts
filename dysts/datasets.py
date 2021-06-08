import json
import warnings
import numpy as np
import os
import pkg_resources


class TimeSeriesDataset:
    """A data structure for handling time series. Reads and writes from JSON"""
    def __init__(self, datapath=None):
        if not datapath:
            pass
        else:
            with open(datapath, "r") as file:
                self.dataset = json.load(file)
                
        self.is_multivariate = any([len(np.squeeze(np.array(self.dataset[item]["values"])).shape) > 1 for item in self.dataset])
        
    def to_pandas(self, standardize=True, filling="constant"):
        """
        data (dict): A dictionary with index given by top-level keys
        
        Args:
            standardize (bool): whether to standardize each dataset before flattening
            filling (str): For multivariate time series of varying demensionality, values to pad 
                in empty columns.
        Returns:
            data_pd (pd.DataFrame): a 2D dataset consisting of indexed time series
        """
        data = self.dataset
        all_names = np.array(list(data.keys()))
        all_times = np.vstack([data[item]["time"] for item in data])
        
#         all_values = np.vstack([data[item]["values"] for item in data])
        all_values = self.to_array(standardize=standardize)
        
        all_indices = (np.arange(all_times.shape[0])[:, None] * np.ones(all_times.shape[1])[None, :]).astype(int)
        all_names = all_names[all_indices]
        all_data = np.vstack([np.ravel(item) for item in (all_indices, all_times, all_values)])
        data_pd = pd.DataFrame(all_data.T, columns=["id", "time", "values"], index=np.ravel(all_names))
        return data_pd
    
    def to_array(self, standardize=False):
        """Return the values of a time series as a matrix of shape (B, T)"""
        data = self.dataset
        all_values = np.vstack([data[item]["values"] for item in data])
        if standardize:
            all_values = (all_values - np.mean(all_values, axis=1, keepdims=True)) / np.std(all_values, axis=1, keepdims=True)
        return all_values
    
    def dump(self, path):
        with open(path, 'w') as file:
            json.dump(self.dataset, file, indent=4)  




def load_file(filename):
    """Locate and import from the module data directory"""
    base_path = pkg_resources.resource_filename('dysts', 'data')
    data_path = os.path.join(base_path, filename)
    dataset = TimeSeriesDataset(data_path)
    return dataset

def load_continuous(subsets="all"):
    """
    Load dynamics from continuous dynamical systems
    """
    dataset = load_file("dataset_univariate__pts_per_period_100__periods_10.json")
    return dataset

def load_discrete(subsets="all"):
    """Load dynamics from discrete dynamical systems
    """
    dataset = load_file("dataset_univariate__pts_per_period_100__periods_10.json")
    return dataset

def make_dataset(random_state=None):
    pass
