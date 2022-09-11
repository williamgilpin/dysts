#!/usr/bin/python

import os
import json
import numpy as np

# import dysts
# from dysts.utils import find_significant_frequencies
# from dysts.flows import *
# from dysts.base import *

import sktime.datasets
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_processing import from_nested_to_3d_numpy, from_3d_numpy_to_nested

all_scores = dict()
np.random.seed(0)

# cwd = os.getcwd()
cwd = os.path.dirname(os.path.realpath(__file__))
output_path = cwd + "/results/baseline_transfer_learning.json"
print("Saving data to: ", output_path)

dataset_names = np.genfromtxt(cwd + "/resources/ucr_ea_names.txt", dtype='str')

try:
    with open(output_path, "r") as file:
        all_scores = json.load(file)
except FileNotFoundError:
    all_scores = dict()

for data_ind, name in enumerate(dataset_names):
    
    if name in all_scores.keys():
        if "score_tsfresh" in all_scores[name].keys():
            print("Skipped " + name, flush=True)
            continue
    print("Evaluating " + name, flush=True)
    
    all_scores[name] = dict()
    X_train, y_train = sktime.datasets.load_UCR_UEA_dataset(name, split="train", return_X_y=True)
    X_test, y_test = sktime.datasets.load_UCR_UEA_dataset(name, split="test", return_X_y=True)
    
    X_train_np = from_nested_to_3d_numpy(X_train)
    X_test_np = from_nested_to_3d_numpy(X_test)
    
    X_train_np -= np.mean(X_train_np, axis=-1, keepdims=True)
    X_train_np /= np.std(X_train_np, axis=-1, keepdims=True)
    X_test_np -= np.mean(X_test_np, axis=-1, keepdims=True)
    X_test_np /= np.std(X_test_np, axis=-1, keepdims=True)

    transformer = TSFreshFeatureExtractor(show_warnings=False)
    X_train_featurized = transformer.fit_transform(from_3d_numpy_to_nested(X_train_np))
    X_test_featurized = transformer.fit_transform(from_3d_numpy_to_nested(X_test_np))

    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X_train_featurized, y_train)

    score = model.score(X_test_featurized, y_test)
    
    all_scores[name]["score_tsfresh"] = score
    
    print(name, score, flush=True)
    
    with open(output_path, 'w') as file:
        json.dump(all_scores, file, indent=4)
    
    

