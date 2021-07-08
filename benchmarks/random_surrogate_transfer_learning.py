#!/usr/bin/python

import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler


import dysts
from dysts.utils import find_significant_frequencies
from dysts.flows import *
from dysts.base import *

from resources.classification_models import Autoencoder, TimeSeriesCollection

import sktime.datasets
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_processing import from_nested_to_3d_numpy, from_3d_numpy_to_nested

all_scores = dict()
np.random.seed(0)
attractor_list = get_attractor_list()

SEQUENCE_LENGTH = 100
BATCH_SUBSAMPLE = 10000
# BATCH_SUBSAMPLE = 5000
# attractor_list = np.random.choice(attractor_list, 40)
# attractor_list = np.random.choice(attractor_list, 80)

# cwd = os.getcwd()
cwd = os.path.dirname(os.path.realpath(__file__))
output_path = cwd + "/results/transfer_learning.json"
print("Saving data to: ", output_path)

dataset_names = np.genfromtxt(cwd + "/resources/ucr_ea_names.txt", dtype='str')

try:
    with open(output_path, "r") as file:
        all_scores = json.load(file)
except FileNotFoundError:
    all_scores = dict()
    
## Create trajectory ensemble at a random frequency
all_sols = list()
for equation_ind, equation_name in enumerate(attractor_list):
    equation = getattr(dysts.flows, equation_name)()
    main_period = np.random.choice(np.arange(10, 100), replace=True) # randomly sample the density
    sol = equation.make_trajectory(1000, resample=True, pts_per_period=int(main_period))
    if len(sol) < 10: # skip undersampled trajectories
        continue
    all_sols.append(standardize_ts(sol)[:, 0])
all_sols = np.array(all_sols).T
print("Finished computing surrogate ensemble.", flush=True)

for data_ind, name in enumerate(dataset_names):
    
    if name in all_scores.keys():
        if "score_transfer" in all_scores[name].keys():
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
    
    ## Train model on ensemble
    model = Autoencoder()
    training_data = TimeSeriesCollection(all_sols, SEQUENCE_LENGTH)
    subset_indices = np.random.choice(np.arange(0, len(training_data)), BATCH_SUBSAMPLE, replace=False) # subsample all traj
    train_dataloader = DataLoader(Subset(training_data, subset_indices), batch_size=64, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, outputs = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(inputs, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print("Finished training autoencoder.", flush=True)
    
    X_train_nn = from_3d_numpy_to_nested(model.encoder(torch.tensor(X_train_np, dtype=torch.float32)).detach().numpy())
    X_test_nn = from_3d_numpy_to_nested(model.encoder(torch.tensor(X_test_np, dtype=torch.float32)).detach().numpy())

    transformer = TSFreshFeatureExtractor(show_warnings=False)
    X_train_featurized = transformer.fit_transform(X_train_nn)
    X_test_featurized = transformer.fit_transform(X_test_nn)

    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X_train_featurized, y_train)

    score = model.score(X_test_featurized, y_test)
    
    all_scores[name]["score_transfer"] = model.score(X_test_featurized, y_test)
    
    print(name, score, flush=True)
    
    with open(output_path, 'w') as file:
        json.dump(all_scores, file, indent=4)
    
    

