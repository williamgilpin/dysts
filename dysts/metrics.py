"""
Metrics for comparing two time series. These metrics are included to faciliate 
benchmarking of the algorithms in this package while reducing dependencies.

For more exhaustive sets of metrics, use the external `tslearn`, `darts`, or `sktime`
"""

import numpy as np

from scipy.stats import spearmanr, pearsonr, kendalltau

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def dtw(y_true, y_pred):
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series.
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # check inputs
    if np.ndim(y_true) > 2:
        raise ValueError("y_true must be at most 2 dimensional.")
    if np.ndim(y_pred) > 2:
        raise ValueError("y_pred must be at most dimensional.")

    if np.ndim(y_true) == 1:
        y_true = y_true[:, None]
    if np.ndim(y_pred) == 1:
        y_pred = y_pred[:, None]

    # get lengths of each series
    n, m = len(y_true), len(y_pred)

    # allocate cost matrix
    D = np.zeros((n + 1, m + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    # compute cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cdist([y_true[i - 1]], [y_pred[j - 1]], metric='euclidean')
            D[i, j] += min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # compute DTW
    cost = D[-1, -1] / sum(D.shape)
    #  compute aligned series
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while (i > 0) and (j > 0):
        tb = np.argmin((D[i, j - 1], D[i - 1, j], D[i - 1, j - 1]))
        if tb == 0:
            i = i
            j = j - 1
        elif tb == 1:
            i = i - 1
            j = j
        else:
            i = i - 1
            j = j - 1
        p.insert(0, i)
        q.insert(0, j)

    return cost, D, p, q

def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def mase(y_true, y_pred, y_train=None):
    """
    Mean Absolute Scaled Error. If the time series are multivariate, the first axis is
    assumed to be the time dimension.
    """    
    if y_train is None:
        y_train = y_true
    return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[1:] - y_train[:-1]))

def mse(y_true, y_pred):
    """
    Mean Squared Error
    """
    return np.mean(np.square(y_true - y_pred))

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))

def coefficient_of_variation(y_true, y_pred):
    """
    Coefficient of Variation of the root mean squared error relative to the mean 
    of the true values
    """
    return 100 * np.std(y_true - y_pred) / np.mean(y_true)

def marre(y_true, y_pred):
    """
    Mean Absolute Ranged Relative Error
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / (np.max(y_true) - np.min(y_true)))

def ope(y_true, y_pred):
    """
    Optimality Percentage Error
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

def rmsle(y_true, y_pred):
    """
    Root Mean Squared Log Error. In case of negative values, the series is shifted
    to the positive domain.
    """
    y_true = y_true - np.min(y_true, axis=0, keepdims=True) + 1e-8
    y_pred = y_pred - np.min(y_pred, axis=0, keepdims=True) + 1e-8
    return np.sqrt(np.mean(np.square(np.log(y_pred + 1) - np.log(y_true + 1))))

def r2_score(y_true, y_pred):
    """
    R2 Score
    """
    return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / y_true)

def smape(x, y):
    """Symmetric mean absolute percentage error"""
    return 100 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y))) * 2

def spearman(y_true, y_pred):
    """
    Spearman Correlation. Returns dimensionwise mean for multivariate time series of
    shape (T, D)
    """
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.ndim != y_pred.ndim:
        raise ValueError("y_true and y_pred must have the same number of dimensions")
    
    if y_true.ndim == 1:
        return spearmanr(y_true, y_pred)[0]

    else:
        all_vals = []
        for i in range(y_true.shape[1]):
            all_vals.append(spearmanr(y_true[:, i], y_pred[:, i])[0])
        return np.mean(all_vals)

    

def pearson(y_true, y_pred):
    """
    Pearson Correlation. Returns dimensionwise mean for multivariate time series of 
    shape (T, D)
    """
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.ndim != y_pred.ndim:
        raise ValueError("y_true and y_pred must have the same number of dimensions")
    
    if y_true.ndim == 1:
        return spearmanr(y_true, y_pred)[0]

    else:
        all_vals = []
        for i in range(y_true.shape[1]):
            all_vals.append(pearsonr(y_true[:, i], y_pred[:, i])[0])
        return np.mean(all_vals)

def kendall(y_true, y_pred):
    """
    Kendall-Tau Correlation. Returns dimensionwise mean for multivariate time series of
    shape (T, D)
    """
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.ndim != y_pred.ndim:
        raise ValueError("y_true and y_pred must have the same number of dimensions")
    
    if y_true.ndim == 1:
        return kendalltau(y_true, y_pred)[0]

    else:
        all_vals = []
        for i in range(y_true.shape[1]):
            all_vals.append(kendalltau(y_true[:, i], y_pred[:, i])[0])
        return np.mean(all_vals)

from sklearn.feature_selection import mutual_info_regression
def mutual_information(y_true, y_pred):
    """
    Mutual Information. Returns dimensionwise mean for multivariate time series of
    shape (T, D). Computes the mutual information separately for each dimension and
    returns the mean.
    """
    mi = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        mi[i] = mutual_info_regression(
            y_true[:, i].reshape(-1, 1), 
            y_pred[:, i].ravel()
        )
    return np.mean(mi)

def nrmse(y_true, y_pred, eps=1e-8, scale=None):
    """
    Normalized Root Mean Squared Error

    Args:
        y_true (np.ndarray): True values of shape (T, D)
        y_pred (np.ndarray): Predicted values of shape (T, D)
        eps (float): Small value to avoid division by zero
        scale (np.ndarray): Standard deviation of the true values of shape (D,). If None,
            the standard deviation is computed from the true values.

    Returns:
        float: NRMSE
    """
    if scale is None:
        sigma = np.std(y_true, axis=0) # D
    else:
        sigma = scale
    vals = (y_true - y_pred)**2 / (sigma**2 + eps) # T x D
    return np.sqrt(np.mean(vals)) # Flatten along both dimensions

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

rmse = lambda x, y: np.sqrt(mse(x, y))

def compute_metrics(y_true, y_pred, standardize=False):
    """
    Compute multiple time series metrics
    """
    if standardize:
        scale_true, scale_pred = np.std(y_true, axis=0, keepdims=1), np.std(y_pred, axis=0, keepdims=1)
        if scale_true == 0:
            scale_true = 1
        if scale_pred == 0:
            scale_pred = 1
        y_true = (y_true - np.mean(y_true, axis=0, keepdims=1)) / scale_true
        y_pred = (y_pred - np.mean(y_pred, axis=0, keepdims=1)) / scale_pred

    metrics = dict()
    metrics["mse"] = mse(y_true, y_pred)
    metrics["mae"] = mae(y_true, y_pred)
    metrics["rmse"] = rmse(y_true, y_pred)
    metrics["nrmse"] = nrmse(y_true, y_pred)
    metrics["marre"] = marre(y_true, y_pred)
    metrics["r2_score"] = r2_score(y_true, y_pred)
    metrics["rmsle"] = rmsle(y_true, y_pred)
    metrics["smape"] = smape(y_true, y_pred)
    metrics["mape"] = mape(y_true, y_pred)
    metrics["wape"] = wape(y_true, y_pred)
    metrics["spearman"] = spearman(y_true, y_pred)
    metrics["pearson"] = pearson(y_true, y_pred)
    metrics["kendall"] = kendall(y_true, y_pred)
    metrics["coefficient_of_variation"] = coefficient_of_variation(y_true, y_pred)
    metrics["mutual_information"] = mutual_information(y_true, y_pred)
    return metrics