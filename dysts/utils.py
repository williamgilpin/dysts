"""
Helper utilities for working with time series arrays. This module is intended to have no
dependencies on the rest of the package.
"""
import numpy as np
from numpy.fft import rfft

import warnings
from scipy.integrate import solve_ivp
from scipy.signal import resample, periodogram
from scipy.signal.windows import blackmanharris
from collections import deque
from functools import partial

try:
    from sdeint import itoint
except ImportError:
    _has_sdeint = False
else:
    _has_sdeint = True

import pkg_resources

# @njit
def polar_to_cartesian(r, th):
    """Convert polar coordinates to 2D Cartesian coordinates"""
    x, y = r * np.cos(th), r * np.sin(th)
    return x, y

# @njit
def cartesian_to_polar(x, y):
    """Convert 2D cartesian coordinates to polar coordinates"""
    th = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, th

def signif(x, figs=6):
    """Round a float to a fixed number of significant digits
    
    Args:
        x (float): The number to round
        figs (int): the desired number of significant figures
    """
    if x == 0 or not np.isfinite(x):
        return x
    figs = int(figs - np.ceil(np.log10(abs(x))))
    return round(x, figs)

def dict_loudassign(d, key, val):
    """
    Assign a key val pair in a dict, and print the result
    """
    print(key + ": " + str(val), flush=True)
    d[key] = val
    return d

def nested_dict_invert(ndict, levels=(0,1)):
    """
    Invert a nested dictionary.
    
    Args:
        ndict (dict): The nested dictionary to invert.
        levels (tuple): The levels of the nested dictionary to invert. Not Implemented.

    Returns:
        inverted (dict): The inverted dictionary.
    """
    inverted = dict()
    for key1 in ndict:
        for key2 in ndict[key1]:
            if key2 not in inverted:
                inverted[key2] = dict()
            inverted[key2][key1] = ndict[key1][key2]
    return inverted

def nanmean_trimmed(arr, percentile=0.5, axis=None):
    """
    Compute the trimmed mean of an array along a specified axis, ignoring NaNs.

    Args:
        arr (np.array): Input array
        percentile (float): The fraction of data to be trimmed. Should be between 0 and
            1. Default is 0.5.
        axis (int): Axis along which to compute the trimmed mean. If None (the default),
            compute over the whole array.
    Returns:
        Trimmed mean of the input array along the specified axis.
    """
    if axis is None:
        arr = arr.ravel()
        axis = 0
    
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")
    
    # Calculate the indices corresponding to the trimming percentage
    n = arr.shape[axis]
    lower = int(n * percentile / 2)
    upper = n - lower
    
    # Sort the array along the specified axis, ignoring NaNs
    sorted_arr = np.partition(arr, (lower, upper), axis=axis)
    sorted_slice = [slice(None)] * arr.ndim
    sorted_slice[axis] = slice(lower, upper)
    sorted_arr = sorted_arr[sorted_slice]
    
    # Calculate the mean along the specified axis, ignoring NaNs
    return np.nanmean(sorted_arr, axis=axis)

def standardize_ts(a, scale=1.0):
    """Standardize an array along dimension -2
    For dimensions with zero variance, divide by one instead of zero
    
    Args:
        a (ndarray): a matrix containing a time series or batch of time series
            with shape (T, D) or (B, T, D)
        scale (float): the number of standard deviations by which to scale
    
    Returns:
        ts_scaled (ndarray): A standardized time series with the same shape as 
            the input
    """
#     if len(a.shape) == 1: a = a[:, None]
    stds = np.std(a, axis=-2, keepdims=True)
    stds[stds==0] = 1
    ts_scaled = (a - np.mean(a, axis=-2, keepdims=True))/(scale*stds)
    return ts_scaled

def integrate_dyn(f, ic, tvals, noise=0, dtval=None, **kwargs):
    """
    Given the RHS of a dynamical system, integrate the system
    noise > 0 requires the Python library sdeint (assumes Brownian noise)
    
    Args:
        f (callable): The right hand side of a system of ODEs.
        ic (ndarray): the initial conditions
        noise_amp (float or iterable): The amplitude of the Langevin forcing term. If a 
            vector is passed, this will be different for each dynamical variable
        dtval (float): The starting integration timestep. This will be the exact timestep for 
            fixed-step integrators, or stochastic integration.
        kwargs (dict): Arguments passed to scipy.integrate.solve_ivp.
        
    Returns:
        sol (ndarray): The integrated trajectory
    """
    ic = np.array(ic)
    
    if np.isscalar(noise):
        if noise > 0:
            noise_flag = True
        else:
            noise_flag = False
    else:
        if np.sum(np.abs(noise)) > 0:
            noise_flag = True
        else:
            noise_flag = False

    if noise_flag:
        if not _has_sdeint:
            raise ImportError("Please install the package sdeint in order to integrate with noise.")
        gw = lambda y, t: noise * np.diag(ic)
        fw = lambda y, t: np.array(f(y, t))
        tvals_fine = np.linspace(np.min(tvals), np.max(tvals), int(np.ptp(tvals)/dtval))
        sol_fine = itoint(fw, gw, np.array(ic), tvals_fine).T
        sol = np.vstack([resample(item, len(tvals)) for item in sol_fine])
    else:
        #dt = np.median(np.diff(tvals))
        fc = lambda t, y : f(y, t)
        sol0 = solve_ivp(fc, [tvals[0], tvals[-1]], ic, t_eval=tvals, first_step=dtval, **kwargs)
        sol = sol0.y
        #sol = odeint(f, np.array(ic), tvals).T

    return sol

def pad_axis(arr, d, axis=-1, padding=0):
    """
    Pad `axis` of `arr` with a constant `padding` to a desired shape
    """
    padding_length = d - arr.shape[axis]
    
    if padding_length <= 0:
        return arr
    if padding_length > 0:
        slice_val = padding + np.zeros_like(np.take(arr, 0, axis=axis))
        padding_chunk = np.stack([slice_val for i in range(padding_length)], axis=axis)
    return np.concatenate([arr, padding_chunk], axis=axis)

def pad_to_shape(arr, target_shape):
    """
    Given an array, and a target shape, pad the dimensions in order to reach the desired shape
    Currently, if the rank of the array is lower than the target shape, singleton
    dimensions are appended to the rank
    
    Args:
        arr (ndarray): The array to pad.
        target_shape (iterable): The desired shape.
    
    Returns:
        arr (ndarray): The padded array,
    """
    rank_difference = len(target_shape) - len(arr.shape)
    if rank_difference > 0:
        for i in range(rank_difference):
            arr = arr[..., None]
    
    for axis, target in enumerate(target_shape):
        arr = pad_axis(arr, target, axis=axis)
        
    return arr

def integrate_weiner(f, noise_amp, ic, tvals):
    """
    Given the RHS of a dynamical system, integrate the 
    system assuming Brownian noise
    Requires the Python library sdeint
    
    Args:
        f (callable): the right hand side of a system of ODE
        noise_amp (float): the amplitude of the Langevin forcing term
    """
    sol = integrate_dyn(f, ic, tvals, noise=noise_amp)
    return sol

def group_consecutives(vals, step=1):
    """
    Return list of consecutive lists of numbers from vals (number list).
    
    References:
        Modified from the following
        https://stackoverflow.com/questions/7352684/
        how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy 
    """
    run = list()
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def find_psd(y, window=True):
    """
    Find the power spectrum of a signal
    """
    if window:
        y = y * blackmanharris(len(y))
    halflen = int(len(y)/2)
    fvals, psd = periodogram(y, fs=1)
    return fvals[:halflen], psd[:halflen]

from scipy.ndimage import gaussian_filter1d
def find_characteristic_timescale(y, k=1, window=True):
    """
    Find the k leading characteristic timescales in a time series
    using the power spectrum..
    """
    y = gaussian_filter1d(y, 3)

    fvals, psd = find_psd(y, window=window)
    max_indices = np.argsort(psd)[::-1]
    
    # Merge adjacent peaks
    grouped_maxima = group_consecutives(max_indices)
    max_indices_grouped = np.array([np.mean(item) for item in grouped_maxima])
    max_indices_grouped = max_indices_grouped[max_indices_grouped != 1]
    
    return np.squeeze(1/(np.median(np.diff(fvals))*max_indices_grouped[:k]))


def parabolic(f, x):
    """
    Quadratic interpolation in order to estimate the location of a maximum
    https://gist.github.com/endolith/255291
    
    Args:
        f (ndarray): a vector a samples
        x (int): an index on the vector

    Returns:
        (vx, vy): the vertex coordinates of  a parabola passing through x 
            and its neighbors
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def parabolic_polyfit(f, x, n):
    """
    Use the built-in polyfit() function to find the peak of a parabola
    https://gist.github.com/endolith/255291
    
    Args:
        f (ndarray): a vector a samples
        x (int): an index on the vector
        n (int): the number of samples on the parabola
    """
    a, b, c = np.polyfit(np.arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
    xv = -0.5 * b/a
    yv = a * xv**2 + b * xv + c
    return (xv, yv)

def freq_from_autocorr(sig, fs=1):
    """
    Estimate frequency using autocorrelation
    
    Args:
        sig (ndarray): A univariate signal
        fs (int): The sampling frequency
        
    Returns:
        out (float): The dominant frequency
    
    References:
        Modified from the following
        https://gist.github.com/endolith/255291
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = np.correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    out = fs / px
    return out

from numpy.fft import rfft
def freq_from_fft(sig, fs=1):
    """
    Estimate frequency of a signal from the peak of the power spectrum
    
    Args:
        sig (ndarray): A univariate signal
        fs (int): The sampling frequency
        
    Returns:
        out (float): The dominant frequency
    
    References:
        Modified from the following
        https://gist.github.com/endolith/255291
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)


def resample_timepoints(model, ic, tpts, pts_per_period=100):
    """
    Given a differential equation, initial condition, and a set of 
    integration points, determine a new set of timepoints that
    scales to the periodicity of the model
    
    Args:
        model (callable): the right hand side of a set of ODEs
        ic (list): the initial conditions
        tpts (array): the timepoints over which to integrate
        pts_per_period (int): the number of timepoints to sample in
            each period
            
    Returns:
        new_timepoints (ndarray): The resampled timepoints
    """
    dt = (tpts[1] - tpts[0])
    samp = integrate_dyn(model, ic, tpts)[0]
    period = dt*(1/freq_from_autocorr(samp[:10000], 1))
    num_periods = len(tpts) // pts_per_period
    new_timepoints = np.linspace(0, num_periods*period, num_periods*pts_per_period)
    #out = integrate_dyn(eq, ic, tt)
    return new_timepoints

def make_surrogate(data, method="rp"):
    """
    
    Args:
        data (ndarray): A one-dimensional time series
        method (str): "rs" or rp"
        
    Returns:
        surr_data (ndarray): A single random surrogate time series
        
    Todo:
        Add ensemble function
    
    """
    if method == "rp":
        phases = np.angle(np.fft.fft(data))
        radii =  np.abs(np.fft.fft(data))
        random_phases = 2 * np.pi * (2*(np.random.random(phases.shape) - 0.5))
        surr_data = np.real(np.fft.ifft(radii * np.cos(random_phases) + 1j * radii * np.sin(random_phases)))
    else:
        surr_data = np.copy(data)
        np.random.shuffle(surr_data)
    return surr_data

def find_significant_frequencies(sig, window=True, fs=1, n_samples=100, 
                                 significance_threshold=0.95,
                                 surrogate_method="rp",
                                 show=False, return_amplitudes=False):
    """
    Find power spectral frequencies that are significant in a signal, by comparing
    the appearance of a peak with its appearance in randomly-shuffled surrogates.

    If no significant freqencies are detected, the significance floor is lowered
    
    Args:
        window (bool): Whether to window the signal before taking the FFT
        thresh (float): The number of standard deviations above mean to be significant
        fs (int): the sampling frequency
        n_samples (int): the number of surrogates to create
        show (bool): whether to show the psd of the signal and the surrogate
    
    Returns:
        freqs (ndarray): The frequencies overrated in the dataset
        amps (ndarray): the amplitudes of the PSD at the identified frequencies

    """
    n = len(sig)
    halflen = n // 2
    
    if window:
        sig = sig * blackmanharris(n)
    
    psd_sig = np.abs(rfft(sig))**2
    
    all_surr_psd = list()
    for i in range(n_samples):
        #surr = np.copy(sig)
        surr = make_surrogate(sig, method=surrogate_method)
        #np.random.shuffle(surr)
        if window:
            surr = surr * blackmanharris(len(surr))
            psd_surr = np.abs(rfft(surr))**2
        all_surr_psd.append(psd_surr)
    all_surr_psd = np.array(all_surr_psd)

#     thresh=1.0 
#     surrogate_floor = np.mean(all_surr_psd, axis=0) + thresh * np.std(all_surr_psd, axis=0)  
#     sel_inds = psd_sig > surrogate_floor
    
    frac_exceed = np.sum((psd_sig > all_surr_psd), axis=0)/all_surr_psd.shape[0]

    sel_inds = (frac_exceed >= significance_threshold)
    while len(sel_inds) == 0 and significance_threshold > 0:
        significance_threshold -= 0.01
        sel_inds = (frac_exceed >= significance_threshold)
    
    freq_inds = np.arange(len(psd_sig))[sel_inds]
    amps = psd_sig[sel_inds]
    freqs = fs * freq_inds / len(sig)
    
    ## cutoff low frequency components: half signal length times safety factor
    freq_floor = (1 / halflen) * 10 # safety factor
    amps = amps[freqs > freq_floor]
    freqs = freqs[freqs > freq_floor]
    
    if return_amplitudes:
        return freqs, amps
    else:
        return freqs


def generate_ic_ensemble(
    model,
    tpts0,
    n_samples,
    frac_perturb_param=0.1,
    frac_transient=0.1,
    ic_range=None,
    random_state=0,
):
    """
    Generate an ensemble of trajectories with random initial conditions, labelled by different
    initial conditions
    
    Args:
        model (callable_): function defining the numerical derivative
        tpts (ndarray): the timesteps over which to run the simulation
        n_samples (int): the number of different initial conditons
        frac_perturb_param (float): the amount to perturb the ic by
        frac_transient (float): the fraction of time for the time series to settle onto the attractor
        ic_range (list): a starting value for the initial conditions
        random_state (int): the seed for the random number generator
        
    Returns:
        all_samples (array): A set of initial conditions
    """
    np.random.seed(random_state)
    ntpts = len(tpts0)
    dt = tpts0[1] - tpts0[0]
    t_range = tpts0[-1] - tpts0[0]
    tpts = np.arange(tpts0[0], tpts0[0] + t_range * (1 + frac_transient), dt)
    all_samples = list()
    ic = model.ic
    for i in range(n_samples):
        ic_perturb = 1 + frac_perturb_param * (2 * np.random.random(len(ic)) - 1)
        ic_prime = ic * ic_perturb
        sol = integrate_dyn(model, ic_prime, tpts)
        all_samples.append(sol[:, -ntpts:])
    return np.array(all_samples)

from scipy.optimize import approx_fprime
def jac_fd(func0, y0, eps=1e-3, m=1, method="central", verbose=False):
    """
    Calculate numerical jacobian of a function with respect to a reference value
    
    Args:
        func (callable): a vector-valued function
        y0 (ndarray): a point around which to take the gradient
        eps (float): the step size for the finite difference calculation
        
    Returns:
        jac (ndarray): a numerical estimate of the Jacobian about that point
    
    """
    func = lambda x : np.array(func0(x)) # ensure an ndarray returned
    y0 = np.array(y0) # ensure an ndarray input
    
#     d = len(y0)
#     all_rows = list()
#     for i in range(d):
#         if m == 1:
#             y0p = np.copy(y0)
#             y0p[i] += eps / 2
#             y0m = np.copy(y0)
#             y0m[i] -= eps / 2
#             if verbose: print(y0m, y0p)
#             dval = (func(y0p) - func(y0m)) / eps
#         elif m == 2:
#             y0p = np.copy(y0)
#             y0p[i] += eps
#             y0m = np.copy(y0)
#             y0m[i] -= eps
#             dval = (func(y0p) + func(y0m) - 2 * func(y0)) / eps**2
#         all_rows.append(dval)
#     jac = np.array(all_rows).T

    d = len(y0)
    all_rows = list()
    for i in range(d):
        row_func = lambda yy : func(yy)[i]
        row = approx_fprime(y0, row_func, epsilon=eps)
        all_rows.append(row)
    jac = np.array(all_rows)

    return jac

    
def find_slope(x, y):
    """
    Given two vectors or arrays, compute the best fit slope using an analytic
    formula. For arrays is computed along the last axis.

    Args:
        x, y (ndarray): (N,) or (M, N)

    Returns:
        b (ndarray): the values of the slope for each of the last dimensions
    """
    n = x.shape[-1]
    b = n * (x * y).sum(axis=-1) - x.sum(axis=-1) * y.sum(axis=-1)
    b /= n * (x * x).sum(axis=-1) - x.sum(axis=-1) * x.sum(axis=-1)
    return b


def make_epsilon_ball(pt, n, eps=1e-5, random_state=None):
    """
    Uniformly sample a fixed-radius ball of points around a given point via
    using Muller's method
    
    Args:
        pt (ndarray): The center of the sampling
        n (int): The number of points to sample
        eps (float): The radius of the ball
        random_state (int): Initialize the random number generator
        
    Returns:
        out (ndarray): The set of randomly-sampled points
    """
    np.random.seed(None)
    pt = np.squeeze(np.array(pt))
    d = len(pt)
    vecs = np.random.normal(0, 1, size=(d, n))
    r = np.random.random(n)**(1./d)
    norm = np.linalg.norm(vecs, axis=0)
    coords = r * vecs / norm
    out = pt[:, None] + eps * coords
    return out

import threading
class ComputationHolder:
    """
    A wrapper class to force a computation to stop after a timeout.

    Parameters
        func (callable): the function to run
        args (tuple): the arguments to pass to the function
        kwargs (dict): the keyword arguments to pass to the function
        timeout (int): the timeout in seconds. If None is passed, the computation
            will run indefinitely until it finishes.

    Example
        >>> def my_func():
        ...     while True:
        ...         print("hello")
        ...         time.sleep(8)
        >>> ch = ComputationHolder(my_func, timeout=3)
        >>> ch.run()
        hello
        hello
        hello
        None

    """

    def __init__(self, func=None, *args, timeout=10, **kwargs):
        self.sol = None
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

        def func_wrapped():
            self.sol = self.func(*self.args, **self.kwargs)

        self.func_wrapped = func_wrapped

    def run(self):
        my_thread = threading.Thread(
            target=self.func_wrapped
        )
        my_thread.start()
        my_thread.join(self.timeout) # kill the thread after `timeout` seconds

        if self.sol is None:
            return None
        else:
            return self.sol
        
        