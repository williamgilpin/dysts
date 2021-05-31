
# try:
#     import jax.numpy as np
#     from jax import jit
#     has_jax = True
# except:
#     import numpy as np
#     has_jax = False
# from functools import partial

# try:
#     import jax.numpy as np
#     has_jax = True
# except ModuleNotFoundError:
#     import numpy as np
#     has_jax = False
#     warnings.warn("JAX not found, falling back to numpy.")
import numpy as np
from numpy.fft import rfft, irfft

import warnings
from scipy.integrate import odeint, solve_ivp
from scipy.signal import blackmanharris, fftconvolve
from collections import deque
from functools import partial

import json

from sdeint import itoint


import pkg_resources

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

def standardize_ts(a, scale=1.0):
    """Standardize a T x D time series along its first dimension
    For dimensions with zero variance, divide by one instead of zero
    """
    stds = np.std(a, axis=0, keepdims=True)
    stds[stds==0] = 1
    return (a - np.mean(a, axis=0, keepdims=True))/(scale*stds)

def integrate_dyn(f, ic, tvals, noise=0, **kwargs):
    """
    Given the RHS of a dynamical system, integrate the system
    noise > 0 requires the Python library sdeint (assumes Brownian noise)
    
    Args:
        f (callable): The right hand side of a system of ODEs.
        ic (ndarray): the initial conditions
        noise_amp (float): The amplitude of the Langevin forcing term.
        kwargs (dict): Arguments passed to scipy.integrate.solve_ivp.
    """
    ic = np.array(ic)
    if noise > 0:
        gw = lambda y, t: noise * np.diag(ic)
        fw = lambda y, t: np.array(f(y, t))
        sol = itoint(fw, gw, np.array(ic), tvals).T
    else:
        #dt = np.median(np.diff(tvals))
        fc = lambda t, y : f(y, t)
        sol0 = solve_ivp(fc, [tvals[0], tvals[-1]], ic, t_eval=tvals, **kwargs)
        sol = sol0.y
        #sol = odeint(f, np.array(ic), tvals).T

    return sol

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

from scipy.signal import periodogram
def group_consecutives(vals, step=1):
    """
    Return list of consecutive lists of numbers from vals (number list).
    Adapted from here
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
    # y = y * blackmanharris(len(y))
    # halflen = int(len(y)/2)
    # fvals, psd = periodogram(y, fs=1)
    max_indices = np.argsort(psd)[::-1]
    
    # Merge adjacent peaks
    grouped_maxima = group_consecutives(max_indices)
    max_indices_grouped = np.array([np.mean(item) for item in grouped_maxima])
    
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
            and its nneighbors
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
    https://gist.github.com/endolith/255291
    
    sig : a univariate signal
    fs : the sampling frequency
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

    return fs / px

from numpy.fft import rfft
def freq_from_fft(sig, fs=1):
    """
    Estimate frequency from peak of FFT
    https://gist.github.com/endolith/255291
    
    sig (array): a univariate signal
    fs (int): the sampling frequency
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
    
    model (callable): the right hand side of a set of ODEs
    ic (list): the initial conditions
    tpts (array): the timepoints over which to integrate
    pts_per_period (int): the number of timepoints to sample in
        each period
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
    
    Parameters
    ----------
    method (str): "rs" or rp"
    
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
    the appearance of a peak with its appearance in randomly-shuffled surrogates
    
    Parameters
    ----------
    window : bool
        whether to window the signal before taking the FFT
    thresh : float
        the number of standard deviations above mean to be significant
    fs (int): the sampling frequency
    n_samples (int): the number of surrogates to create
    show (bool): whether to show the psd of the signal and the surrogate
    
    Returns
    -------
    freqs (ndarray): The frequencies overrated in the dataset
    amps (ndarray): the amplitudes of the PSD at the identified frequencies

    """
    n = len(sig)
    halflen = n // 2
    
    if window:
        sig = sig * blackmanharris(n)
    
    psd_sig = rfft(sig)
    
    all_surr_psd = list()
    for i in range(n_samples):
        #surr = np.copy(sig)
        surr = make_surrogate(sig, method=surrogate_method)
        #np.random.shuffle(surr)
        if window:
            surr = surr * blackmanharris(len(surr))
            psd_surr = rfft(surr)
        all_surr_psd.append(psd_surr)
    all_surr_psd = np.array(all_surr_psd)

#     thresh=1.0 
#     surrogate_floor = np.mean(all_surr_psd, axis=0) + thresh * np.std(all_surr_psd, axis=0)  
#     sel_inds = psd_sig > surrogate_floor
    
    frac_exceed = np.sum((psd_sig > all_surr_psd), axis=0)/all_surr_psd.shape[0]
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
        model : function defining the numerical derivative
        tpts (ndarray): the timesteps over which to run the simulation
        n_samples (int): the number of different initial conditons
        frac_perturb_param : float, the amount to perturb the ic by
        frac_transient (float): the fraction of time for the time series to settle onto the attractor
        ic_range : a starting value for the initial conditions
        random_state (int): the seed for the random number generator
    """
    np.random.seed(random_state)
    ntpts = len(tpts0)
    dt = tpts0[1] - tpts0[0]
    t_range = tpts0[-1] - tpts0[0]
    tpts = np.arange(tpts0[0], tpts0[0] + t_range * (1 + frac_transient), dt)
    all_samples = list()
    for i in range(n_samples):
        ic_perturb = 1 + frac_perturb_param * (2 * np.random.random(len(ic)) - 1)
        ic_prime = ic * ic_perturb
        sol = integrate_dyn(model, ic_prime, tpts)
        all_samples.append(sol[:, -ntpts:])
    return np.array(all_samples)


def jac_fd(func, y0, eps=1e-8):
    """
    Calculate numerical jacobian of a function with respect to a reference value
    
    Args:
        func (callable): a vector-valued function
        y0 (ndarray): a point around which to take the gradient
        eps (float): the step size for the finite difference calculation
        
    Returns:
        jac (ndarray): a numerical estimate of the Jacobian about that point
    
    """
    d = len(y0)
    all_rows = list()
    for i in range(d):
        y0p = np.copy(y0)
        y0p[i] += eps
        y0m = np.copy(y0)
        y0m[i] += eps
        dval = 0.5 * (func(y0p) - func(y0)) / 1e-8 + 0.5 * (func(y0m) - func(y0)) / 1e-8
        all_rows.append(dval)
    jac = np.array(all_rows).T
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


# def generate_lorenz_ensemble(tpts0, n_samples, params, frac_perturb_param=.1, 
#                              n_classes=2, frac_transient=0.1, 
#                              ic_range=None,
#                             random_state=0):
#     """
#     Generate an ensemble of trajectories with random initial conditions, labelled by different
#     sets of parameters.
    
#     tpts : the timesteps over which to run the simulation
#     params : iterable, the starting values for the parameters
#     n_samples : int, the number of different initial conditons
#     n_classes : int , the number of different parameters
#     frac_perturb_param : float, the amount to perturb the parameters by
#     frac_transient : float, the fraction of time for the time series to settle onto the attractor
#     ic_range : a starting value for the initial conditions
#     random_state : int, the seed for the random number generator
#     """
#     np.random.seed(random_state)
    
#     ntpts = len(tpts0)
#     dt = tpts0[1] - tpts0[0]
#     t_range = tpts0[-1] - tpts0[0]
#     tpts = np.arange(tpts0[0], tpts0[0] + t_range*(1 + frac_transient), dt)
    
#     num_per_class = int(n_samples/n_classes)

#     all_params = list()
#     all_samples = list()
#     for i in range(n_classes):
    
#         params_perturb = 1 + frac_perturb_param*(2*np.random.random(len(params)) - 1)
#         params_prime = params*params_perturb
#         all_params.append(params_prime)
    
#         eq = Lorenz(*params_prime)
        
#         all_samples_per_class = list()
#         for j in range(num_per_class):
#             ic_prime = (-8.60632853, -14.85273055,  15.53352487)*np.random.random(3)
#             sol = integrate_dyn(eq, ic_prime, tpts)
            
#             all_samples_per_class.append(sol[:, -ntpts:]) # remove transient
#         all_samples.append(all_samples_per_class)
    
#     all_samples, all_params = np.array(all_samples), np.array(all_params)
#     return all_samples, all_params
    
# num_samples = 120

# data, labels = generate_lorenz_ensemble(np.linspace(0, 500, 125000), 2*num_samples, (10, 28, 2.5), 
#                                     n_classes=8, frac_perturb_param=.2, frac_transient=.2)
