
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


import warnings
from scipy.integrate import odeint
from collections import deque
from functools import partial

from sdeint import itoint

## Numba support removed until jitclass API stabilizes
# try:
#     from numba import jit, njit, jitclass
#     has_numba = True
# except ModuleNotFoundError:
#     warnings.warn("Numba installation not found, numerical integration will run slower.")
#     has_numba = False
#     # Define placeholder functions
#     def jit(func):
#         return func
#     def njit(func):
#         return func
#     def autojit(func):
#         return func
#     def jitclass(c):
#         return c

def standardize_ts(a, scale=1.0):
    """
    Standardize a T x D time series along its first dimension
    For dimensions with zero variance, divide by one instead of zero
    """
    stds = np.std(a, axis=0, keepdims=True)
    stds[stds==0] = 1
    return (a - np.mean(a, axis=0, keepdims=True))/(scale*stds)

def integrate_dyn(f, ic, tvals, noise=0, use_compile=True):
    """
    Given the RHS of a dynamical system, integrate the system
    noise > 0 requires the Python library sdeint (assumes Brownian noise)

    f : callable, the right hand side of a system of ODE
    ic : the initial conditions
    noise_amp : the amplitude of the Langevin forcing term
    use_compile : bool, whether to compile the function with numba 
        before performing integration
    
    DEV:
    scipy.integrate.solve_ivp(eq, (tpts[0], tpts[-1]), np.array(ic), 
    method='DOP853', dense_output=True)
    eq takes (t, X) and not vice-versa
    """
#     try:
#         fc = jit(f.__call__)
#     except:
#         warnings.warn("Unable to compile function.")
#         fc = f
    fc = f # jit currently doesn't play well with objects
    if noise > 0:

        def gw(y, t):
            return noise * np.diag(ic)

        def fw(y, t):
            return np.array(fc(y, t))

        sol = itoint(fw, gw, np.array(ic), tvals).T
    else:
        sol = odeint(fc, np.array(ic), tvals).T

    return sol

def integrate_weiner(f, noise_amp, ic, tvals):
    """
    Given the RHS of a dynamical system, integrate the 
    system assuming Brownian noise
    Requires the Python library sdeint

    f : the right hand side of a system of ODE
    noise_amp : the amplitude of the Langevin forcing term
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

def find_characteristic_timescale(y, k=1):
    """
    Find the k leading characteristic timescales in a time series
    using the power spectrum
    """
    halflen = int(len(y)/2)
    fvals, psd = periodogram(y, fs=1)
    max_indices = np.argsort(psd[:halflen])[::-1]
    
    # Merge adjacent peaks
    grouped_maxima = group_consecutives(max_indices)
    max_indices_grouped = np.array([np.mean(item) for item in grouped_maxima])
    
    return np.squeeze(1/(np.median(np.diff(fvals))*max_indices_grouped[:k]))


def find_initial_condtion():
    """
    Simulate a system until it settles onto an attractor
    """
    pass



from numpy.fft import rfft, irfft
from scipy.signal import blackmanharris, fftconvolve

def parabolic(f, x):
    """
    Quadratic interpolation in order to estimate the location of a maximum

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

    Args:
        f (ndarray): a vector a samples
        x (int): an index on the vector
        n (int): the number of samples on the parabola
    """
    a, b, c = np.polyfit(np.arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
    xv = -0.5 * b/a
    yv = a * xv**2 + b * xv + c
    return (xv, yv)

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
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

def resample_timepoints(model, ic, tpts, pts_per_period=100):
    """
    Given a differential equation, initial condition, and a set of 
    integration points, determine a new set of timepoints that
    scales to the periodicity of the model
    
    - model : callable, the right hand side of a set of ODEs
    - ic : list, the initial conditions
    - tpts : array, the timepoints over which to integrate
    - pts_per_period : int, the number of timepoints to sample in
        each period
    """
    dt = (tpts[1] - tpts[0])
    samp = integrate_dyn(model, ic, tpts)[0]
    period = dt*(1/freq_from_autocorr(samp[:10000], 1))
    num_periods = len(tpts)/pts_per_period
    new_timepoints = np.linspace(0, num_periods*period, num_periods*pts_per_period)
    #out = integrate_dyn(eq, ic, tt)
    return new_timepoints

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
    
    Parameters
    - model : function defining the numerical derivative
    - tpts : the timesteps over which to run the simulation
    - n_samples : int, the number of different initial conditons
    - frac_perturb_param : float, the amount to perturb the ic by
    - frac_transient : float, the fraction of time for the time series to settle onto the attractor
    - ic_range : a starting value for the initial conditions
    - random_state : int, the seed for the random number generator
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