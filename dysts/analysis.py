"""

Functions that act on DynSys or DynMap objects

"""

import numpy as np
import warnings

try:
    import neurokit2 # Used for computing multiscale entropy
    has_neurokit = True
except:
    warnings.warn("Neurokit2 must be installed before computing multiscale entropy")
    has_neurokit = False

from .utils import *
from .utils import standardize_ts
from .utils import ComputationHolder


def sample_initial_conditions(
    model, points_to_sample, traj_length=1000, pts_per_period=30
):
    """
    Generate a random sample of initial conditions from a dynamical system

    Args:
        model (callable): the right hand side of a differential equation, in format func(X, t)
        points_to_sample (int): the number of random initial conditions to sample
        traj_length (int): the total length of the reference trajectory from which points are drawn
        pts_per_period (int): the sampling density of the trajectory

    Returns:
        sample_points (ndarray): The points with shape (points_to_sample, d)

    """
    initial_sol = model.make_trajectory(
        traj_length, resample=True, pts_per_period=pts_per_period, postprocess=False
    )
    sample_inds = np.random.choice(
        np.arange(initial_sol.shape[0]), points_to_sample, replace=False
    )
    sample_pts = initial_sol[sample_inds]
    return sample_pts


def compute_timestep(
    model,
    total_length=40000,
    transient_fraction=0.2,
    num_iters=20,
    pts_per_period=1000,
    visualize=False,
    return_period=True,
):
    """Given a dynamical system object, find the integration timestep based on the largest
    signficant frequency

    Args:
        model (DynSys): A dynamical systems object.
        total_length (int): The total trajectory length to use to determine timescales.
        transient_fraction (float): The fraction of a trajectory to discard as transient
        num_iters (int): The number of refinements to the timestep
        pts_per_period (int): The target integration timestep relative to the signal.
        visualize (bool): Whether to plot timestep versus time, in order to identify problems
            with the procedure
        return_period (bool): Whether to calculate and retunr the dominant timescale in the signal

    Returns
        dt (float): The best integration timestep
        period (float, optional): The dominant timescale in the signal

    """

    base_freq = 1 / pts_per_period
    cutoff = int(transient_fraction * total_length)

    step_history = [np.copy(model.dt)]
    for i in range(num_iters):
        sol = model.make_trajectory(total_length, standardize=True)[cutoff:]
        all_freqs = list()
        for comp in sol.T:
            try:
                all_freqs = find_significant_frequencies(comp, surrogate_method="rs")
                all_freqs.append(np.percentile(all_freqs, 98))
                # all_freqs.append(np.max(all_freqs))
            except:
                pass
        freq = np.median(all_freqs)
        period = base_freq / freq
        model.dt = model.dt * period

        step_history.append(model.dt)
        if i % 5 == 0:
            print(f"Completed step {i} of {num_iters}")
    dt = model.dt

    if visualize:
        plt.plot(step_history)

    if return_period:
        sol = model.make_trajectory(total_length, standardize=True)[cutoff:]
        all_freqs = list()
        for comp in sol.T:
            try:
                freqs, amps = find_significant_frequencies(
                    comp, surrogate_method="rs", return_amplitudes=True
                )
                all_freqs.append(freqs[np.argmax(np.abs(amps))])
            except:
                pass
        freq = np.median(all_freqs)
        period = model.dt * (1 / freq)
        return dt, period
    else:
        return dt
    

from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit


def estimate_powerlaw(data0):
    """
    Given a 1D array of continuous-valued data, estimate the power law exponent using the 
    maximum likelihood estimator proposed by Clauset, Shalizi, Newman (2009).
    
    Args:
        data0 (np.ndarray): An array of continuous-valued data

    Returns:
        float: The estimated power law exponent
    """
    data = np.sort(data0, axis=0).copy()
    xmin = np.min(data, axis=0)
    n = data.shape[0]
    ahat = 1 + n / np.sum(np.log(data / xmin), axis=0)
    return ahat

def gp_dim(data, y_data=None, rvals=None, nmax=100):
    """
    Estimate the Grassberger-Procaccia dimension for a numpy array using the 
    empirical correlation integral.

    Args:
        data (np.array): T x D, where T is the number of datapoints/timepoints, and D
            is the number of features/dimensions
        y_data (np.array, Optional): A second dataset of shape T2 x D, for 
            computing cross-correlation.
        rvals (np.array): A list of radii
        nmax (int): The number of points at which to evaluate the correlation integral

    Returns:
        rvals (np.array): The discrete bins at which the correlation integral is 
            estimated
        corr_sum (np.array): The estimates of the correlation integral at each bin

    TODO:
        power law fit
        Check Clauset paper
        Implement robust least squares linear fitting

    For time series or spatial aurocorrelation, what is a standard scaled distance
    based on cross-correlation versus individual autocorrections?

    <ox oy> / ox oy

    """

    data = np.asarray(data)
    # data = embed(data)

    ## For self-correlation
    if y_data is None:
        y_data = data.copy()

    if rvals is None:
        std = np.std(data)
        rvals = np.logspace(np.log10(0.1 * std), np.log10(0.5 * std), nmax)

    n = len(data)
    
    # dists = cdist(data, y_data)
    # corr_sum = []
    # for r in rvals:
    #     corr_sum.append(np.sum(dists < r))
    # corr_sum = np.array(corr_sum) / (n * (n - 1))

    # dists = cdist(data, y_data)
    # hist, _ = np.histogram(dists, bins=np.hstack([0, rvals])) # can we skip this and direct fit?
    # corr_sum = np.cumsum(hist).astype(float)
    # corr_sum /= n * (n - 1)

    dists = cdist(data, y_data)
    rvals = np.sort(dists.ravel())
    corr_sum = np.arange(len(rvals)).astype(float)
    corr_sum /= n * (n - 1)
    std = np.std(data)
    sel_inds = rvals > 0.1 * std
    rvals = rvals[sel_inds]
    corr_sum = corr_sum[sel_inds]
    sel_inds = rvals < 0.5 * std
    rvals = rvals[sel_inds]
    corr_sum = corr_sum[sel_inds]

    # corr_sum = count_pairs_within_thresholds(data, rvals).astype(float)
    # corr_sum /= n * (n - 1)

    ## Drop zeros before regression
    sel_inds = corr_sum > 0
    rvals = rvals[sel_inds]
    corr_sum = corr_sum[sel_inds]
    
    # poly = np.polyfit(np.log(rvals), np.log(corr_sum), 1)
    # return poly[0]

    power_law = lambda x, a, b: a * (x ** b)
    fit_vals = curve_fit(power_law, rvals, corr_sum)
    return fit_vals[0][1]
    
def corr_gpdim(traj1, traj2, register=False, standardize=False, **kwargs):
    """
    Given two multivariate time series, estimate their similarity using the cross
    Grassberger-Procaccia dimension

    Args:
        traj1 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        traj2 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        register (bool): Whether to register the two time series before computing the
            cross-correlation
        standardize (bool): Whether to standardize the time series before computing the
            cross-correlation

    Returns:
        float: The cross-correlation between the two time series
    """
    if register:
        model = RidgeCV()
        model.fit(traj1, traj2)
        traj1 = model.predict(traj1)

    if standardize:
        traj1 = (traj1 - np.mean(traj1, axis=0)) / np.std(traj1, axis=0)
        traj2 = (traj2 - np.mean(traj2, axis=0)) / np.std(traj2, axis=0)

    return gp_dim(traj1, traj2, **kwargs) / np.sqrt(gp_dim(traj1, **kwargs) * gp_dim(traj2, **kwargs))

from sklearn.linear_model import RidgeCV
def gpdistance(traj1, traj2, standardize=True, register=False, **kwargs):
    """
    Given two multivariate time series, estimate their similarity using the cross
    Grassberger-Procaccia distance

    Args:
        traj1 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        traj2 (np.array): T x D, where T is the number of timepoints, and D
            is the number of dimensions
        register (bool): Whether to register the two time series before computing the
            cross-correlation
        standardize (bool): Whether to standardize the time series before computing the
            cross-correlation
    """

    if register:
        model = RidgeCV()
        model.fit(traj1, traj2)
        traj1 = model.predict(traj1)

    if standardize:
        traj1 = (traj1 - np.mean(traj1, axis=0)) / np.std(traj1, axis=0)
        traj2 = (traj2 - np.mean(traj2, axis=0)) / np.std(traj2, axis=0)
        
    return np.abs(np.log(corr_gpdim(traj1, traj2, **kwargs)))



def find_lyapunov_exponents(
    model, traj_length, pts_per_period=500, tol=1e-8, min_tpts=10, **kwargs
):
    """
    Given a dynamical system, compute its spectrum of Lyapunov exponents.
    Args:
        model (callable): the right hand side of a differential equation, in format 
            func(X, t)
        traj_length (int): the length of each trajectory used to calulate Lyapunov
            exponents
        pts_per_period (int): the sampling density of the trajectory
        kwargs: additional keyword arguments to pass to the model's make_trajectory 
            method

    Returns:
        final_lyap (ndarray): A list of computed Lyapunov exponents

    References:
        Christiansen & Rugh (1997). Computing Lyapunov spectra with continuous
            Gram-Schmidt orthonormalization

    Example:
        >>> import dysts
        >>> model = dysts.Lorenz()
        >>> lyap = dysts.find_lyapunov_exponents(model, 1000, pts_per_period=1000)
        >>> print(lyap)

    """
    d = np.asarray(model.ic).shape[-1]
    tpts, traj = model.make_trajectory(
        traj_length, pts_per_period=pts_per_period, resample=True, return_times=True,
        postprocessing=False,
        **kwargs
    )
    dt = np.median(np.diff(tpts))
    # traj has shape (traj_length, d), where d is the dimension of the system
    # tpts has shape (traj_length,)
    # dt is the dimension of the system

    u = np.identity(d)
    all_lyap = list()
    # for i in range(traj_length):
    for i, (t, X) in enumerate(zip(tpts, traj)):
        X = traj[i]

        if model.jac(model.ic, 0) is None:
            rhsy = lambda x: np.array(model.rhs(x, t))
            jacval = jac_fd(rhsy, X)
        else:
            jacval = np.array(model.jac(X, t))

        # If postprocessing is applied to a trajectory, transform the jacobian into the
        # new coordinates.
        if hasattr(model, "_postprocessing"):
            X0 = np.copy(X)
            y2h = lambda y: model._postprocessing(*y)
            dhdy = jac_fd(y2h, X0)
            dydh = np.linalg.inv(dhdy)  # dy/dh
            ## Alternate version if good second-order fd is ever available
            # dydh = jac_fd(y2h, X0, m=2, eps=1e-2) @ rhsy(X0) + jac_fd(y2h, y0) @ jac_fd(rhsy, X0))
            jacval = dhdy @ jacval @ dydh

        ## Forward Euler update
        # u_n = np.matmul(np.eye(d) + jacval * dt, u)
        
        ## Backward Euler update
        if i < 1: continue
        u_n = np.matmul(np.linalg.inv(np.eye(d) - jacval * dt), u)
        
        q, r = np.linalg.qr(u_n)
        lyap_estimate = np.log(abs(r.diagonal()))
        all_lyap.append(lyap_estimate)
        u = q  # post-iteration update axes

        ## early stopping if middle exponents are close to zero, a requirement for
        ## continuous-time dynamical systems
        if (np.min(np.abs(lyap_estimate)) < tol) and (i > min_tpts):
            traj_length = i

    all_lyap = np.array(all_lyap)
    final_lyap = np.sum(all_lyap, axis=0) / (dt * traj_length)
    return np.sort(final_lyap)[::-1]

from scipy.stats import linregress
def calculate_lyapunov_exponent(traj1, traj2, dt=1.0):
    """
    Calculate the lyapunov exponent of two multidimensional trajectories using
    simple linear regression based on the log-transformed separation of the
    trajectories.

    Args:
        traj1 (np.ndarray): trajectory 1 with shape (n_timesteps, n_dimensions)
        traj2 (np.ndarray): trajectory 2 with shape (n_timesteps, n_dimensions)
        dt (float): time step between timesteps

    Returns:
        float: lyapunov exponent
    """
    separation = np.linalg.norm(traj1 - traj2, axis=1)
    log_separation = np.log(separation)
    time_vals = np.arange(log_separation.shape[0])
    slope, intercept, r_value, p_value, std_err = linregress(time_vals, log_separation)
    lyap = slope / dt
    return lyap

def lyapunov_exponent_naive(
    eq, rtol=1e-3, atol=1e-10, n_samples=1000, traj_length=5000, max_walltime=None,
    **kwargs
    ):
    """
    Calculate the lyapunov spectrum of the system using a naive method based on the
    log-transformed separation of the trajectories over time.

    Args:
        eq (dysts.DynSys): equation to calculate the lyapunov spectrum of
        rtol (float): relative tolerance for the separation of the trajectories
        atol (float): absolute tolerance for the separation of the trajectories
        n_samples (int): number of initial conditions to sample
        traj_length (int): length of the trajectories to sample. This should be long
            enough to ensure that most trajectories explore the attractor.
        max_walltime (float): maximum walltime in seconds to spend on the calculation
            of a given trajectory. If the calculation takes longer than this, the
            trajectory is discarded and a new one is sampled.
        **kwargs: keyword arguments to pass to the sample / make_trajectory method of
             the dynamical equation

    Returns:
        float: largest lyapunov exponent

    Example:
        >>> import dysts
        >>> eq = dysts.Lorenz()
        >>> max_lyap = dysts.lyapunov_exponent_naive(eq)
        
    """
    all_ic = sample_initial_conditions(
        eq, 
        n_samples, 
        traj_length=max(traj_length, n_samples),
        pts_per_period=15,
    )
    pts_per_period = 100
    eps = atol
    eps_max = rtol
    all_lyap = []
    all_cutoffs = []
    for ind, ic in enumerate(all_ic):
        np.random.seed(ind)
        eq.random_state = ind
        eq.ic = ic
        out = ComputationHolder(
            eq.make_trajectory, 
            traj_length, 
            timeout=max_walltime,
            resample=True, 
            return_times=True,
            **kwargs
        ).run()
        if out is None:
            continue
        if np.sum(np.isnan(out[1])) > 0:
            continue
        else:
            tvals, traj1 = out

        np.random.seed(ind)
        eq.random_state = ind
        eq.ic = ic
        eq.ic *= (1 + eps * (np.random.random(eq.ic.shape) - 0.5))
        # traj2 = eq.sample(traj_length, resample=True, **kwargs)
        traj2 = ComputationHolder(
            eq.make_trajectory, 
            traj_length,
            timeout=max_walltime, 
            resample=True, 
            **kwargs
        ).run()
        if traj2 is None:
            continue
        if np.sum(np.isnan(traj2)) > 0:
            continue

        ## Truncate traj1 and traj2 to when their scaled separation is less than eps_max
        separation = np.linalg.norm(traj1 - traj2, axis=1) / np.linalg.norm(traj1, axis=1)
        cutoff_index = np.where(separation < eps_max)[0][-1]
        all_cutoffs.append(cutoff_index)
        traj1 = traj1[:cutoff_index]
        traj2 = traj2[:cutoff_index]
        lyap = calculate_lyapunov_exponent(traj1, traj2, dt=np.median(np.diff(tvals)))
        all_lyap.append(lyap)

    ## Return None if no trajectories were successful
    if len(all_lyap) == 0:
        return None

    if len(all_lyap) < int(0.6 * n_samples):
        warnings.warn(
            "The number of successful trajectories is less than 60% of the total number " \
            + "of trajectories attempted. This may indicate that the integration " \
            + "is unstable"
        )

    if np.median(all_cutoffs) < pts_per_period:
        warnings.warn(
            "The median cutoff index is less than the number of points per period. " \
            + "This may indicate that the integration is not long enough to capture " \
            + "the invariant properties."
        )

    return np.mean(all_lyap)

def kaplan_yorke_dimension(spectrum0):
    """Calculate the Kaplan-Yorke dimension, given a list of
    Lyapunov exponents"""
    spectrum = np.sort(spectrum0)[::-1]
    d = len(spectrum)
    cspec = np.cumsum(spectrum)
    j = np.max(np.where(cspec >= 0))
    if j > d - 2:
        j = d - 2
        warnings.warn(
            "Cumulative sum of Lyapunov exponents never crosses zero. System may be ill-posed or undersampled."
        )
    dky = 1 + j + cspec[j] / np.abs(spectrum[j + 1])

    return dky




def mse_mv(traj):
    """
    Generate an estimate of the multivariate multiscale entropy. The current version 
    computes the entropy separately for each channel and then averages. It therefore 
    represents an upper-bound on the true multivariate multiscale entropy

    Args:
        traj (ndarray): a trajectory of shape (n_timesteps, n_channels)

    Returns:
        mmse (float): the multivariate multiscale entropy

    TODO:
        Implement algorithm from Ahmed and Mandic PRE 2011
    """

    if not has_neurokit:
        raise Exception("NeuroKit not installed; multiscale entropy cannot be computed.")

    #mmse_opts = {"composite": True, "refined": False, "fuzzy": True}
    mmse_opts = {"composite": True, "fuzzy": True}
    if len(traj.shape) == 1:
        mmse = neurokit2.entropy_multiscale(sol, dimension=2, **mmse_opts)[0]
        return mmse

    traj = standardize_ts(traj)
    all_mse = list()
    for sol_coord in traj.T:
        all_mse.append(
            neurokit2.entropy_multiscale(sol_coord, dimension=2, **mmse_opts)[0]
        )
    return np.median(all_mse)



def get_train_test(eq, n_train=1000, n_test=200, standardize=True, **kwargs):
    """
    Generate train and test trajectories for a given dynamical system

    Args:
        eq (dysts.DynSys): a dynamical system object
        n_train (int): number of points in the training trajectory
        n_test (int): number of points in the test trajectory
        standardize (bool): whether to standardize the trajectories
        **kwargs: additional keyword arguments to pass to make_trajectory

    Returns:
        (tuple): a tuple containing:
            (tuple): a tuple containing:
                (ndarray): the timepoints of the training trajectory
                (ndarray): the training trajectory
            (tuple): a tuple containing:
                (ndarray): the timepoints of the test trajectory
                (ndarray): the test trajectory
                
    """
    train_ic, test_ic = sample_initial_conditions(eq, 2)

    eq.ic = train_ic
    tpts_train, sol_train = eq.make_trajectory(
        n_train, resample=True, return_times=True, **kwargs
    )
    eq.ic = test_ic
    tpts_test, sol_test = eq.make_trajectory(
        n_test, resample=True, return_times=True, **kwargs
    )
    
    if standardize:
        center = np.mean(sol_train, axis=0)
        scale = np.std(sol_train, axis=0)
        sol_train = (sol_train - center) / scale
        sol_test = (sol_test - center) / scale
    
    return (tpts_train, sol_train), (tpts_test, sol_test)

