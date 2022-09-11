"""

Functions that act on DynSys or DynMap objects

"""

import numpy as np

try:
    import neurokit2 # Used for computing multiscale entropy
except:
    warnings.warn("Neurokit2 must be installed before computing multiscale entropy")
    has_neurokit = False

from .utils import *
from .utils import standardize_ts



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


def find_lyapunov_exponents(
    model, traj_length, pts_per_period=500, tol=1e-8, min_tpts=10
):
    """
    Given a dynamical system, compute its spectrum of Lyapunov exponents

    Args:
        model (callable): the right hand side of a differential equation, in format func(X, t)
        traj_length (int): the length of each trajectory used to calulate Lyapunov
            exponents
        pts_per_period (int): the sampling density of the trajectory

    Returns:
        final_lyap (ndarray): A list of computed Lyapunov exponents
        
    References:
        Christiansen & Rugh (1997). Computing Lyapunov spectra with continuous 
            Gram-Schmidt orthonormalization


    """
    d = np.asarray(model.ic).shape[-1]
    tpts, traj = model.make_trajectory(
        traj_length, pts_per_period=pts_per_period, resample=True, return_times=True
    )
    dt = np.median(np.diff(tpts))

    u = np.identity(d)
    all_lyap = list()
    for i in range(traj_length):
        yval = traj[i]
        # rhsy = lambda x: np.array(model.rhs(x, tpts[i]))

        rhsy = lambda x: np.array(model.rhs(x, tpts[i]))
        jacval = jac_fd(rhsy, yval)

        # If postprocessing is applied to a trajectory,
        # boost the jacobian to the new coordinates
        if hasattr(model, "_postprocessing"):
            #             print("using different formulation")
            y0 = np.copy(yval)
            y2h = lambda y: model._postprocessing(*y)

            rhsh = lambda y: jac_fd(y2h, y) @ rhsy(y)  # dh/dy * dy/dt = dh/dt
            dydh = np.linalg.inv(jac_fd(y2h, y0))  # dy/dh
            ## Alternate version if good second-order fd is ever available
            # dydh = jac_fd(y2h, y0, m=2, eps=1e-2) @ rhsy(y0) + jac_fd(y2h, y0) @ jac_fd(rhsy, y0))

            jacval = jac_fd(rhsh, y0, eps=1e-3) @ dydh

        u_n = np.matmul(np.identity(d) + jacval * dt, u)
        q, r = np.linalg.qr(u_n)
        lyap_estimate = np.log(abs(r.diagonal()))
        all_lyap.append(lyap_estimate)
        u = q  # post-iteration update axes

        #         ## early stopping
        if (np.min(np.abs(lyap_estimate)) < tol) and (i > min_tpts):
            traj_length = i
            print("stopped early.")

    all_lyap = np.array(all_lyap)
    final_lyap = np.sum(all_lyap, axis=0) / (dt * traj_length)
    return np.sort(final_lyap)[::-1]


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
    Generate an estimate of the multivariate multiscale entropy. The current version computes 
    the entropy separately for each channel and then averages. It therefore represents an upper bound on the true 
    multivariate multiscale entropy
    
    Todo: 
        Implement algorithm from Ahmed and Mandic PRE 2011
    """

    if not has_neurokit:
        raise Exception("NeuroKit not installed; multiscale entropy cannot be computed.")

    mmse_opts = {"composite": True, "refined": False, "fuzzy": True}
    if len(traj.shape) == 1:
        mmse = neurokit2.complexity.entropy_multiscale(sol, dimension=2, **mmse_opts)
        return mmse

    traj = standardize_ts(traj)
    all_mse = list()
    for sol_coord in traj.T:
        all_mse.append(
            neurokit2.complexity.entropy_multiscale(sol_coord, dimension=2, **mmse_opts)
        )
    return np.median(all_mse)
