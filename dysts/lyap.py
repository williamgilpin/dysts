"""
This file is directly adapted from the nolds Python library:
https://github.com/CSchoel/nolds/

The modifications allow certain quantities, like the correlation dimension, to be calculated 
for multivariate time series. 

We are not connected to the creators of nolds, and we have included this file here only to ensure 
reproducibility of our benchmark experiments. Please do not use or adapt this code without citing 
the creators of nolds, and please heed their license.

"""

import numpy as np
import warnings

def rowwise_euclidean(x, y):
    """Computes the euclidean distance across rows"""
    return np.sqrt(np.sum((x - y)**2, axis=1))

def corr_dim(*args, **kwargs):
    raise NotImplementedError("corr_dim has been removed from the library." + \
                              "Please use dysts.analysis.gp_dim instead.")

def lyap_r_len(**kwargs):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Helper function that calculates the minimum number of data points required
    to use lyap_r.
    Note that none of the required parameters may be set to None.
    Kwargs:
        kwargs(dict):
            arguments used for lyap_r (required: emb_dim, lag, trajectory_len and
            min_tsep)
    Returns:
        minimum number of data points required to call lyap_r with the given
        parameters
    """
    # minimum length required to find single orbit vector
    min_len = (kwargs['emb_dim'] - 1) * kwargs['lag'] + 1
    # we need trajectory_len orbit vectors to follow a complete trajectory
    min_len += kwargs['trajectory_len'] - 1
    # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
    min_len += kwargs['min_tsep'] * 2 + 1
    return min_len

def lyap_r(data, lag=None, min_tsep=None, tau=1, min_neighbors=20,
                     trajectory_len=20, fit="RANSAC", debug_plot=False, debug_data=False,
                     plot_file=None, fit_offset=0):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py


    Estimates the largest Lyapunov exponent using the algorithm of Rosenstein
    et al. [lr_1]_.
    Explanation of Lyapunov exponents:
        See lyap_e.
    Explanation of the algorithm:
        The algorithm of Rosenstein et al. is only able to recover the largest
        Lyapunov exponent, but behaves rather robust to parameter choices.
        The idea for the algorithm relates closely to the definition of Lyapunov
        exponents. First, the dynamics of the data are reconstructed using a delay
        embedding method with a lag, such that each value x_i of the data is mapped
        to the vector
        X_i = [x_i, x_(i+lag), x_(i+2*lag), ..., x_(i+(emb_dim-1) * lag)]
        For each such vector X_i, we find the closest neighbor X_j using the
        euclidean distance. We know that as we follow the trajectories from X_i and
        X_j in time in a chaotic system the distances between X_(i+k) and X_(j+k)
        denoted as d_i(k) will increase according to a power law
        d_i(k) = c * e^(lambda * k) where lambda is a good approximation of the
        highest Lyapunov exponent, because the exponential expansion along the axis
        associated with this exponent will quickly dominate the expansion or
        contraction along other axes.
        To calculate lambda, we look at the logarithm of the distance trajectory,
        because log(d_i(k)) = log(c) + lambda * k. This gives a set of lines
        (one for each index i) whose slope is an approximation of lambda. We
        therefore extract the mean log trajectory d'(k) by taking the mean of
        log(d_i(k)) over all orbit vectors X_i. We then fit a straight line to
        the plot of d'(k) versus k. The slope of the line gives the desired
        parameter lambda.
    Method for choosing min_tsep:
        Usually we want to find neighbors between points that are close in phase
        space but not too close in time, because we want to avoid spurious
        correlations between the obtained trajectories that originate from temporal
        dependencies rather than the dynamic properties of the system. Therefore it
        is critical to find a good value for min_tsep. One rather plausible
        estimate for this value is to set min_tsep to the mean period of the
        signal, which can be obtained by calculating the mean frequency using the
        fast fourier transform. This procedure is used by default if the user sets
        min_tsep = None.
    Method for choosing lag:
        Another parameter that can be hard to choose by instinct alone is the lag
        between individual values in a vector of the embedded orbit. Here,
        Rosenstein et al. suggest to set the lag to the distance where the
        autocorrelation function drops below 1 - 1/e times its original (maximal)
        value. This procedure is used by default if the user sets lag = None.
    References:
        .. [lr_1] M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
             “A practical method for calculating largest Lyapunov exponents from
             small data sets,” Physica D: Nonlinear Phenomena, vol. 65, no. 1,
             pp. 117–134, 1993.
    Reference Code:
        .. [lr_a] mirwais, "Largest Lyapunov Exponent with Rosenstein's Algorithm",
             url: http://www.mathworks.com/matlabcentral/fileexchange/38424-largest-lyapunov-exponent-with-rosenstein-s-algorithm
        .. [lr_b] Shapour Mohammadi, "LYAPROSEN: MATLAB function to calculate
             Lyapunov exponent",
             url: https://ideas.repec.org/c/boc/bocode/t741502.html
    Args:
        data (iterable of float):
            (one-dimensional) time series
    Kwargs:
        emb_dim (int):
            embedding dimension for delay embedding
        lag (float):
            lag for delay embedding
        min_tsep (float):
            minimal temporal separation between two "neighbors" (default:
            find a suitable value by calculating the mean period of the data)
        tau (float):
            step size between data points in the time series in seconds
            (normalization scaling factor for exponents)
        min_neighbors (int):
            if lag=None, the search for a suitable lag will be stopped when the
            number of potential neighbors for a vector drops below min_neighbors
        trajectory_len (int):
            the time (in number of data points) to follow the distance
            trajectories between two neighboring points
        fit (str):
            the fitting method to use for the line fit, either 'poly' for normal
            least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
            is more robust to outliers
        debug_plot (boolean):
            if True, a simple plot of the final line-fitting step will
            be shown
        debug_data (boolean):
            if True, debugging data will be returned alongside the result
        plot_file (str):
            if debug_plot is True and plot_file is not None, the plot will be saved
            under the given file name instead of directly showing it through
            ``plt.show()``
        fit_offset (int):
            neglect the first fit_offset steps when fitting
    Returns:
        float:
            an estimate of the largest Lyapunov exponent (a positive exponent is
            a strong indicator for chaos)
        (1d-vector, 1d-vector, list):
            only present if debug_data is True: debug data of the form
            ``(ks, div_traj, poly)`` where ``ks`` are the x-values of the line fit, 
            ``div_traj`` are the y-values and ``poly`` are the line coefficients
            (``[slope, intercept]``).
    """
    # convert data to float to avoid overflow errors in rowwise_euclidean
    data = np.asarray(data, dtype="float32")
    n = len(data)
    max_tsep_factor = 0.25
    if lag is None or min_tsep is None:
        # both the algorithm for lag and min_tsep need the fft
        f = np.fft.rfft(data, n * 2 - 1)
    if min_tsep is None:
        # calculate min_tsep as mean period (= 1 / mean frequency)
        mf = np.fft.rfftfreq(n * 2 - 1) * np.abs(f)
        mf = np.mean(mf[1:]) / np.sum(np.abs(f[1:]))
        min_tsep = int(np.ceil(1.0 / mf))
        if min_tsep > max_tsep_factor * n:
            min_tsep = int(max_tsep_factor * n)
            msg = "signal has very low mean frequency, setting min_tsep = {:d}"
            warnings.warn(msg.format(min_tsep), RuntimeWarning)
    # if lag is None:
    #     # calculate the lag as point where the autocorrelation drops to (1 - 1/e)
    #     # times its maximum value
    #     # note: the Wiener–Khinchin theorem states that the spectral
    #     # decomposition of the autocorrelation function of a process is the power
    #     # spectrum of that process
    #     # => we can use fft to calculate the autocorrelation
    #     acorr = np.fft.irfft(f * np.conj(f))
    #     acorr = np.roll(acorr, n - 1)
    #     eps = acorr[n - 1] * (1 - 1.0 / np.e)
    #     lag = 1
    #     # small helper function to calculate resulting number of vectors for a
    #     # given lag value
    #     def nb_neighbors(lag_value):
    #         min_len = lyap_r_len(
    #             emb_dim=emb_dim, lag=i, trajectory_len=trajectory_len,
    #             min_tsep=min_tsep
    #         )
    #         return max(0, n - min_len)
    #     # find lag
    #     for i in range(1,n):
    #         lag = i
    #         if acorr[n - 1 + i] < eps or acorr[n - 1 - i] < eps:
    #             break
    #         if nb_neighbors(i) < min_neighbors:
    #             msg = "autocorrelation declined too slowly to find suitable lag" \
    #                 + ", setting lag to {}"
    #             warnings.warn(msg.format(lag), RuntimeWarning)
    #             break
    # min_len = lyap_r_len(
    #     emb_dim=emb_dim, lag=lag, trajectory_len=trajectory_len,
    #     min_tsep=min_tsep
    # )
    # if len(data) < min_len:
    #     msg = "for emb_dim = {}, lag = {}, min_tsep = {} and trajectory_len = {}" \
    #         + " you need at least {} datapoints in your time series"
    #     warnings.warn(
    #         msg.format(emb_dim, lag, min_tsep, trajectory_len, min_len),
    #         RuntimeWarning
    #     )
    # delay embedding
    #orbit = delay_embedding(data, emb_dim, lag)
    orbit = data
    m = len(orbit)
    # construct matrix with pairwise distances between vectors in orbit
    dists = np.array([rowwise_euclidean(orbit, orbit[i]) for i in range(m)])
    # we do not want to consider vectors as neighbor that are less than min_tsep
    # time steps together => mask the distances min_tsep to the right and left of
    # each index by setting them to infinity (will never be considered as nearest
    # neighbors)
    for i in range(m):
        dists[i, max(0, i - min_tsep):i + min_tsep + 1] = float("inf")
    # check that we have enough data points to continue
    ntraj = m - trajectory_len + 1
    min_traj = min_tsep * 2 + 2 # in each row min_tsep + 1 disances are inf
    if ntraj <= 0:
        msg = "Not enough data points. Need {} additional data points to follow " \
                + "a complete trajectory."
        raise ValueError(msg.format(-ntraj+1))
    if ntraj < min_traj:
        # not enough data points => there are rows where all values are inf
        assert np.any(np.all(np.isinf(dists[:ntraj, :ntraj]), axis=1))
        msg = "Not enough data points. At least {} trajectories are required " \
                + "to find a valid neighbor for each orbit vector with min_tsep={} " \
                + "but only {} could be created."
        raise ValueError(msg.format(min_traj, min_tsep, ntraj))
    assert np.all(np.any(np.isfinite(dists[:ntraj, :ntraj]), axis=1))
    # find nearest neighbors (exclude last columns, because these vectors cannot
    # be followed in time for trajectory_len steps)
    nb_idx = np.argmin(dists[:ntraj, :ntraj], axis=1)
    
    # build divergence trajectory by averaging distances along the trajectory
    # over all neighbor pairs
    div_traj = np.zeros(trajectory_len, dtype=float)
    for k in range(trajectory_len):
        # calculate mean trajectory distance at step k
        indices = (np.arange(ntraj) + k, nb_idx + k)
        div_traj_k = dists[indices]
        # filter entries where distance is zero (would lead to -inf after log)
        nonzero = np.where(div_traj_k != 0)
        if len(nonzero[0]) == 0:
            # if all entries where zero, we have to use -inf
            div_traj[k] = -np.inf
        else:
            div_traj[k] = np.mean(np.log(div_traj_k[nonzero]))
    # filter -inf entries from mean trajectory
    ks = np.arange(trajectory_len)
    finite = np.where(np.isfinite(div_traj))
    ks = ks[finite]
    div_traj = div_traj[finite]
    if len(ks) < 1:
        # if all points or all but one point in the trajectory is -inf, we cannot
        # fit a line through the remaining points => return -inf as exponent
        poly = [-np.inf, 0]
    else:
        # normal line fitting
        poly = poly_fit(ks[fit_offset:], div_traj[fit_offset:], 1, fit=fit)
    if debug_plot:
        plot_reg(ks[fit_offset:], div_traj[fit_offset:], poly, "k", "log(d(k))", fname=plot_file)
    le = poly[0] / tau
    if debug_data:
        return (le, (ks, div_traj, poly))
    else:
        return le

def logarithmic_n(min_n, max_n, factor):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.
    Non-integer results are rounded down.
    Args:
        min_n (float):
            minimum value (must be < max_n)
        max_n (float):
            maximum value (must be > min_n)
        factor (float):
            factor used to increase min_n (must be > 1)
    Returns:
        list of integers:
            min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
            without duplicates
    """
    assert max_n > min_n
    assert factor > 1
    # stop condition: min * f^x = max
    # => f^x = max/min
    # => x = log(max/min) / log(f)
    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(np.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return ns


def dfa(data, nvals=None, overlap=True, order=1, fit_trend="poly",
                fit_exp="RANSAC", debug_plot=False, debug_data=False, plot_file=None):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Performs a detrended fluctuation analysis (DFA) on the given data
    Recommendations for parameter settings by Hardstone et al.:
        * nvals should be equally spaced on a logarithmic scale so that each window
            scale hase the same weight
        * min(nvals) < 4 does not make much sense as fitting a polynomial (even if
            it is only of order 1) to 3 or less data points is very prone.
        * max(nvals) > len(data) / 10 does not make much sense as we will then have
            less than 10 windows to calculate the average fluctuation
        * use overlap=True to obtain more windows and therefore better statistics
            (at an increased computational cost)
    Explanation of DFA:
        Detrended fluctuation analysis, much like the Hurst exponent, is used to
        find long-term statistical dependencies in time series.
        The idea behind DFA originates from the definition of self-affine
        processes. A process X is said to be self-affine if the standard deviation
        of the values within a window of length n changes with the window length
        factor L in a power law:
        std(X,L * n) = L^H * std(X, n)
        where std(X, k) is the standard deviation of the process X calculated over
        windows of size k. In this equation, H is called the Hurst parameter, which
        behaves indeed very similar to the Hurst exponent.
        Like the Hurst exponent, H can be obtained from a time series by
        calculating std(X,n) for different n and fitting a straight line to the
        plot of log(std(X,n)) versus log(n).
        To calculate a single std(X,n), the time series is split into windows of
        equal length n, so that the ith window of this size has the form
        W_(n,i) = [x_i, x_(i+1), x_(i+2), ... x_(i+n-1)]
        The value std(X,n) is then obtained by calculating std(W_(n,i)) for each i
        and averaging the obtained values over i.
        The aforementioned definition of self-affinity, however, assumes that the
        process is    non-stationary (i.e. that the standard deviation changes over
        time) and it is highly influenced by local and global trends of the time
        series.
        To overcome these problems, an estimate alpha of H is calculated by using a
        "walk" or "signal profile" instead of the raw time series. This walk is
        obtained by substracting the mean and then taking the cumulative sum of the
        original time series. The local trends are removed for each window
        separately by fitting a polynomial p_(n,i) to the window W_(n,i) and then
        calculating W'_(n,i) = W_(n,i) - p_(n,i) (element-wise substraction).
        We then calculate std(X,n) as before only using the "detrended" window
        W'_(n,i) instead of W_(n,i). Instead of H we obtain the parameter alpha
        from the line fitting.
        For alpha < 1 the underlying process is stationary and can be modelled as
        fractional Gaussian noise with H = alpha. This means for alpha = 0.5 we
        have no correlation or "memory", for 0.5 < alpha < 1 we have a memory with
        positive correlation and for alpha < 0.5 the correlation is negative.
        For alpha > 1 the underlying process is non-stationary and can be modeled
        as fractional Brownian motion with H = alpha - 1.
    References:
        .. [dfa_1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
                             H. E. Stanley, and A. L. Goldberger, “Mosaic organization of
                             DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.
        .. [dfa_2] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
                             V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
                             “Detrended fluctuation analysis: A scale-free view on neuronal
                             oscillations,” Frontiers in Physiology, vol. 30, 2012.
    Reference code:
        .. [dfa_a] Peter Jurica, "Introduction to MDFA in Python",
             url: http://bsp.brain.riken.jp/~juricap/mdfa/mdfaintro.html
        .. [dfa_b] JE Mietus, "dfa",
             url: https://www.physionet.org/physiotools/dfa/dfa-1.htm
        .. [dfa_c] "DFA" function in R package "fractal"
    Args:
        data (array-like of float):
            time series
    Kwargs:
        nvals (iterable of int):
            subseries sizes at which to calculate fluctuation
            (default: logarithmic_n(4, 0.1*len(data), 1.2))
        overlap (boolean):
            if True, the windows W_(n,i) will have a 50% overlap,
            otherwise non-overlapping windows will be used
        order (int):
            (polynomial) order of trend to remove
        fit_trend (str):
            the fitting method to use for fitting the trends, either 'poly'
            for normal least squares polynomial fitting or 'RANSAC' for
            RANSAC-fitting which is more robust to outliers but also tends to
            lead to unstable results
        fit_exp (str):
            the fitting method to use for the line fit, either 'poly' for normal
            least squares polynomial fitting or 'RANSAC' for RANSAC-fitting which
            is more robust to outliers
        debug_plot (boolean):
            if True, a simple plot of the final line-fitting step will be shown
        debug_data (boolean):
            if True, debugging data will be returned alongside the result
        plot_file (str):
            if debug_plot is True and plot_file is not None, the plot will be saved
            under the given file name instead of directly showing it through
            ``plt.show()``
    Returns:
        float:
            the estimate alpha for the Hurst parameter (alpha < 1: stationary
            process similar to fractional Gaussian noise with H = alpha,
            alpha > 1: non-stationary process similar to fractional Brownian
            motion with H = alpha - 1)
        (1d-vector, 1d-vector, list):
            only present if debug_data is True: debug data of the form
            ``(nvals, fluctuations, poly)`` where ``nvals`` are the values used for
            log(n), ``fluctuations`` are the corresponding log(std(X,n)) and ``poly``
            are the line coefficients (``[slope, intercept]``)
    """
    data = np.asarray(data)
    total_N = len(data)
    if nvals is None:
        if total_N > 70:
            nvals = logarithmic_n(4, 0.1 * total_N, 1.2)
        elif total_N > 10:
            nvals = [4, 5, 6, 7, 8, 9]
        else:
            nvals = [total_N-2, total_N-1]
            msg = "choosing nvals = {} , DFA with less than ten data points is " \
                    + "extremely unreliable"
            warnings.warn(msg.format(nvals),RuntimeWarning)
    if len(nvals) < 2:
        raise ValueError("at least two nvals are needed")
    if np.min(nvals) < 2:
        raise ValueError("nvals must be at least two")
    if np.max(nvals) >= total_N:
        raise ValueError("nvals cannot be larger than the input size")
    # create the signal profile
    # (cumulative sum of deviations from the mean => "walk")
    walk = np.cumsum(data - np.mean(data))
    fluctuations = []
    for n in nvals:
        assert n >= 2
        # subdivide data into chunks of size n
        if overlap:
            # step size n/2 instead of n
            d = np.array([walk[i:i + n] for i in range(0, len(walk) - n, n // 2)])
        else:
            # non-overlapping windows => we can simply do a reshape
            d = walk[:total_N - (total_N % n)]
            d = d.reshape((total_N // n, n))
        # calculate local trends as polynomes
        x = np.arange(n)
        tpoly = [poly_fit(x, d[i], order, fit=fit_trend)
                         for i in range(len(d))]
        tpoly = np.array(tpoly)
        trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
        # calculate standard deviation ("fluctuation") of walks in d around trend
        flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
        # calculate mean fluctuation over all subsequences
        f_n = np.sum(flucs) / len(flucs)
        fluctuations.append(f_n)
    fluctuations = np.array(fluctuations)
    # filter zeros from fluctuations
    nonzero = np.where(fluctuations != 0)
    nvals = np.array(nvals)[nonzero]
    fluctuations = fluctuations[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        poly = [np.nan, np.nan]
    else:
        poly = poly_fit(np.log(nvals), np.log(fluctuations), 1,
                                        fit=fit_exp)
    if debug_plot:
        plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)",
                         fname=plot_file)
    if debug_data:
        return (poly[0], (np.log(nvals), np.log(fluctuations), poly))
    else:
        return poly[0]


def lyap_e(*args, **kwargs):
    raise NotImplementedError("lyap_e has been removed from the library." + \
                              "Please use dysts.analysis.find_lyapunov_exponents instead.")
