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

def rowwise_chebyshev(x, y):
    return np.max(np.abs(x - y), axis=1)

def rowwise_euclidean(x, y):
    """Computes the euclidean distance across rows"""
    return np.sqrt(np.sum((x - y)**2, axis=1))

def logarithmic_r(min_n, max_n, factor):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.
    Args:
    min_n (float):
      minimum value (must be < max_n)
    max_n (float):
      maximum value (must be > min_n)
    factor (float):
      factor used to increase min_n (must be > 1)
    Returns:
    list of floats:
      min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
    """
    assert max_n > min_n
    assert factor > 1
    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    return [min_n * (factor ** i) for i in range(max_i + 1)]

def poly_fit(x, y, degree, fit="RANSAC"):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Check if we can use RANSAC

    """
    if fit == "RANSAC":
        try:
            # ignore ImportWarnings in sklearn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ImportWarning)
                import sklearn.linear_model as sklin
                import sklearn.preprocessing as skpre
        except ImportError:
            warnings.warn(
                "fitting mode 'RANSAC' requires the package sklearn, using"
                + " 'poly' instead",
                RuntimeWarning)
            fit = "poly"

    if fit == "poly":
        return np.polyfit(x, y, degree)
    elif fit == "RANSAC":
        model = sklin.RANSACRegressor(sklin.LinearRegression(fit_intercept=False))
        xdat = np.asarray(x)
        if len(xdat.shape) == 1:
            # interpret 1d-array as list of len(x) samples instead of
            # one sample of length len(x)
            xdat = xdat.reshape(-1, 1)
        polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
        try:
            model.fit(polydat, y)
            coef = model.estimator_.coef_[::-1]
        except ValueError:
            warnings.warn(
                "RANSAC did not reach consensus, "
                + "using numpy's polyfit",
                RuntimeWarning)
            coef = np.polyfit(x, y, degree)
        return coef
    else:
        raise ValueError("invalid fitting mode ({})".format(fit))

def corr_dim(data, rvals=None, dist=rowwise_euclidean,
                    fit="RANSAC", debug_plot=False, debug_data=False, plot_file=None):
    """
    Adapted from the nolds Python library:
    https://github.com/CSchoel/nolds/blob/master/nolds/measures.py

    Calculates the correlation dimension with the Grassberger-Procaccia algorithm
    Explanation of correlation dimension:
        The correlation dimension is a characteristic measure that can be used
        to describe the geometry of chaotic attractors. It is defined using the
        correlation sum C(r) which is the fraction of pairs of points X_i in the
        phase space whose distance is smaller than r.
        If the relation between C(r) and r can be described by the power law
        C(r) ~ r^D
        then D is called the correlation dimension of the system.
        In a d-dimensional system, the maximum value for D is d. This value is
        obtained for systems that expand uniformly in each dimension with time.
        The lowest possible value is 0 for a system with constant C(r) (i.e. a
        system that visits just one point in the phase space). Generally if D is
        lower than d and the system has an attractor, this attractor is called
        "strange" and D is a measure of this "strangeness".
    Explanation of the algorithm:
        The Grassberger-Procaccia algorithm calculates C(r) for a range of
        different r and then fits a straight line into the plot of log(C(r))
        versus log(r).
    References:
        .. [cd_1] P. Grassberger and I. Procaccia, “Characterization of strange
                            attractors,” Physical review letters, vol. 50, no. 5, p. 346,
                            1983.
        .. [cd_2] P. Grassberger and I. Procaccia, “Measuring the strangeness of
                            strange attractors,” Physica D: Nonlinear Phenomena, vol. 9,
                            no. 1, pp. 189–208, 1983.
        .. [cd_3] P. Grassberger, “Grassberger-Procaccia algorithm,”
                            Scholarpedia, vol. 2, no. 5, p. 3043.
                            urL: http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm
    Reference Code:
        .. [cd_a] "corrDim" function in R package "fractal",
                            url: https://cran.r-project.org/web/packages/fractal/fractal.pdf
        .. [cd_b] Peng Yuehua, "Correlation dimension",
                            url: http://de.mathworks.com/matlabcentral/fileexchange/24089-correlation-dimension
    Args:
        data (array-like of float):
            time series of data points
        emb_dim (int):
            embedding dimension
    Kwargs:
        rvals (iterable of float):
            list of values for to use for r
            (default: logarithmic_r(0.1 * std, 0.5 * std, 1.03))
        dist (function (2d-array, 1d-array) -> 1d-array):
            row-wise difference function
        fit (str):
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
            correlation dimension as slope of the line fitted to log(r) vs log(C(r))
        (1d-vector, 1d-vector, list):
            only present if debug_data is True: debug data of the form
            ``(rvals, csums, poly)`` where ``rvals`` are the values used for log(r), 
            ``csums`` are the corresponding log(C(r)) and ``poly`` are the line 
            coefficients (``[slope, intercept]``)
    """
    data = np.asarray(data)

    # TODO what are good values for r?
    # TODO do this for multiple values of emb_dim?
    if rvals is None:
        sd = np.std(data)
        rvals = logarithmic_r(0.1 * sd, 0.5 * sd, 1.03)
    n = len(data)
    #orbit = delay_embedding(data, emb_dim, lag=1)
    orbit = data
    dists = np.array([dist(orbit, orbit[i]) for i in range(len(orbit))])
    csums = []
    for r in rvals:
        s = 1.0 / (n * (n - 1)) * np.sum(dists < r)
        csums.append(s)
    csums = np.array(csums)
    # filter zeros from csums
    nonzero = np.where(csums != 0)
    rvals = np.array(rvals)[nonzero]
    csums = csums[nonzero]
    if len(csums) == 0:
        # all sums are zero => we cannot fit a line
        poly = [np.nan, np.nan]
    else:
        poly = poly_fit(np.log(rvals), np.log(csums), 1, fit=fit)
    if debug_plot:
        plot_reg(np.log(rvals), np.log(csums), poly, "log(r)", "log(C(r))",
                         fname=plot_file)
    if debug_data:
        return (poly[0], (np.log(rvals), np.log(csums), poly))
    else:
        return poly[0]



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
    
def dimension_difference(a1, a2):
    """
    Compare the dimensionality of two time series
    """
    d_true = a1.shape[0]
    #print(d_true)
    r = np.mean(a2**2, axis=0) # energy
    plt.plot(r)
    d_pred = 1 + np.where(np.abs(np.diff(r))<1e-2)[0][0]
    
    return d_pred - d_true

def standardize_scale(a, scale=1.0):
    """
    Standardize an array
    scale : float
        The number of standard deviations
    """
    return (a - np.mean(a))/(scale*np.std(a))

def lyap_e_len(**kwargs):
    """
    Helper function that calculates the minimum number of data points required
    to use lyap_e.
    Note that none of the required parameters may be set to None.
    Kwargs:
        kwargs(dict):
            arguments used for lyap_e (required: emb_dim, matrix_dim, min_nb
            and min_tsep)
    Returns:
        minimum number of data points required to call lyap_e with the given
        parameters
    """
    m = (kwargs['emb_dim'] - 1) // (kwargs['matrix_dim'] - 1)
    # minimum length required to find single orbit vector
    min_len = kwargs['emb_dim']
    # we need to follow each starting point of an orbit vector for m more steps
    min_len += m
    # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
    min_len += kwargs['min_tsep'] * 2
    # we need at least min_nb neighbors for each orbit vector
    min_len += kwargs['min_nb']
    return min_len

def lyap_e(data, emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1,
                     debug_plot=False, debug_data=False, plot_file=None):
    """
    Estimates the Lyapunov exponents for the given data using the algorithm of
    Eckmann et al. [le_1]_.
    Recommendations for parameter settings by Eckmann et al.:
        * long recording time improves accuracy, small tau does not
        * use large values for emb_dim
        * matrix_dim should be 'somewhat larger than the expected number of
            positive Lyapunov exponents'
        * min_nb = min(2 * matrix_dim, matrix_dim + 4)
    Explanation of Lyapunov exponents:
        The Lyapunov exponent describes the rate of separation of two
        infinitesimally close trajectories of a dynamical system in phase space.
        In a chaotic system, these trajectories diverge exponentially following
        the equation:
        \|X(t, X_0) - X(t, X_0 + eps)| = e^(lambda * t) * \|eps|
        In this equation X(t, X_0) is the trajectory of the system X starting at
        the point X_0 in phase space at time t. eps is the (infinitesimal)
        difference vector and lambda is called the Lyapunov exponent. If the
        system has more than one free variable, the phase space is
        multidimensional and each dimension has its own Lyapunov exponent. The
        existence of at least one positive Lyapunov exponent is generally seen as
        a strong indicator for chaos.
    Explanation of the Algorithm:
        To calculate the Lyapunov exponents analytically, the Jacobian of the
        system is required. The algorithm of Eckmann et al. therefore tries to
        estimate this Jacobian by reconstructing the dynamics of the system from
        which the time series was obtained. For this, several steps are required:
        * Embed the time series [x_1, x_2, ..., x_(N-1)] in an orbit of emb_dim
            dimensions (map each point x_i of the time series to a vector
            [x_i, x_(i+1), x_(i+2), ... x_(i+emb_dim-1)]).
        * For each vector X_i in this orbit find a radius r_i so that at least
            min_nb other vectors lie within (chebyshev-)distance r_i around X_i.
            These vectors will be called "neighbors" of X_i.
        * Find the Matrix T_i that sends points from the neighborhood of X_i to
            the neighborhood of X_(i+1). To avoid undetermined values in T_i, we
            construct T_i not with size (emb_dim x emb_dim) but with size
            (matrix_dim x matrix_dim), so that we have a larger "step size" m in the
            X_i, which are now defined as X'_i = [x_i, x_(i+m), x_(i+2m),
            ... x_(i+(matrix_dim-1)*m)]. This means that emb_dim-1 must be divisible
            by matrix_dim-1. The T_i are then found by a linear least squares fit,
            assuring that T_i (X_j - X_i) ~= X_(j+m) - X_(i+m) for any X_j in the
            neighborhood of X_i.
        * Starting with i = 1 and Q_0 = identity successively decompose the matrix
            T_i * Q_(i-1) into the matrices Q_i and R_i by a QR-decomposition.
        * Calculate the Lyapunov exponents from the mean of the logarithm of the
            diagonal elements of the matrices R_i. To normalize the Lyapunov
            exponents, they have to be divided by m and by the step size tau of the
            original time series.
    References:
        .. [le_1] J. P. Eckmann, S. O. Kamphorst, D. Ruelle, and S. Ciliberto,
             “Liapunov exponents from time series,” Physical Review A,
             vol. 34, no. 6, pp. 4971–4979, 1986.
    Reference code:
        .. [le_a] Manfred Füllsack, "Lyapunov exponent",
             url: http://systems-sciences.uni-graz.at/etextbook/sw2/lyapunov.html
        .. [le_b] Steve SIU, Lyapunov Exponents Toolbox (LET),
             url: http://www.mathworks.com/matlabcentral/fileexchange/233-let/content/LET/findlyap.m
        .. [le_c] Rainer Hegger, Holger Kantz, and Thomas Schreiber, TISEAN,
             url: http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html
    Args:
        data (array-like of float):
            (scalar) data points
    Kwargs:
        emb_dim (int):
            embedding dimension
        matrix_dim (int):
            matrix dimension (emb_dim - 1 must be divisible by matrix_dim - 1)
        min_nb (int):
            minimal number of neighbors
            (default: min(2 * matrix_dim, matrix_dim + 4))
        min_tsep (int):
            minimal temporal separation between two "neighbors"
        tau (float):
            step size of the data in seconds
            (normalization scaling factor for exponents)
        debug_plot (boolean):
            if True, a histogram matrix of the individual estimates will be shown
        debug_data (boolean):
            if True, debugging data will be returned alongside the result
        plot_file (str):
            if debug_plot is True and plot_file is not None, the plot will be saved
            under the given file name instead of directly showing it through
            ``plt.show()``
    Returns:
        float array:
            array of matrix_dim Lyapunov exponents (positive exponents are indicators
            for chaos)
        2d-array of floats:
            only present if debug_data is True: all estimates for the matrix_dim
            Lyapunov exponents from the x iterations of R_i. The shape of this debug
            data is (x, matrix_dim).
    """
    data = np.asarray(data)
    emb_dim = data.shape[-1]
    n = len(data)

    matrix_dim = emb_dim
    if (emb_dim - 1) % (matrix_dim - 1) != 0:
        raise ValueError("emb_dim - 1 must be divisible by matrix_dim - 1!")
    
    m = (emb_dim - 1) // (matrix_dim - 1)
    if min_nb is None:
        # minimal number of neighbors as suggested by Eckmann et al.
        min_nb = min(2 * matrix_dim, matrix_dim + 4)

    min_len = lyap_e_len(
        emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb, min_tsep=min_tsep
    )
    #print(min_len)
    if n < min_len:
        msg = "{} data points are not enough! For emb_dim = {}, matrix_dim = {}, " \
            + "min_tsep = {} and min_nb = {} you need at least {} data points " \
            + "in your time series"
        warnings.warn(
            msg.format(n, emb_dim, matrix_dim, min_tsep, min_nb, min_len),
            RuntimeWarning
        )

    # construct orbit as matrix (e = emb_dim)
    # x0 x1 x2 ... xe-1
    # x1 x2 x3 ... xe
    # x2 x3 x4 ... xe+1
    # ...

    # note: we need to be able to step m points further for the beta vector
    #             => maximum start index is n - emb_dim - m
#     orbit = delay_embedding(data[:-m], emb_dim, lag=1)
    
    orbit = data
    if len(orbit) < min_nb:
        print(len(data), "t", min_len)
        assert len(data) < min_len
        msg = "Not enough data points. Need at least {} additional data points " \
                + "to have min_nb = {} neighbor candidates"
        raise ValueError(msg.format(min_nb-len(orbit), min_nb))
    old_Q = np.identity(matrix_dim)
    lexp = np.zeros(matrix_dim, dtype="float32")
    lexp_counts = np.zeros(lexp.shape)
    debug_values = []
    # TODO reduce number of points to visit?
    # TODO performance test!
    for i in range(len(orbit)):
        # find neighbors for each vector in the orbit using the chebyshev distance
        diffs = rowwise_chebyshev(orbit, orbit[i])
        # ensure that we do not count the difference of the vector to itself
        diffs[i] = float('inf')
        # mask all neighbors that are too close in time to the vector itself
        mask_from = max(0, i - min_tsep)
        mask_to = min(len(diffs), i + min_tsep + 1)
        diffs[mask_from:mask_to] = np.inf
        indices = np.argsort(diffs)
        idx = indices[min_nb - 1]    # index of the min_nb-nearest neighbor
        r = diffs[idx]    # corresponding distance
        if np.isinf(r):
            assert len(data) < min_len
            msg = "Not enough data points. Orbit vector {} has less than min_nb = " \
                    + "{} valid neighbors that are at least min_tsep = {} time steps " \
                    + "away. Input must have at least length {}."
            raise ValueError(msg.format(i, min_nb, min_tsep, min_len))
        # there may be more than min_nb vectors at distance r (if multiple vectors
        # have a distance of exactly r)
        # => update index accordingly
        indices = np.where(diffs <= r)[0]
#         print(indices)
        
#         print(max(np.max(indices), i) + matrix_dim * m >= len(data))
        
#         print("indices", indices.shape)

        # find the matrix T_i that satisifies
        # T_i (orbit'[j] - orbit'[i]) = (orbit'[j+m] - orbit'[i+m])
        # for all neighbors j where orbit'[i] = [x[i], x[i+m],
        # ... x[i + (matrix_dim-1)*m]]

        # note that T_i has the following form:
        # 0    1    0    ... 0
        # 0    0    1    ... 0
        # ...
        # a0 a1 a2 ... a(matrix_dim-1)

        # This is because for all rows except the last one the aforementioned
        # equation has a clear solution since orbit'[j+m] - orbit'[i+m] =
        # [x[j+m]-x[i+m], x[j+2*m]-x[i+2*m], ... x[j+d_M*m]-x[i+d_M*m]]
        # and
        # orbit'[j] - orbit'[i] =
        # [x[j]-x[i], x[j+m]-x[i+m], ... x[j+(d_M-1)*m]-x[i+(d_M-1)*m]]
        # therefore x[j+k*m] - x[i+k*m] is already contained in
        # orbit'[j] - orbit'[x] for all k from 1 to matrix_dim-1. Only for
        # k = matrix_dim there is an actual problem to solve.

        # We can therefore find a = [a0, a1, a2, ... a(matrix_dim-1)] by
        # formulating a linear least squares problem (mat_X * a = vec_beta)
        # as follows.

        # build matrix X for linear least squares (d_M = matrix_dim)
        # x_j1 - x_i     x_j1+m - x_i+m     ...     x_j1+(d_M-1)m - x_i+(d_M-1)m
        # x_j2 - x_i     x_j2+m - x_i+m     ...     x_j2+(d_M-1)m - x_i+(d_M-1)m
        # ...

        # note: emb_dim = (d_M - 1) * m + 1

        mat_X = np.array([data[j] for j in indices])
        mat_X -= data[i]
        # build vector beta for linear least squares
        # x_j1+(d_M)m - x_i+(d_M)m
        # x_j2+(d_M)m - x_i+(d_M)m
        # ...

        if max(np.max(indices),i) + matrix_dim * m >= len(data):
#             print("t")
#             print(np.max(indices), i, matrix_dim, m, "L", len(data))
            continue
            
#             assert len(data) < min_len
#             msg = "Not enough data points. Cannot follow orbit vector {} for " \
#                     + "{} (matrix_dim * m) time steps. Input must have at least length " \
#                     + "{}."
#             raise ValueError(msg.format(i, matrix_dim * m, min_len))
        
        vec_beta = np.sum(data[indices + matrix_dim * m] - data[i + matrix_dim * m], axis=-1)
        # perform linear least squares
        a, _, _, _ = np.linalg.lstsq(mat_X, vec_beta, rcond=-1)
        
        a, _, _, _ = np.linalg.lstsq(data[indices] - data[i], data[indices + m] - data[i + m], rcond=-1)
        
#         all_vec_beta = (data[indices + matrix_dim * m] - data[i + matrix_dim * m]).T
#         all_a = list()
#         for item in all_vec_beta:
#             # perform linear least squares
#             a, _, _, _ = np.linalg.lstsq(mat_X, vec_beta, rcond=-1)
#             all_a.append(a)
            
#         print(all_a)
#         a = np.mean(all_a)        

        # build matrix T
        # 0    1    0    ... 0
        # 0    0    1    ... 0
        # ...
        # 0    0    0    ... 1
        # a1 a2 a3 ... a_(d_M)
#         mat_T = np.zeros((matrix_dim, matrix_dim))
#         mat_T[:-1, 1:] = np.identity(matrix_dim - 1)
#         mat_T[-1] = a
        
        mat_T = a

        # QR-decomposition of T * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(mat_T, old_Q))
        # force diagonal of R to be positive
        # (if QR = A then also QLL'R = A with L' = L^-1)
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[np.where(sign_diag == 0)] = 1
        sign_diag = np.diag(sign_diag)
        mat_Q = np.dot(mat_Q, sign_diag)
        mat_R = np.dot(sign_diag, mat_R)

        old_Q = mat_Q
        # successively build sum for Lyapunov exponents
        diag_R = np.diag(mat_R)
        # filter zeros in mat_R (would lead to -infs)
        idx = np.where(diag_R > 0)
        lexp_i = np.zeros(diag_R.shape, dtype="float32")
        lexp_i[idx] = np.log(diag_R[idx])
        lexp_i[np.where(diag_R == 0)] = np.inf
        if debug_plot or debug_data:
            debug_values.append(lexp_i / tau / m)
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1
    # end of loop over orbit vectors
    # it may happen that all R-matrices contained zeros => exponent really has
    # to be -inf
    if debug_plot:
        plot_histogram_matrix(np.array(debug_values), "layp_e", fname=plot_file)
    # normalize exponents over number of individual mat_Rs
    idx = np.where(lexp_counts > 0)
    lexp[idx] /= lexp_counts[idx]
    lexp[np.where(lexp_counts == 0)] = np.inf
    # normalize with respect to tau
    lexp /= tau
    # take m into account
    lexp /= m
    if debug_data:
        return (lexp, np.array(debug_values))
    return lexp