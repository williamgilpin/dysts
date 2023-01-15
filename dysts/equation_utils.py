"""
    Set of functions for computing the description length metric, as in e.g.
    the AI Feynman 2.0 paper, from a set of equations.
"""
import inspect
import warnings

from scipy.special import comb
import dysts.flows as flows
import numpy as np

try:
    from sympy import count_ops
    from sympy.parsing.sympy_parser import parse_expr
    has_sympy = True
except ModuleNotFoundError:
    has_sympy = False
    warnings.warn("Sympy not found. Please install Sympy before using the equation analysis utilities.")

"""
    Some of the following functions,
    Methods, bestApproximation, get_numberDL_scanned, and get_expr_complexity,
    are retrieved from https://github.com/SJ001/AI-Feynman.
    We modified get_expr_complexity to make it behave correctly with powers.
"""

"""
Citation:
@article{udrescu2020ai,
  title={AI Feynman: A physics-inspired method for symbolic regression},
  author={Udrescu, Silviu-Marian and Tegmark, Max},
  journal={Science Advances},
  volume={6},
  number={16},
  pages={eaay2631},
  year={2020},
  publisher={American Association for the Advancement of Science}
}

@article{udrescu2020ai,
  title={AI Feynman 2.0:
  Pareto-optimal symbolic regression exploiting graph modularity},
  author={Udrescu, Silviu-Marian and Tan, Andrew and Feng, Jiahai and Neto,
  Orisvaldo and Wu, Tailin and Tegmark, Max},
  journal={arXiv preprint arXiv:2006.10782},
  year={2020}
}
"""


def compute_medl(systems_list, param_list):
    """
    Computes the mean-error-description-length (MEDL) of all
    systems in the given system list.

    Attributes
    ----------
    systems_list: list
        a list of system's names whose to be computed
    param_list:
        a list of dictionaries which contains the parameter values
        used to generate the system

    Returns
    -------
        compl_list: a list MEDL of the systems in the same order as systems in
        the system list
    """
    compl_list = []
    for i, system in enumerate(systems_list):
        params = param_list[i]
        std_eqs = get_stand_expr(system, params)
        curr_compl = 0
        for eq in std_eqs:
            curr_compl += get_expr_complexity(eq)
        compl_list.append(curr_compl)
    return compl_list


def get_stand_expr(system, params):
    """
    Turns python functions of a given system into SymPy standard strings.
    Requires some ugly coding to deal with all the edge cases throughout
    the dyst database equations in the flows.py file.
    Attributes
    ----------
    system: string
        a string of the system name
    params:
        a dictionary which contains the parameter values
        used to generate the system

    Returns
    -------
        chunks: the system equation, reformatted for SymPy
    """
    system_str = inspect.getsource(getattr(flows, system))
    cut1 = system_str.find("return")
    system_str = system_str[: cut1 - 1]
    cut2 = system_str.rfind("):")
    system_str = system_str[cut2 + 5 :]
    chunks = system_str.split("\n")[:-1]
    for j, chunk in enumerate(chunks):
        cind = chunk.rfind("=")
        chunk = chunk[cind + 1 :]
        for key in params.keys():
            if "Lorenz" in system and "rho" in params.keys():
                chunk = chunk.replace("rho", str(params["rho"]), 10)
            if "Bouali2" in system:
                chunk = chunk.replace("bb", "0", 10)
            chunk = chunk.replace(key, str(params[key]), 10)
        # print(chunk)
        chunk = chunk.replace("--", "", 10)
        # get all variables into (x, y, z, w) form
        chunk = chunk.replace("q1", "x", 10)
        chunk = chunk.replace("q2", "y", 10)
        chunk = chunk.replace("p1", "z", 10)
        chunk = chunk.replace("p2", "w", 10)
        chunk = chunk.replace("px", "z", 10)
        chunk = chunk.replace("py", "w", 10)

        # Do any unique ones
        chunk = chunk.replace("(-10 + -4)", "-14", 10)
        chunk = chunk.replace("(-10 * -4)", "40", 10)
        chunk = chunk.replace("3.0 * 1.0", "3", 10)
        chunk = chunk.replace(" - 0 * z", "", 10)
        chunk = chunk.replace("(28 - 35)", "-7", 10)
        chunk = chunk.replace("(1 / 0.2 - 0.001)", "4.999", 10)
        chunk = chunk.replace("- (1.0 - 1.0) * x^2 ", "", 10)
        chunk = chunk.replace("(26 - 37)", "-11", 10)
        chunk = chunk.replace("64^2", "4096", 10)
        chunk = chunk.replace("64**2", "4096", 10)
        chunk = chunk.replace("3 / np.sqrt(2) * 0.55", "1.166726189", 10)
        chunk = chunk.replace("3 * np.sqrt(2) * 0.55", "2.333452378", 10)
        chunk = chunk.replace("+ -", "- ", 10)
        chunk = chunk.replace("-1.5 * -0.0026667", "0.00400005", 10)

        chunk = chunk.replace("- 0.0026667 * 0xz", "", 10)
        chunk = chunk.replace("1/4096", "0.000244140625", 10)
        chunk = chunk.replace("10/4096", "0.00244140625", 10)
        chunk = chunk.replace("28/4096", "0.0068359375", 10)
        chunk = chunk.replace("2.667/4096", "0.000651123046875", 10)
        chunk = chunk.replace("0.2 * 9", "1.8", 10)
        chunk = chunk.replace(" - 3 * 0", "", 10)
        chunk = chunk.replace("2 * 1", "2", 10)
        chunk = chunk.replace("3 * 2.1 * 0.49", "3.087", 10)
        chunk = chunk.replace("2 * 2.1", "4.2", 10)
        chunk = chunk.replace("-40 / -14", "2.85714285714", 10)
        # change notation of squared and cubed terms
        chunk = chunk.replace(" 1x", " x", 10)
        chunk = chunk.replace(" 1y", " y", 10)
        chunk = chunk.replace(" 1z", " z", 10)
        chunk = chunk.replace(" 1w", " w", 10)
        chunks[j] = chunk
        chunk = chunk.replace(" ", "", 400)
        chunk = chunk.replace("-x", "-1x", 10)
        chunk = chunk.replace("-y", "-1y", 10)
        chunk = chunk.replace("-z", "-1z", 10)
        chunk = chunk.replace("-w", "-1w", 10)
        chunk = chunk.replace("--", "-", 20)
    return chunks


def get_expr_complexity(expr):
    """Todo -- add comments to all the subfunctions here"""
    expr = parse_expr(expr, evaluate=True)
    compl = 0

    def is_atomic_number(expr):
        return expr.is_Atom and expr.is_number

    numbers_expr = [
        subexpression for subexpression in expr.args if is_atomic_number(subexpression)
    ]
    variables_expr = [
        subexpression
        for subexpression in expr.args
        if not (is_atomic_number(subexpression))
    ]

    for j in numbers_expr:
        try:
            compl = compl + get_number_DL_snapped(float(j))
        except Exception as e:
            compl = compl + 1000000
            print(e.message, e.args)

    # compute n, k: n basis functions appear k times
    n_uniq_vars = len(expr.free_symbols)
    n_uniq_ops = len(count_ops(expr, visual=True).free_symbols)

    N = n_uniq_vars + n_uniq_ops

    n_ops = count_ops(expr)
    n_vars = len(variables_expr)

    n_power_addional = 0
    for subexpression in variables_expr:
        if subexpression.is_Pow:
            b, e = subexpression.as_base_exp()
            if b.is_Symbol and e.is_Integer:
                n_power_addional += (e - 1) * 2 - 1

    K = n_ops + n_vars + n_power_addional

    if n_uniq_ops != 0 or n_uniq_ops != 0:
        compl = compl + K * np.log2(N)

    return compl


def bestApproximation(x, imax):
    def float2contfrac(x, nmax):
        x = float(x)
        c = [np.floor(x)]
        y = x - np.floor(x)
        k = 0
        while np.abs(y) != 0 and k < nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c

    def contfrac2frac(seq):
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num * u, num
        return num, den

    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[: i + 1]) for i in range(len(c))))

    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:, 0] / float(q[:, 1])

    def truncateContFrac(q, imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k, 0]), q[k, 1]) <= imax:
            k = k + 1
        return q[:k]

    def pval(p):
        p = p.astype(float)
        return 1 - np.exp(-(p**0.87) / 0.36)

    xsign = np.sign(x)
    q = truncateContFrac(
        contFracRationalApproximations(float2contfrac(abs(x), 20)), imax
    )

    if len(q) > 0:
        p = (
            np.abs(q[:, 0] / q[:, 1] - abs(x)).astype(float)
            * (1 + np.abs(q[:, 0]))
            * q[:, 1]
        )
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i, 0] / float(q[i, 1]), xsign * q[i, 0], q[i, 1], p[i])
    else:
        return (None, 0, 0, 1)


def get_number_DL_snapped(n):
    epsilon = 1e-10
    n = float(n)
    if np.isnan(n):
        return 1000000
    elif np.abs(n - int(n)) < epsilon:
        return np.log2(1 + abs(int(n)))
    elif np.abs(n - bestApproximation(n, 10000)[0]) < epsilon:
        _, numerator, denominator, _ = bestApproximation(n, 10000)
        return np.log2((1 + abs(numerator)) * abs(denominator))
    elif np.abs(n - np.pi) < epsilon:
        return np.log2(1 + 3)
    else:
        PrecisionFloorLoss = 1e-14
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2


def nonlinear_terms_from_coefficients(true_coefficients):
    """
        From the true coefficients extracted from the dysts flows,
        compute the number of each kind of nonlinear term. Only implemented
        for chaotic systems that are polynomial in nonlinearity, with degree
        less than or equal to four.
    """
    # number of terms that are constant, linear, quadratic, cubic, and quartic
    num_attractors = len(true_coefficients)
    number_nonlinear_terms = np.zeros((num_attractors, 5))
    for i in range(num_attractors):
        dim = true_coefficients[i].shape[0]
        number_nonlinear_terms[i, 0] = np.count_nonzero(true_coefficients[i][:, 0])
        number_nonlinear_terms[i, 1] = np.count_nonzero(
            true_coefficients[i][:, 1 : dim + 1]
        )
        num_quad = int(comb(2 + dim - 1, dim - 1))
        num_cubic = int(comb(3 + dim - 1, dim - 1))
        num_quartic = int(comb(4 + dim - 1, dim - 1))
        coeff_index = dim + 1 + num_quad
        number_nonlinear_terms[i, 2] = np.count_nonzero(
            true_coefficients[i][:, dim + 1 : coeff_index]
        )
        number_nonlinear_terms[i, 3] = np.count_nonzero(
            true_coefficients[i][:, coeff_index : coeff_index + num_cubic]
        )
        coeff_index += num_cubic
        number_nonlinear_terms[i, 4] = np.count_nonzero(
            true_coefficients[i][:, coeff_index:]
        )
    return number_nonlinear_terms


def make_dysts_true_coefficients(
    systems_list, all_sols_train, dimension_list, param_list
):
    """
        Turn dysts flows, written as python functions,
        into SymPy strings that we can use to extract the coefficients. This
        is currently very bad code, since it needs to go through and handle
        all sorts of edge cases for every equation. It seems to work
        as is, but will need modificationn as new flows are added.

        Note: this function only works for polynomially nonlinear systems,
        with degree less than or equal to four.
    """
    true_coefficients = []

    for i, system in enumerate(systems_list):
        # print(i, system)
        x_train = all_sols_train[system][0]
        if dimension_list[i] == 3:
            feature_names = ['1',
                             'x',
                             'y',
                             'z',
                             'x^2',
                             'x y',
                             'x z',
                             'y^2',
                             'y z',
                             'z^2',
                             'x^3',
                             'x^2 y',
                             'x^2 z',
                             'x y^2',
                             'x y z',
                             'x z^2',
                             'y^3',
                             'y^2 z',
                             'y z^2',
                             'z^3',
                             'x^4',
                             'x^3 y',
                             'x^3 z',
                             'x^2 y^2',
                             'x^2 y z',
                             'x^2 z^2',
                             'x y^3',
                             'x y^2 z',
                             'x y z^2',
                             'x z^3',
                             'y^4',
                             'y^3 z',
                             'y^2 z^2',
                             'y z^3',
                             'z^4']
        else:
            feature_names = ['1',
                             'x',
                             'y',
                             'z',
                             'w',
                             'x^2',
                             'x y',
                             'x z',
                             'x w',
                             'y^2',
                             'y z',
                             'y w',
                             'z^2',
                             'z w',
                             'w^2',
                             'x^3',
                             'x^2 y',
                             'x^2 z',
                             'x^2 w',
                             'x y^2',
                             'x y z',
                             'x y w',
                             'x z^2',
                             'x z w',
                             'x w^2',
                             'y^3',
                             'y^2 z',
                             'y^2 w',
                             'y z^2',
                             'y z w',
                             'y w^2',
                             'z^3',
                             'z^2 w',
                             'z w^2',
                             'w^3',
                             'x^4',
                             'x^3 y',
                             'x^3 z',
                             'x^3 w',
                             'x^2 y^2',
                             'x^2 y z',
                             'x^2 y w',
                             'x^2 z^2',
                             'x^2 z w',
                             'x^2 w^2',
                             'x y^3',
                             'x y^2 z',
                             'x y^2 w',
                             'x y z^2',
                             'x y z w',
                             'x y w^2',
                             'x z^3',
                             'x z^2 w',
                             'x z w^2',
                             'x w^3',
                             'y^4',
                             'y^3 z',
                             'y^3 w',
                             'y^2 z^2',
                             'y^2 z w',
                             'y^2 w^2',
                             'y z^3',
                             'y z^2 w',
                             'y z w^2',
                             'y w^3',
                             'z^4',
                             'z^3 w',
                             'z^2 w^2',
                             'z w^3',
                             'w^4']
        for k, feature in enumerate(feature_names):
            feature = feature.replace(" ", "", 10)
            feature = feature.replace("y^3z", "zy^3", 10)
            feature = feature.replace("x^3z", "zx^3", 10)
            feature = feature.replace("x^3y", "yx^3", 10)
            feature = feature.replace("z^3y", "yz^3", 10)
            feature = feature.replace("y^3x", "xy^3", 10)
            feature = feature.replace("z^3x", "xz^3", 10)
            feature_names[k] = feature
        # print(feature_names)
        num_poly = len(feature_names)
        coef_matrix_i = np.zeros((dimension_list[i], num_poly))
        system_str = inspect.getsource(getattr(flows, system))
        cut1 = system_str.find("return")
        system_str = system_str[: cut1 - 1]
        cut2 = system_str.rfind("):")
        system_str = system_str[cut2 + 5 :]
        chunks = system_str.split("\n")[:-1]
        params = param_list[i]
        # print(system, chunks)
        for j, chunk in enumerate(chunks):
            cind = chunk.rfind("=")
            chunk = chunk[cind + 1 :]
            for key in params.keys():
                if "Lorenz" in system and "rho" in params.keys():
                    chunk = chunk.replace("rho", str(params["rho"]), 10)
                if "Bouali2" in system:
                    chunk = chunk.replace("bb", "0", 10)
                chunk = chunk.replace(key, str(params[key]), 10)
            # print(chunk)
            chunk = chunk.replace("--", "", 10)
            chunk = chunk.replace("- -", "+ ", 10)
            # get all variables into (x, y, z, w) form
            chunk = chunk.replace("q1", "x", 10)
            chunk = chunk.replace("q2", "y", 10)
            chunk = chunk.replace("p1", "z", 10)
            chunk = chunk.replace("p2", "w", 10)
            chunk = chunk.replace("px", "z", 10)
            chunk = chunk.replace("py", "w", 10)
            # change notation of squared and cubed terms
            chunk = chunk.replace(" ** 2", "^2", 10)
            chunk = chunk.replace(" ** 3", "^3", 10)
            # reorder cubic terms
            chunk = chunk.replace("y * x^2", "x^2y", 10)
            chunk = chunk.replace("z * x^2", "x^2z", 10)
            chunk = chunk.replace("z * y^2", "y^2z", 10)
            # reorder quartic terms
            chunk = chunk.replace("y * x^3", "yx^3", 10)
            chunk = chunk.replace("z * x^3", "zx^3", 10)
            chunk = chunk.replace("z * y^2", "zy^3", 10)
            # Reorder quadratics
            chunk = chunk.replace("x * y", "xy", 10)
            chunk = chunk.replace("x * z", "xz", 10)
            chunk = chunk.replace("y * x", "xy", 10)
            chunk = chunk.replace("z * x", "xz", 10)
            chunk = chunk.replace("y * z", "yz", 10)
            chunk = chunk.replace("z * y", "yz", 10)
            chunk = chunk.replace("x * w", "xw", 10)
            chunk = chunk.replace("w * x", "xw", 10)
            chunk = chunk.replace("y * w", "yw", 10)
            chunk = chunk.replace("w * y", "yw", 10)
            chunk = chunk.replace("z * w", "zw", 10)
            chunk = chunk.replace("w * z", "zw", 10)

            # Do any unique ones
            chunk = chunk.replace("1 / 0.03", "33.3333333333", 10)
            chunk = chunk.replace("1.0 / 0.03", "33.3333333333", 10)
            chunk = chunk.replace("1 / 0.8", "1.25", 10)
            chunk = chunk.replace("1.0 / 0.8", "1.25", 10)
            chunk = chunk.replace("0.0322 / 0.8", "0.04025", 10)
            chunk = chunk.replace("0.49 / 0.03", "16.3333333333", 10)
            chunk = chunk.replace("(-10 + -4)", "-14", 10)
            chunk = chunk.replace("(-10 * -4)", "40", 10)
            chunk = chunk.replace("3.0 * 1.0", "3", 10)
            chunk = chunk.replace(" - 0 * z", "", 10)
            chunk = chunk.replace("(28 - 35)", "-7", 10)
            chunk = chunk.replace("(1 / 0.2 - 0.001)", "4.999", 10)
            chunk = chunk.replace("- (1.0 - 1.0) * x^2 ", "", 10)
            chunk = chunk.replace("(26 - 37)", "-11", 10)
            chunk = chunk.replace("64^2", "4096", 10)
            chunk = chunk.replace("64**2", "4096", 10)
            chunk = chunk.replace("3 / np.sqrt(2) * 0.55", "1.166726189", 10)
            chunk = chunk.replace("3 * np.sqrt(2) * 0.55", "2.333452378", 10)
            chunk = chunk.replace("+ -", "- ", 10)
            chunk = chunk.replace("-1.5 * -0.0026667", "0.00400005", 10)

            for num_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                for x_str in ["x", "y", "z", "w"]:
                    chunk = chunk.replace(num_str + " * " + x_str, num_str + x_str, 20)

            chunk = chunk.replace("- 0.0026667 * 0xz", "", 10)
            chunk = chunk.replace("1/4096", "0.000244140625", 10)
            chunk = chunk.replace("10/4096", "0.00244140625", 10)
            chunk = chunk.replace("28/4096", "0.0068359375", 10)
            chunk = chunk.replace("2.667/4096", "0.000651123046875", 10)
            chunk = chunk.replace("0.2 * 9", "1.8", 10)
            chunk = chunk.replace(" - 3 * 0", "", 10)
            chunk = chunk.replace("2 * 1", "2", 10)
            chunk = chunk.replace("3 * 2.1 * 0.49", "3.087", 10)
            chunk = chunk.replace("2 * 2.1", "4.2", 10)
            chunk = chunk.replace("-40 / -14", "2.85714285714", 10)
            # change notation of squared and cubed terms
            chunk = chunk.replace(" 1x", " x", 10)
            chunk = chunk.replace(" 1y", " y", 10)
            chunk = chunk.replace(" 1z", " z", 10)
            chunk = chunk.replace(" 1w", " w", 10)
            chunks[j] = chunk
            chunk = chunk.replace(" ", "", 400)
            chunk = chunk.replace("-x", "-1x", 10)
            chunk = chunk.replace("-y", "-1y", 10)
            chunk = chunk.replace("-z", "-1z", 10)
            chunk = chunk.replace("-w", "-1w", 10)
            chunk = chunk.replace("--", "-", 20)
            #         chunk = chunk.replace('- x', '-1x')
            #         chunkt = feature_chunk_compact.replace('- y', '-1y')
            #         chunk = feature_chunk_compact.replace('- z', '-1z')
            #         chunk = feature_chunk_compact.replace('- w', '-1w')

            # Okay strings are formatted. Time to read them into the
            # coefficient matrix
            for k, feature in enumerate(np.flip(feature_names[1:])):
                # print(k, feature)
                feature_ind = (chunk + " ").find(feature)
                if feature_ind != -1:
                    feature_chunk = chunk[: feature_ind + len(feature)]
                    find = max(feature_chunk.rfind("+"), feature_chunk.rfind("-"))
                    # print('find = ', find, feature_chunk)
                    if find == -1 or find == 0:
                        feature_chunk = feature_chunk[0:] + " "
                    else:
                        feature_chunk = feature_chunk[find:] + " "
                    # print(feature_chunk)
                    if feature_chunk != chunk:
                        feature_chunk_compact = feature_chunk.replace("+", "")
                        # print(feature, feature_chunk_compact[:-len(feature) - 1])
                        if (
                            len(
                                feature_chunk_compact[: -len(feature) - 1].replace(
                                    " ", ""
                                )
                            )
                            == 0
                        ):
                            coef_matrix_i[j, len(feature_names) - k - 1] = 1
                        else:
                            coef_matrix_i[j, len(feature_names) - k - 1] = float(
                                feature_chunk_compact[: -len(feature) - 1]
                            )
                        # print(feature_chunk, chunk)
                        chunk = chunk.replace(feature_chunk.replace(" ", ""), "")
                        #
                        # print(feature, 'Chunk after = ', chunk)
                        # if len(chunk.replace(' ', '')) == 0:
                        #    break
            if len(chunk.replace(" ", "")) != 0:
                coef_matrix_i[j, 0] = chunk.replace(" ", "")

        true_coefficients.append(coef_matrix_i)
    return true_coefficients
