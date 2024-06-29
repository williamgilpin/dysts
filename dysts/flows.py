"""
Various low-dimensional dynamical systems in Python.
For flows that occur on unbounded intervals (eg non-autonomous systems),
coordinates are transformed to a basis where the domain remains bounded

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)

"""

import numpy as np
from .base import DynSys, DynSysDelay, staticjit



class Lorenz(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, rho, sigma):
        xdot = sigma * y - sigma * x
        ydot = rho * x - x * z - y
        zdot = x * y - beta * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, beta, rho, sigma):
        row1 = [-sigma, sigma, 0]
        row2 = [rho - z, -1, -x]
        row3 = [y, x, -beta]
        return row1, row2, row3


class LorenzBounded(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, r, rho, sigma):
        xdot = sigma * y - sigma * x - sigma/r**2 * y * x ** 2 - sigma/r**2 * y ** 3 - sigma/r**2 * y * z ** 2 + sigma/r**2 * x ** 3 + sigma/r**2 * x * y ** 2 + sigma/r**2 * x * z ** 2
        ydot = rho * x - x * z - y - rho/r**2 * x ** 3 - rho/r**2 * x * y ** 2 - rho/r**2 * x * z ** 2 + 1/r**2 * z * x ** 3 + 1/r**2 * x * z * y ** 2 + 1/r**2 * x * z ** 3 + 1/r**2 * y * x ** 2 + 1/r**2 * y ** 3 + 1/r**2 * y * z ** 2
        zdot = x * y - beta * z - 1/r**2 * y * x ** 3 - 1/r**2 * x * y ** 3 - 1/r**2 * x * y * z ** 2 + beta/r**2 * z * x ** 2 + beta/r**2 * z * y ** 2 + beta/r**2 * z ** 3
        return xdot, ydot, zdot


class LorenzCoupled(DynSys):
    @staticjit
    def _rhs(x1, y1, z1, x2, y2, z2, t, beta, eps, rho, rho1, rho2, sigma):
        x1dot = sigma * y1 - sigma * x1
        y1dot = rho1 * x1 - x1 * z1 - y1
        z1dot = x1 * y1 - beta * z1
        x2dot = sigma * y2 - sigma * x2 + eps * x1 - eps * x2
        y2dot = rho2 * x2 - x2 * z2 - y2
        z2dot = x2 * y2 - beta * z2
        return x1dot, y1dot, z1dot, x2dot, y2dot, z2dot


class Lorenz96(DynSys):
    def rhs(self, X, t):
        Xdot = np.zeros_like(X)
        Xdot[0] = (X[1] - X[-2]) * X[-1] - X[0] + self.f
        Xdot[1] = (X[2] - X[-1]) * X[0] - X[1] + self.f
        Xdot[-1] = (X[0] - X[-3]) * X[-2] - X[-1] + self.f
        Xdot[2:-1] = (X[3:] - X[:-3]) * X[1:-2] - X[2:-1] + self.f
        return Xdot


class Lorenz84(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, g):
        xdot = -a * x - y ** 2 - z ** 2 + a * f
        ydot = -y + x * y - b * x * z + g
        zdot = -z + b * x * y + x * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, f, g):
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, -1 + x]
        return row1, row2, row3


class Rossler(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * x - c * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [0, -1, -1]
        row2 = [1, a, 0]
        row3 = [z, 0, x - c]
        return row1, row2, row3


class Thomas(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x + b * np.sin(y)
        ydot = -a * y + b * np.sin(z)
        zdot = -a * z + b * np.sin(x)
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [-a, b * np.cos(y), 0]
        row2 = [0, -a, b * np.cos(z)]
        row3 = [b * np.cos(x), 0, -a]
        return row1, row2, row3


class ThomasLabyrinth(Thomas):
    pass


class DoublePendulum(DynSys):
    @staticjit
    def _rhs(th1, th2, p1, p2, t, d, m):
        g = 9.82
        pre = 6 / (m * d ** 2)
        denom = 16 - 9 * np.cos(th1 - th2) ** 2
        th1_dot = pre * (2 * p1 - 3 * np.cos(th1 - th2) * p2) / denom
        th2_dot = pre * (8 * p2 - 3 * np.cos(th1 - th2) * p1) / denom
        p1_dot = (
            -0.5
            * (m * d ** 2)
            * (th1_dot * th2_dot * np.sin(th1 - th2) + 3 * (g / d) * np.sin(th1))
        )
        p2_dot = (
            -0.5
            * (m * d ** 2)
            * (-th1_dot * th2_dot * np.sin(th1 - th2) + 3 * (g / d) * np.sin(th2))
        )
        return th1_dot, th2_dot, p1_dot, p2_dot

    @staticjit
    def _postprocessing(th1, th2, p1, p2):
        return np.sin(th1), np.sin(th2), p1, p2


class SwingingAtwood(DynSys):
    @staticjit
    def _rhs(r, th, pr, pth, t, m1, m2):
        g = 9.82
        rdot = pr / (m1 + m2)
        thdot = pth / (m1 * r ** 2)
        prdot = pth ** 2 / (m1 * r ** 3) - m2 * g + m1 * g * np.cos(th)
        pthdot = -m1 * g * r * np.sin(th)
        return rdot, thdot, prdot, pthdot

    @staticjit
    def _postprocessing(r, th, pr, pth):
        return r, np.sin(th), pr, pth


class GuckenheimerHolmes(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, e, f):
        xdot = a * x - b * y + c * z * x + d * z * x ** 2 + d * z * y ** 2
        ydot = a * y + b * x + c * z * y
        zdot = e - z ** 2 - f * x ** 2 - f * y ** 2 - a * z ** 3
        return xdot, ydot, zdot
    
    # @staticjit
    # def _jac(x, y, z, t, a, b, c, d, e, f):
    #     row1 = a + c * z + 2 * d * z * x, b, -2 * f * x
    #     row2 = -b + 2 * d * z * y, a + c * z, -2 * f * y
    #     row3 = c * x + d * x ** 2 + d * y ** 2, c * y, -2 * z - 3 * a * z ** 2
    #     return row1, row2, row3


class HenonHeiles(DynSys):
    @staticjit
    def _rhs(x, y, px, py, t, lam):
        xdot = px
        ydot = py
        pxdot = -x - 2 * lam * x * y
        pydot = -y - lam * x ** 2 + lam * y ** 2
        return xdot, ydot, pxdot, pydot
    @staticjit
    def _jac(x, y, px, py, t, lam):
        row1 = [0, 0, 1, 0]
        row2 = [0, 0, 0, 1]
        row3 = [-1 - 2 * lam * y, -2 * lam * x, 0, 0]
        row4 = [-2 * lam * x, 2 * lam * y - 1, 0, 0]
        return row1, row2, row3, row4


class Halvorsen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x - b * y - b * z - y ** 2
        ydot = -a * y - b * z - b * x - z ** 2
        zdot = -a * z - b * x - b * y - x ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [-a, -b - 2 * y, -b]
        row2 = [-b, -a, -b - 2 * z]
        row3 = [-b - 2 * x, -b, -a]
        return row1, row2, row3


class Chua(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, m0, m1):
        ramp_x = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
        xdot = alpha * (y - x - ramp_x)
        ydot = x - y + z
        zdot = -beta * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, alpha, beta, m0, m1):
        dramp_xdx = m1 + 0.5 * (m0 - m1) * (np.sign(x + 1) - np.sign(x - 1))
        row1 = [-alpha - alpha * dramp_xdx, alpha, 0]
        row2 = [1, -1, 1]
        row3 = [0, -beta, 0]
        return row1, row2, row3

class MultiChua(DynSys):
    def diode(self, x):
        m, c = self.m, self.c
        total = m[-1] * x
        for i in range(1, 6):
            total += 0.5 * (m[i - 1] - m[i]) * (np.abs(x + c[i]) - np.abs(x - c[i]))
        return total

    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a * (y - self.diode(x))
        ydot = x - y + z
        zdot = -self.b * y
        return (xdot, ydot, zdot)


class Duffing(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, delta, gamma, omega):
        xdot = y
        ydot = -delta * y - beta * x - alpha * x ** 3 + gamma * np.cos(z)
        zdot = omega
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, alpha, beta, delta, gamma, omega):
        row1 = [0, 1, 0]
        row2 = [-3 * alpha * x ** 2 - beta, -delta, -gamma * np.sin(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.cos(z)


class MackeyGlass(DynSysDelay):
    @staticjit
    def _rhs(x, xt, t, beta, gamma, n, tau):
        xdot = beta * (xt / (1 + xt ** n)) - gamma * x
        return xdot


class IkedaDelay(DynSysDelay):
    @staticjit
    def _rhs(x, xt, t, c, mu, tau, x0):
        xdot = mu * np.sin(xt - x0) - c * x
        return xdot


class SprottDelay(IkedaDelay):
    pass


class VossDelay(DynSysDelay):
    @staticjit
    def _rhs(x, xt, t, alpha, tau):
        f = -10.44 * xt ** 3 - 13.95 * xt ** 2 - 3.63 * xt + 0.85
        xdot = -alpha * x + f
        return xdot


class ScrollDelay(DynSysDelay):
    @staticjit
    def _rhs(x, xt, t, alpha, beta, tau):
        f = np.tanh(10 * xt)
        xdot = -alpha * xt + beta * f
        return xdot


class PiecewiseCircuit(DynSysDelay):
    @staticjit
    def _rhs(x, xt, t, alpha, beta, c, tau):
        f = -((xt / c) ** 3) + 3 * xt / c
        xdot = -alpha * xt + beta * f
        return xdot


# ## this was not chaotic
# class ENSODelay(DynSysDelay):
#     @staticjit
#     def _rhs(x, xt, t, alpha, beta, tau):
#         xdot = x - x**3 - alpha * xt + beta
#         return xdot


class DoubleGyre(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, eps, omega):
        a = eps * np.sin(z)
        b = 1 - 2 * eps * np.sin(z)
        f = a * x ** 2 + b * x
        dx = -alpha * np.pi * np.sin(np.pi * f) * np.cos(np.pi * y)
        dy = alpha * np.pi * np.cos(np.pi * f) * np.sin(np.pi * y) * (2 * a * x + b)
        dz = omega
        return dx, dy, dz

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class BlinkingRotlet(DynSys):
    @staticjit
    def _rotlet(r, theta, a, b, bc):
        """A rotlet velocity field"""
        kappa = a ** 2 + (b ** 2 * r ** 2) / a ** 2 - 2 * b * r * np.cos(theta)
        gamma = (1 - r ** 2 / a ** 2) * (a ** 2 - (b ** 2 * r ** 2) / a ** 2)
        iota = (b ** 2 * r) / a ** 2 - b * np.cos(theta)
        zeta = b ** 2 + r ** 2 - 2 * b * r * np.cos(theta)
        nu = a ** 2 + b ** 2 - (2 * b ** 2 * r ** 2) / a ** 2
        vr = b * np.sin(theta) * (-bc * (gamma / kappa ** 2) - 1 / kappa + 1 / zeta)
        vth = (
            bc * (gamma * iota) / kappa ** 2
            + bc * r * nu / (a ** 2 * kappa)
            + iota / kappa
            - (r - b * np.cos(theta)) / zeta
        )
        return vr, vth

    @staticjit
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * np.tanh(tau * stiffness * np.sin(2 * np.pi * t / tau))

    def rhs(self, X, t):
        r, theta, tt = X
        weight = self._protocol(tt, self.tau)
        dr1, dth1 = self._rotlet(r, theta, self.a, self.b, self.bc)
        dr2, dth2 = self._rotlet(r, theta, self.a, -self.b, self.bc)
        dr = weight * dr1 + (1 - weight) * dr2
        dth = (weight * dth1 + (1 - weight) * dth2) / r
        dtt = 1
        return self.sigma * dr, self.sigma * dth, dtt

    def _postprocessing(self, r, th, tt):
        return r * np.cos(th), r * np.sin(th), np.sin(2 * np.pi * tt / self.tau)

class LidDrivenCavityFlow(DynSys):
    @staticjit
    def _lid(x, y, a, b, tau, u1, u2):
        """The velocity field when the left domain drives"""
        prefactor1 = 2 * u1 * np.sin(np.pi * x / a) / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
        prefactor2 = 2 * u2 * np.sin(2 * np.pi * x / a) / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
        vx1 = -b * np.pi * np.sinh(np.pi * b / a) * np.sinh(np.pi * y / a) + np.cosh(np.pi * b / a) * (np.pi * y * np.cosh(np.pi * y /a) + a * np.sinh(np.pi * y / a))
        vx2 = -2 * b * np.pi * np.sinh(2 * np.pi * b / a) * np.sinh(2 * np.pi * y / a) + np.cosh(2 * np.pi * b / a) * (2 * np.pi * y * np.cosh(2 * np.pi * y / a) + a * np.sinh(2 * np.pi * y / a))
        vx = prefactor1 * vx1 + prefactor2 * vx2

        prefactor1 = 2 * np.pi * u1 * np.cos(np.pi * x / a) / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
        prefactor2 = 4 * np.pi * u2 * np.cos(2 * np.pi * x / a) / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
        vy1 = b * np.sinh(np.pi * b / a) * np.cosh(np.pi * y / a) - np.cosh(np.pi * b / a) * y * np.sinh(np.pi * y / a)
        vy2 = b * np.sinh(2 * np.pi * b / a) * np.cosh(2 * np.pi * y / a) - np.cosh(2 * np.pi * b / a) * y * np.sinh(2 * np.pi * y / a)
        vy = prefactor1 * vy1 + prefactor2 * vy2
        
        # vy1 = b * np.sinh(np.pi * b / a) * np.cosh(np.pi * y / a) - np.cosh(np.pi * b / a) * y * np.sinh(np.pi * y / a)
        # vy2 = b * np.sinh(2 * np.pi * b / a) * np.cosh(2 * np.pi * y / a) - np.cosh(2 * np.pi * b / a) * y * np.sinh(2 * np.pi * y / a)
        # vy = np.pi * prefactor1 * vy1 + 2 * np.pi * prefactor2 * vy2

        return vx, vy

    # @staticjit
    # def _right(x, y, a, b, tau, u1, u2):
    #     """The velocity field when the right domain drives"""
    #     prefactor1 = 2 * u1 * np.sin(np.pi * x / a) / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
    #     prefactor2 = 2 * u2 * np.sin(2 * np.pi * x / a) / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
    #     vx1 = -b * np.pi * np.sinh(np.pi * b / a) * np.sinh(np.pi * y / a) - np.cosh(np.pi * b / a) * (np.pi * y * np.cosh(np.pi * y /a) + a * np.sinh(np.pi * y /a))
    #     vx2 = -4 * b * np.pi * np.sinh(2 * np.pi * b / a) * np.sinh(2 * np.pi * y / a) - np.cosh(2 * np.pi * b / a) * (2 * np.pi * y * np.cosh(2 * np.pi * y /a) + a * np.sinh(2 * np.pi * y /a))
    #     vx = prefactor1 * vx1 - prefactor2 * vx2

    #     prefactor1 = 2 * np.pi * u1 * np.cos(np.pi * x / a) / (2 * b * np.pi + a * np.sinh(2 * np.pi * b / a))
    #     prefactor2 = 4 * np.pi * u2 * np.cos(2 * np.pi * x / a) / (4 * b * np.pi + a * np.sinh(4 * np.pi * b / a))
    #     vy1 = -b * np.sinh(np.pi * b / a) * np.cosh(np.pi * y / a) + np.cosh(np.pi * b / a) * y * np.sinh(np.pi * y / a)
    #     vy2 = -2 * b * np.sinh(2 * np.pi * b / a) * np.cosh(2 * np.pi * y / a) + np.cosh(2 * np.pi * b / a) * 2 * y * np.sinh(2 * np.pi * y / a)
    #     vy = prefactor1 * vy1 + prefactor2 * vy2

    #     return vx, vy

    @staticjit
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * np.tanh(tau * stiffness * np.sin(2 * np.pi * t / tau))

    def rhs(self, X, t):
        x, y, tt = X
        weight = self._protocol(tt, self.tau)
        dx1, dy1 = self._lid(x, y, self.a, self.b, self.tau, self.u1, self.u2)
        dx2, dy2 = self._lid(x, y, self.a, self.b, self.tau, -self.u1, self.u2)
        dx = weight * dx1 + (1 - weight) * dx2
        dy = weight * dy1 + (1 - weight) * dy2
        dtt = 1
        return dx, dy, dtt

    def _postprocessing(self, x, y, tt):
        return x, y, np.sin(2 * np.pi * tt / self.tau)

class BlinkingVortex(BlinkingRotlet):
    pass

class InteriorSquirmer(DynSys):

    @staticjit
    def _rhs_static(r, th, t, a, g, n):

        nvals = np.arange(1, n + 1)
        sinvals, cosvals = np.sin(th * nvals), np.cos(th * nvals)
        rnvals = r ** nvals

        vrn = g * cosvals + a * sinvals
        vrn *= (nvals * rnvals * (r ** 2 - 1)) / r

        vth = 2 * r + (r ** 2 - 1) * nvals / r
        vth *= a * cosvals - g * sinvals
        vth *= rnvals

        return np.sum(vrn), np.sum(vth) / r
    
    @staticjit
    def _jac_static(r, th, t, a, g, n):

        nvals = np.arange(1, n + 1)
        sinvals, cosvals = np.sin(th * nvals), np.cos(th * nvals)
        rnvals = r ** nvals
        trigsum = a * sinvals + g * cosvals
        trigskew = a * cosvals - g * sinvals

        j11 = np.copy(trigsum)
        j11 *= nvals * rnvals * (2 * r ** 2 + (r ** 2 - 1) * (nvals - 1))
        j11 = (1 / r ** 2) * np.sum(j11)

        j12 = np.copy(trigskew)
        j12 *= -(nvals ** 2) * rnvals * (1 - r ** 2) / r
        j12 = np.sum(j12)

        j21 = 2 * rnvals * (2 * nvals + 1) * (-np.copy(trigskew))
        j21 += (n * (1 - r ** 2) * rnvals * (nvals - 1) / r ** 2) * np.copy(
            g * sinvals + a * cosvals
        )
        j21 = -np.sum(j21)

        j22 = np.copy(trigsum)
        j22 *= -nvals * rnvals * (2 * r + (r ** 2 - 1) * nvals / r)
        j22 = np.sum(j22)
        # (1 / r**2) *

        ## Correct for polar coordinates
        vth = np.copy(trigskew)
        vth *= 2 * r + (r ** 2 - 1) * nvals / r
        vth *= rnvals
        vth = np.sum(vth) / r
        j21 = j21 / r - vth / r
        j22 /= r

        return np.array([[j11, j12], [j21, j22]])

    @staticjit
    def _protocol(t, tau, stiffness=20):
        return 0.5 + 0.5 * np.tanh(tau * stiffness * np.sin(2 * np.pi * t / tau))
    
    def _postprocessing(self, r, th, tt):
        return r * np.cos(th), r * np.sin(th), np.sin(2 * np.pi * tt / self.tau)
    
    # def jac(self, X, t):
    #     r, th = X[0], X[1]
    #     phase = self._protocol(t, self.tau)
    #     return self._jac_static(r, th, t, self.a * phase, self.g * (1 - phase), self.n)
    
    def rhs(self, X, t):
        r, th, tt = X
        phase = self._protocol(tt, self.tau)
        dtt = 1
        dr, dth = self._rhs_static(r, th, t, self.a * phase, self.g * (1 - phase), self.n)
        return dr, dth, dtt


class OscillatingFlow(DynSys):
    @staticjit
    def _rhs(x, y, z, t, b, k, omega, u):
        f = x + b * np.sin(z)
        dx = u * np.cos(k * y) * np.sin(k * f)
        dy = -u * np.sin(k * y) * np.cos(k * f)
        dz = omega
        return dx, dy, dz

    def _postprocessing(self, x, y, z):
        return np.cos(self.k * x), y, np.sin(z)


class BickleyJet(DynSys):
    @staticjit
    def _rhs(y, x, z, t, ell, eps, k, omega, sigma, u):
        sechy = 1 / np.cosh(y / ell)
        inds = np.arange(3)
        un = k[inds] * (x - z * sigma[inds])
        dx = u * sechy ** 2 * (-1 - 2 * np.dot(np.cos(un), eps) * np.tanh(y / ell))
        dy = ell * u * sechy ** 2 * np.dot(eps * k, np.sin(un))
        dz = omega
        return dy, dx, dz

    def _postprocessing(self, x, y, z):
        km = np.min(self.k)
        sm = np.min(self.sigma)
        return x, np.sin(km * y), np.sin(self.omega * z * km * sm)


class ArnoldBeltramiChildress(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        dx = a * np.sin(z) + c * np.cos(y)
        dy = b * np.sin(x) + a * np.cos(z)
        dz = c * np.sin(y) + b * np.cos(x)
        return dx, dy, dz

    @staticjit
    def _postprocessing(x, y, z):
        return np.sin(x), np.cos(y), np.sin(z)


class JerkCircuit(DynSys):
    @staticjit
    def _rhs(x, y, z, t, eps, y0):
        xdot = y
        ydot = z
        zdot = -z - x - eps * (np.exp(y / y0) - 1)
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, eps, y0):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, -eps * np.exp(y / y0) / y0, -1]
        return row1, row2, row3


class ForcedBrusselator(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, w):
        xdot = a + x ** 2 * y - (b + 1) * x + f * np.cos(z)
        ydot = b * x - x ** 2 * y
        zdot = w
        return xdot, ydot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class WindmiReduced(DynSys):
    @staticjit
    def _rhs(i, v, p, t, a1, b1, b2, b3, d1, vsw):
        idot = a1 * (vsw - v)
        vdot = b1 * i - b2 * p ** 1 / 2 - b3 * v
        pdot = (
            vsw ** 2 - p ** (5 / 4) * vsw ** (1 / 2) * (1 + np.tanh(d1 * (i - 1))) / 2
        )
        return idot, vdot, pdot


class MooreSpiegel(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, eps):
        xdot = y
        ydot = a * z
        zdot = -z + eps * y - y * x ** 2 - b * x
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, eps):
        row1 = [0, 1, 0]
        row2 = [0, 0, a]
        row3 = [-2 * x * y - b, eps - x**2, -1]
        return row1, row2, row3


class CoevolvingPredatorPrey(DynSys):
    @staticjit
    def _rhs(x, y, alpha, t, a1, a2, a3, b1, b2, d1, d2, delta, k1, k2, k4, vv):
        xdot = x * (
            -((a3 * y) / (1 + b2 * x))
            + (a1 * alpha * (1 - k1 * x * (-alpha + alpha * delta))) / (1 + b1 * alpha)
            - d1
            * (
                1
                - k2 * (-(alpha ** 2) + (alpha * delta) ** 2)
                + k4 * (-(alpha ** 4) + (alpha * delta) ** 4)
            )
        )
        ydot = (-d2 + (a2 * x) / (1 + b2 * x)) * y
        alphadot = vv * (
            -((a1 * k1 * x * alpha * delta) / (1 + b1 * alpha))
            - d1 * (-2 * k2 * alpha * delta ** 2 + 4 * k4 * alpha ** 3 * delta ** 4)
        )
        return xdot, ydot, alphadot


class KawczynskiStrizhak(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, gamma, kappa, mu):
        xdot = gamma * y - gamma * x ** 3 + 3 * mu * gamma * x
        ydot = -2 * mu * x - y - z + beta
        zdot = kappa * x - kappa * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, beta, gamma, kappa, mu):
        row1 = [-3 * gamma * x ** 2 + 3 * mu * gamma, gamma, 0]
        row2 = [-2 * mu, -1, -1]
        row3 = [kappa, 0, -kappa]
        return row1, row2, row3


class BelousovZhabotinsky(DynSys):
    @staticjit
    def _rhs(
        x,
        z,
        v,
        t,
        c1,
        c10,
        c11,
        c12,
        c13,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        ci,
        kf,
        t0,
        y0,
        yb1,
        yb2,
        yb3,
        z0,
    ):
        ybar = (1 / y0) * yb1 * z * v / (yb2 * x + yb3 + kf)
        if x < 0.0:
            x = 0
        rf = (ci - z0 * z) * np.sqrt(x)
        xdot = c1 * x * ybar + c2 * ybar + c3 * x ** 2 + c4 * rf + c5 * x * z - kf * x
        zdot = (c6 / z0) * rf + c7 * x * z + c8 * z * v + c9 * z - kf * z
        vdot = c10 * x * ybar + c11 * ybar + c12 * x ** 2 + c13 * z * v - kf * v
        return xdot * t0, zdot * t0, vdot * t0


class IsothermalChemical(DynSys):
    @staticmethod
    def _rhs(alpha, beta, gamma, t, delta, kappa, mu, sigma):
        alphadot = mu * (kappa + gamma) - alpha * beta ** 2 - alpha
        betadot = (alpha * beta ** 2 + alpha - beta) / sigma
        gammadot = (beta - gamma) / delta
        return alphadot, betadot, gammadot


class VallisElNino(DynSys):
    @staticmethod
    def _rhs(x, y, z, t, b, c, p):
        xdot = b * y - c * x - c * p
        ydot = -y + x * z
        zdot = -z - x * y + 1
        return xdot, ydot, zdot


class RabinovichFabrikant(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, g):
        xdot = y * z - y + y * x ** 2 + g * x
        ydot = 3 * x * z + x - x ** 3 + g * y
        zdot = -2 * a * z  - 2 * x * y * z
        return (xdot, ydot, zdot)


class NoseHoover(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = -x + y * z
        zdot = a - y ** 2
        return xdot, ydot, zdot


class Dadras(DynSys):
    @staticjit
    def _rhs(x, y, z, t, c, e, o, p, r):
        xdot = y - p * x + o * y * z
        ydot = r * y - x * z + z
        zdot = c * x * y - e * z
        return xdot, ydot, zdot


class RikitakeDynamo(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, mu):
        xdot = -mu * x + y * z
        ydot = -mu * y - a * x + x * z
        zdot = 1 - x * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, mu):
        row1 = [-mu, z, y]
        row2 = [-a + z, -mu, x]
        row3 = [-y, -x, 0]
        return row1, row2, row3


class NuclearQuadrupole(DynSys):
    @staticjit
    def _rhs(q1, q2, p1, p2, t, a, b, d):
        q1dot = a * p1
        q2dot = a * p2
        p1dot = - a * q1 + 3 / np.sqrt(2) * b * q1 ** 2 - 3 / np.sqrt(2) * b * q2 ** 2 - d * q1 ** 3 - d * q1 * q2 ** 2
        p2dot = -a * q2 - 3 * np.sqrt(2) * b * q1 * q2 - d * q2 * q1 ** 2 - d * q2 ** 3
        return q1dot, q2dot, p1dot, p2dot


class PehlivanWei(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y - y * z
        ydot = y + y * z - 2 * x
        zdot = 2 - x * y - y ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, 1 - z, -y]
        row2 = [-2, 1 + z, y]
        row3 = [-y, -x - 2 * y, 0]
        return row1, row2, row3


class SprottTorus(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y + 2 * x * y + x * z
        ydot = 1 - 2 * x ** 2 + y * z
        zdot = x - x ** 2 - y ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [2 * y + z, 2 * x + 1, x]
        row2 = [-4 * x, z, y]
        row3 = [1 - 2 * x, -2 * y, 0]
        return row1, row2, row3


class SprottJerk(DynSys):
    @staticjit
    def _rhs(x, y, z, t, mu):
        xdot = y
        ydot = z
        zdot = -x + y ** 2 - mu * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, mu):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, 2 * y, -mu]
        return row1, row2, row3


## Not chaotic
# class JerkCircuit(DynSys):
#     def rhs(self, X, t):
#         x, y, z = X
#         xdot = y
#         ydot = z
#         zdot = -z - x  - self.eps*(np.exp(y/self.y0) - 1)
#         return (xdot, ydot, zdot)


class SprottA(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y
        ydot = -x + y * z
        zdot = 1 - y ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, 1, 0]
        row2 = [-1, z, y]
        row3 = [0, -2 * y, 0]
        return row1, row2, row3


class SprottB(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x - y
        zdot = 1 - x * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-y, -x, 0]
        return row1, row2, row3


class SprottC(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x - y
        zdot = 1 - x ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-2 * x, 0, 0]
        return row1, row2, row3


class SprottD(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -y
        ydot = x + z
        zdot = x * z + 3 * y ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, -1, 0]
        row2 = [1, 0, 1]
        row3 = [z, 6 * y, x]
        return row1, row2, row3


class SprottE(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x ** 2 - y
        zdot = 1 - 4 * x
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, z, y]
        row2 = [2 * x, -1, 0]
        row3 = [-4, 0, 0]
        return row1, row2, row3


class SprottF(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y + z
        ydot = -x + a * y
        zdot = x ** 2 - z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 1]
        row2 = [-1, a, 0]
        row3 = [2 * x, 0, -1]
        return row1, row2, row3


class SprottG(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = a * x + z
        ydot = x * z - y
        zdot = -x + y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [a, 0, 1]
        row2 = [z, -1, x]
        row3 = [-1, 1, 0]
        return row1, row2, row3


class SprottH(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -y + z ** 2
        ydot = x + a * y
        zdot = x - z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, -1, 2 * z]
        row2 = [1, a, 0]
        row3 = [1, 0, -1]
        return row1, row2, row3


class SprottI(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -a * y
        ydot = x + z
        zdot = x + y ** 2 - z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, -a, 0]
        row2 = [1, 0, 1]
        row3 = [1, 2 * y, -1]
        return row1, row2, row3


class SprottJ(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = 2 * z
        ydot = -2 * y + z
        zdot = -x + y + y ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, 0, 2]
        row2 = [0, -2, 1]
        row3 = [-1, 1 + 2 * y, 0]
        return row1, row2, row3
    


class SprottK(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = x * y - z
        ydot = x - y
        zdot = x + a * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [y, x, -1]
        row2 = [1, -1, 0]
        row3 = [1, 0, a]
        return row1, row2, row3


class SprottL(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = y + b * z
        ydot = a * x ** 2 - y
        zdot = 1 - x
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, 1, b]
        row2 = [2 * a * x, -1, 0]
        row3 = [-1, 0, 0]
        return row1, row2, row3


class SprottM(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -z
        ydot = -x ** 2 - y
        zdot = a + a * x + y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 0, -1]
        row2 = [-2 * x, -1, 0]
        row3 = [a, 1, 0]
        return row1, row2, row3


class SprottN(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -2 * y
        ydot = x + z ** 2
        zdot = 1 + y - 2 * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [0, -2, 0]
        row2 = [1, 0, 2 * z]
        row3 = [0, 1, -2]
        return row1, row2, row3


class SprottO(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = x - z
        zdot = x + x * z + a * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, 1, 0]
        row2 = [1, 0, -1]
        row3 = [1 + z, a, x]
        return row1, row2, row3


class SprottP(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = a * y + z
        ydot = -x + y ** 2
        zdot = x + y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [0, a, 1]
        row2 = [-1, 2 * y, 0]
        row3 = [1, 1, 0]
        return row1, row2, row3


class SprottQ(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -z
        ydot = x - y
        zdot = a * x + y ** 2 + b * z
        return (xdot, ydot, zdot)
    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, 0, -1]
        row2 = [1, -1, 0]
        row3 = [a, 2 * y, b]
        return row1, row2, row3


class SprottR(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = a - y
        ydot = b + z
        zdot = x * y - z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, -1, 0]
        row2 = [0, 0, 1]
        row3 = [y, x, -1]
        return row1, row2, row3


class SprottS(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -x - 4 * y
        ydot = x + z ** 2
        zdot = 1 + x
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t):
        row1 = [-1, -4, 0]
        row2 = [1, 0, 2 * z]
        row3 = [1, 0, 0]
        return row1, row2, row3


class SprottMore(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y
        ydot = -x - np.sign(z) * y
        zdot = y ** 2 - np.exp(-(x ** 2))
        return xdot, ydot, zdot
    # @staticjit
    # def _jac(x, y, z, t):
    #     row1 = [0, 1, 0]
    #     row2 = [-1, -np.sign(z), 0]
    #     row3 = [-2 * x * np.exp(-x ** 2), 2 * y, 0]
    #     return row1, row2, row3


class Arneodo(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = y
        ydot = z
        zdot = -a * x - b * y - c * z + d * x ** 3
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c, d):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-a + 3 * d * x ** 2, -b, -c]
        return row1, row2, row3


class Coullet(Arneodo):
    pass


class Rucklidge(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x + b * y - y * z
        ydot = x
        zdot = -z + y ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [-a, b - z, -y]
        row2 = [1, 0, 0]
        row3 = [0, 2 * y, -1]
        return row1, row2, row3


class Sakarya(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, h, p, q, r, s):
        xdot = a * x + h * y + s * y * z
        ydot = -b * y - p * x + q * x * z
        zdot = c * z - r * x * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c, h, p, q, r, s):
        row1 = [a, h + s * z, s * y]
        row2 = [-p + q * z, -b, q * x]
        row3 = [-r * y, -r * x, c]
        return row1, row2, row3


class LiuChen(Sakarya):
    pass


class RayleighBenard(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, r):
        xdot = a * y - a * x
        ydot = r * y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, r):
        row1 = [-a, a, 0]
        row2 = [-z, r, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class Finance(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = (1 / b - a) * x + z + x * y
        ydot = -b * y - x ** 2
        zdot = -x - c * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [(1 / b - a) + y, x, 1]
        row2 = [-2 * x, -b, 0]
        row3 = [-1, 0, -c]
        return row1, row2, row3


class Bouali2(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, bb, c, g, m, y0):
        xdot = a * y0 * x - a * x * y - b * z
        ydot = -g * y + g * y * x ** 2
        zdot = -1.5 * m * x + m * bb * x * z - c * z
        return xdot, ydot, zdot
    # @staticjit
    # def _jac(x, y, z, t, a, b, bb, c, g, m, y0):
    #     row1 = [a * y0 - a * y, -a * x, -b]
    #     row2 = [2 * g * x, g * x ** 2 - g, 0]
    #     row3 = [-1.5 * m + m * bb * z, 0, -c + m * bb * x]
    #     return row1, row2, row3


class Bouali(Bouali2):
    pass


class LuChenCheng(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = -(a * b) / (a + b) * x - y * z + c
        ydot = a * y + x * z
        zdot = b * z + x * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-(a * b) / (a + b), -z, -y]
        row2 = [z, a, x]
        row3 = [y, x, b]
        return row1, row2, row3


class LuChen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * y - a * x
        ydot = -x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-a, a, 0]
        row2 = [-z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class QiChen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * y - a * x + y * z
        ydot = c * x + y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-a, a + z, y]
        row2 = [c - z, 1, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ZhouChen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, e):
        xdot = a * x + b * y + y * z
        ydot = c * y - x * z + d * y * z
        zdot = e * z - x * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c, d, e):
        row1 = [a, b + z, y]
        row2 = [-z, c + d * z, -x + d * y]
        row3 = [-y, -x, e]
        return row1, row2, row3


class BurkeShaw(DynSys):
    @staticjit
    def _rhs(x, y, z, t, e, n):
        xdot = -n * x - n * y
        ydot = y - n * x * z
        zdot = n * x * y + e
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, e, n):
        row1 = [-n, -n, 0]
        row2 = [-n * z, 1, -n * x]
        row3 = [n * y, n * x, 0]
        return row1, row2, row3


class Chen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = x * y - b * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [-a, a, 0]
        row2 = [c - a - z, c, -x]
        row3 = [y, x, -b]
        return row1, row2, row3


class ChenLee(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = a * x - y * z
        ydot = b * y + x * z
        zdot = c * z + 0.3333333333333333333333333 * x * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [a, -z, -y]
        row2 = [z, b, x]
        row3 = [0.3333333333333333333333333 * y, 0.3333333333333333333333333 * x, c]
        return row1, row2, row3


class WangSun(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, d, e, f, q):
        xdot = a * x + q * y * z
        ydot = b * x + d * y - x * z
        zdot = e * z + f * x * y
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, d, e, f, q):
        row1 = [a, q * z, q * y]
        row2 = [b - z, d, -x]
        row3 = [f * y, f * x, e]
        return row1, row2, row3


class YuWang(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = np.exp(x * y) - d * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c, d):
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * np.exp(x * y), x * np.exp(x * y), -d]
        return row1, row2, row3


class YuWang2(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = a * (y - x)
        ydot = b * x - c * x * z
        zdot = np.cosh(x * y) - d * z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c, d):
        row1 = [-a, a, 0]
        row2 = [b - c * z, 0, -c * x]
        row3 = [y * np.sinh(x * y), x * np.sinh(x * y), -d]
        return row1, row2, row3


class SanUmSrisuchinwong(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y - x
        ydot = -z * np.tanh(x)
        zdot = -a + x * y + np.abs(y)
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a):
        row1 = [-1, 1, 0]
        row2 = [-z * (1 - np.tanh(x) ** 2), 0, -np.tanh(x)]
        row3 = [y, x + np.sign(y), 0]
        return row1, row2, row3


class DequanLi(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, c, d, eps, f, k):
        xdot = a * y - a * x + d * x * z
        ydot = k * x + f * y - x * z
        zdot = c * z + x * y - eps * x ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, c, d, eps, f, k):
        row1 = [-a + d * z, a, d * x]
        row2 = [k - z, f, -x]
        row3 = [y - 2 * eps * x, x, c]
        return row1, row2, row3


class PanXuZhou(DequanLi):
    pass


class Tsucs2(DequanLi):
    pass


class ArnoldWeb(DynSys):
    @staticjit
    def _rhs(p1, p2, x1, x2, z, t, mu, w):
        denom = 4 + np.cos(z) + np.cos(x1) + np.cos(x2)
        p1dot = -mu * np.sin(x1) / denom ** 2
        p2dot = -mu * np.sin(x2) / denom ** 2
        x1dot = p1
        x2dot = p2
        zdot = w
        return p1dot, p2dot, x1dot, x2dot, zdot
    # @staticjit
    # def _jac(p1, p2, x1, x2, z, t, mu, w):
    #     row1 = [-mu * np.cos(x1) / (4 + np.cos(z) + np.cos(x1) + np.cos(x2)) ** 2, 0, 0, 0, 0]
    #     row2 = [0, -mu * np.cos(x2) / (4 + np.cos(z) + np.cos(x1) + np.cos(x2)) ** 2, 0, 0, 0]
    #     row3 = [1, 0, 0, 0, 0]
    #     row4 = [0, 1, 0, 0, 0]
    #     row5 = [0, 0, 0, 0, 0]
    #     return row1, row2, row3, row4, row5

    @staticjit
    def _postprocessing(p1, p2, x1, x2, z):
        return p1, p2, np.sin(x1), np.sin(x2), np.cos(z)


class NewtonLiepnik(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a * x + y + 10 * y * z
        ydot = -x - 0.4 * y + 5 * x * z
        zdot = b * z - 5 * x * y
        return xdot, ydot, zdot
    # @staticjit
    # def _jac(x, y, z, t, a, b):


class HyperRossler(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = -y - z
        ydot = x + a * y + w
        zdot = b + x * z
        wdot = -c * z + d * w
        return xdot, ydot, zdot, wdot


class HyperLorenz(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + w
        ydot = -x * z + c * x - y
        zdot = -b * z + x * y
        wdot = d * w - x * z
        return xdot, ydot, zdot, wdot


class HyperCai(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d, e):
        xdot = a * y - a * x
        ydot = b * x + c * y - x * z + w
        zdot = -d * z + y ** 2
        wdot = -e * x
        return xdot, ydot, zdot, wdot


class HyperBao(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d, e):
        xdot = a * y - a * x + w
        ydot = c * y - x * z
        zdot = x * y - b * z
        wdot = e * x + d * y * z
        return xdot, ydot, zdot, wdot


class HyperJha(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + w
        ydot = -x * z + b * x - y
        zdot = x * y - c * z
        wdot = -x * z + d * w
        return xdot, ydot, zdot, wdot


class HyperQi(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d, e, f):
        xdot = a * y - a * x + y * z
        ydot = b * x + b * y - x * z
        zdot = -c * z - e * w + x * y
        wdot = -d * w + f * z + x * y
        return xdot, ydot, zdot, wdot


class Qi(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + y * z * w
        ydot = b * x + b * y - x * z * w
        zdot = -c * z + x * y * w
        wdot = -d * w + x * y * z
        return xdot, ydot, zdot, wdot


class LorenzStenflo(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a, b, c, d):
        xdot = a * y - a * x + d * w
        ydot = c * x - x * z - y
        zdot = x * y - b * z
        wdot = -x - a * w
        return xdot, ydot, zdot, wdot


class HyperYangChen(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=30, b=3, c=35, d=8):
        xdot = a * y - a * x
        ydot = c * x - x * z + w
        zdot = -b * z + x * y
        wdot = -d * x
        return xdot, ydot, zdot, wdot


class HyperYan(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=37, b=3, c=26, d=38):
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = -b * z + x * y - y * z + x * z - w
        wdot = -d * w + y * z - x * z
        return xdot, ydot, zdot, wdot


class HyperXu(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=10, b=40, c=2.5, d=2, e=16):
        xdot = a * y - a * x + w
        ydot = b * x + e * x * z
        zdot = -c * z - x * y
        wdot = x * z - d * y
        return xdot, ydot, zdot, wdot


class HyperWang(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=10, b=40, c=2.5, d=10.6, e=4):
        xdot = a * y - a * x
        ydot = -x * z + b * x + w
        zdot = -c * z + e * x ** 2
        wdot = -d * x
        return xdot, ydot, zdot, wdot


class HyperPang(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=36, b=3, c=20, d=2):
        xdot = a * y - a * x
        ydot = -x * z + c * y + w
        zdot = x * y - b * z
        wdot = -d * x - d * y
        return xdot, ydot, zdot, wdot


class HyperLu(DynSys):
    @staticjit
    def _rhs(x, y, z, w, t, a=36, b=3, c=20, d=1.3):
        xdot = a * y - a * x + w
        ydot = -x * z + c * y
        zdot = x * y - b * z
        wdot = d * w + x * z
        return xdot, ydot, zdot, wdot


class SaltonSea(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, d, k, lam, m, mu, r, th):
        xdot = r * x * (1 - (x + y) / k) - lam * x * y
        ydot = lam * x * y - m * y * z / (y + a) - mu * y
        zdot = th * y * z / (y + a) - d * z
        return xdot, ydot, zdot


class ExcitableCell(DynSys):
    def rhs(self, X, t):
        v, n, c = X

        alpham = 0.1 * (25 + v) / (1 - np.exp(-0.1 * v - 2.5))
        betam = 4 * np.exp(-(v + 50) / 18)
        minf = alpham / (alpham + betam)

        alphah = 0.07 * np.exp(-0.05 * v - 2.5)
        betah = 1 / (1 + np.exp(-0.1 * v - 2))
        hinf = alphah / (alphah + betah)

        alphan = 0.01 * (20 + v) / (1 - np.exp(-0.1 * v - 2))
        betan = 0.125 * np.exp(-(v + 30) / 80)
        ninf = alphan / (alphan + betan)
        tau = 1 / (230 * (alphan + betan))

        ca = c / (1 + c)

        vdot = (
            self.gi * minf ** 3 * hinf * (self.vi - v)
            + self.gkv * n ** 4 * (self.vk - v)
            + self.gkc * ca * (self.vk - v)
            + self.gl * (self.vl - v)
        )
        ndot = (ninf - n) / tau
        cdot = self.rho * (minf ** 3 * hinf * (self.vc - v) - self.kc * c)
        return vdot, ndot, cdot


class CaTwoPlus(DynSys):
    def rhs(self, X, t):
        z, y, a = X
        Vin = self.V0 + self.V1 * self.beta
        V2 = self.Vm2 * (z ** 2) / (self.K2 ** 2 + z ** 2)
        V3 = (
            (self.Vm3 * (z ** self.m) / (self.Kz ** self.m + z ** self.m))
            * (y ** 2 / (self.Ky ** 2 + y ** 2))
            * (a ** 4 / (self.Ka ** 4 + a ** 4))
        )
        V5 = (
            self.Vm5
            * (a ** self.p / (self.K5 ** self.p + a ** self.p))
            * (z ** self.n / (self.Kd ** self.n + z ** self.n))
        )
        zdot = Vin - V2 + V3 + self.kf * y - self.k * z
        ydot = V2 - V3 - self.kf * y
        adot = self.beta * self.V4 - V5 - self.eps * a
        return (zdot, ydot, adot)


class CellCycle(DynSys):
    def rhs(self, X, t):
        c1, m1, x1, c2, m2, x2 = X
        Vm1, Um1 = 2 * [self.Vm1]
        vi1, vi2 = 2 * [self.vi]
        H1, H2, H3, H4 = 4 * [self.K]
        K1, K2, K3, K4 = 4 * [self.K]
        V2, U2 = 2 * [self.V2]
        Vm3, Um3 = 2 * [self.Vm3]
        V4, U4 = 2 * [self.V4]
        Kc1, Kc2 = 2 * [self.Kc]
        vd1, vd2 = 2 * [self.vd]
        Kd1, Kd2 = 2 * [self.Kd1]
        kd1, kd2 = 2 * [self.kd1]
        Kim1, Kim2 = 2 * [self.Kim]
        V1 = Vm1 * c1 / (Kc1 + c1)
        U1 = Um1 * c2 / (Kc2 + c2)
        V3 = m1 * Vm3
        U3 = m2 * Um3
        c1dot = vi1 * Kim1 / (Kim1 + m2) - vd1 * x1 * c1 / (Kd1 + c1) - kd1 * c1
        c2dot = vi2 * Kim2 / (Kim2 + m1) - vd2 * x2 * c2 / (Kd2 + c2) - kd2 * c2
        m1dot = V1 * (1 - m1) / (K1 + (1 - m1)) - V2 * m1 / (K2 + m1)
        m2dot = U1 * (1 - m2) / (H1 + (1 - m2)) - U2 * m2 / (H2 + m2)
        x1dot = V3 * (1 - x1) / (K3 + (1 - x1)) - V4 * x1 / (K4 + x1)
        x2dot = U3 * (1 - x2) / (H3 + (1 - x2)) - U4 * x2 / (H4 + x2)
        return c1dot, m1dot, x1dot, c2dot, m2dot, x2dot


class CircadianRhythm(DynSys):
    @staticjit
    def _rhs(
        m,
        fc,
        fs,
        fn,
        th,
        t,
        Ki,
        k,
        k1,
        k2,
        kd,
        kdn,
        km,
        ks,
        n,
        vd,
        vdn,
        vm,
        vmax,
        vmin,
        v,
    ):
        vs = 2.5 * ((0.5 + 0.5 * np.cos(th)) + vmin) * (vmax - vmin)
        mdot = vs * (Ki ** n) / (Ki ** n + fn ** n) - vm * m / (km + m)
        fcdot = ks * m - k1 * fc + k2 * fn - k * fc
        fsdot = k * fc - vd * fs / (kd + fs)
        fndot = k1 * fc - k2 * fn - vdn * fn / (kdn + fn)
        thdot = 2 * np.pi / 24
        return mdot, fcdot, fsdot, fndot, thdot

    @staticjit
    def _postprocessing(m, fc, fs, fn, th):
        return m, fc, fs, fn, np.cos(th)


class FluidTrampoline(DynSys):
    @staticmethod
    def _rhs(x, y, th, t, gamma, psi, w):
        xdot = y
        ydot = -1 - np.heaviside(-x, 0) * (x + psi * y * np.abs(y)) + gamma * np.cos(th)
        thdot = w
        return (xdot, ydot, thdot)

    @staticjit
    def _postprocessing(x, y, th):
        return x, y, np.cos(th)


class Aizawa(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, e, f):
        xdot = x * z - b * x - d * y
        ydot = d * x + y * z - b * y
        zdot = c + a * z - 0.333333333333333333 * z ** 3 - x ** 2 - y ** 2 - e * z * x ** 2 - e * z * y ** 2 + f * z * x ** 3
        return xdot, ydot, zdot
    # @staticjit
    # def _jac(x, y, z, t, a, b, c, d, e, f):
    #     xdot = x * z - b * x - d * y
    #     ydot = d * x + y * z - b * y
    #     zdot = c + a * z - 0.333333333333333333 * z ** 3 - x ** 2 - y ** 2 - e * z * x ** 2 - e * z * y ** 2 + f * z * x ** 3
    #     row1 = [z - b, -d, x]
    #     row2 = [d, z - b, y]
    #     row3 = [- 2 * x - 2 * e * z + 3 * f * z * x ** 2, - 2 * y
    #     return row1, row2, row3


class AnishchenkoAstakhov(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        mu, eta = self.mu, self.eta
        xdot = mu * x + y - x * z
        ydot = -x
        zdot = -eta * z + eta * np.heaviside(x, 0) * x ** 2
        return (xdot, ydot, zdot)


class ShimizuMorioka(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = y
        ydot = x - a * y - x * z
        zdot = -b * z + x ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b):
        row1 = [0, 1, 0]
        row2 = [1 - z, -a, -x]
        row3 = [2 * x, 0, -b]
        return row1, row2, row3


class GenesioTesi(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = y
        ydot = z
        zdot = -c * x - b * y - a * z + x ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c):
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-c + 2 * x, -b, -a]
        return row1, row2, row3


class AtmosphericRegime(DynSys):
    @staticjit
    def _rhs(
        x, y, z, t, alpha, beta, mu1, mu2, omega, sigma
    ):
        xdot = mu1 * x + sigma * x * y
        ydot = mu2 * y + omega * z + alpha * y * z + beta * z ** 2 - sigma * x ** 2
        zdot = mu2 * z - omega * y - alpha * y ** 2 - beta * y * z
        return xdot, ydot, zdot


class Hadley(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, g):
        xdot = -y ** 2 - z ** 2 - a * x + a * f
        ydot = x * y - b * x * z - y + g
        zdot = b * x * y + x * z - z
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, f, g):
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, x - 1]
        return row1, row2, row3
    


class ForcedVanDerPol(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, mu, w):
        ydot = mu * (1 - x ** 2) * y - x + a * np.sin(z)
        xdot = y
        zdot = w
        return xdot, ydot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class ForcedFitzHughNagumo(DynSys):
    @staticjit
    def _rhs(v, w, z, t, a, b, curr, f, gamma, omega):
        vdot = v - v ** 3 / 3 - w + curr + f * np.sin(z)
        wdot = gamma * (v + a - b * w)
        zdot = omega
        return vdot, wdot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class HindmarshRose(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, s, tx, tz):
        xdot = -x + 1 / tx * y - a / tx * x ** 3 + b / tx * x ** 2 + 1 / tx * z
        ydot = -a * x ** 3 - (d - b) * x ** 2 + z
        zdot = -s / tz * x - 1 / tz * z + c / tz
        return xdot, ydot, zdot
    # @staticjit
    # def _jac(x, y, z, t, a, b, c, d, s, tx, tz):
    #     row1 = [-1 / tx - 3 * a / tx * x ** 2 + 2 * b / tx * x, 1 / tx, 1 / tx]
    #     row2 = [-3 * a * x ** 2 - 2 * (d - b) * x, 0, 1]
    #     row3 = [-s / tz, 0, -1 / tz - c / tz]
    #     return row1, row2, row3


class Colpitts(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        u = z - (self.e - 1)
        fz = -u * (1 - np.heaviside(u, 0))
        xdot = y - self.a * fz
        ydot = self.c - x - self.b * y - z
        zdot = y - self.d * z
        return (xdot, ydot, zdot)


class Laser(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d, h, k):
        xdot = a * y - a * x + b * y * z ** 2
        ydot = c * x + d * x * z ** 2
        zdot = h * z + k * x ** 2
        return xdot, ydot, zdot
    @staticjit
    def _jac(x, y, z, t, a, b, c, d, h, k):
        row1 = [-a, a + b * z ** 2, 2 * b * y * z]
        row2 = [c + d * z ** 2, 0, 2 * d * x * z]
        row3 = [2 * k * x, 0, h]
        return row1, row2, row3


class Blasius(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, alpha1, alpha2, b, c, k1, k2, zs):
        xdot = a * x - alpha1 * x * y / (1 + k1 * x)
        ydot = -b * y + alpha1 * x * y / (1 + k1 * x) - alpha2 * y * z / (1 + k2 * y)
        zdot = -c * (z - zs) + alpha2 * y * z / (1 + k2 * y)
        return xdot, ydot, zdot


class TurchinHanski(DynSys):
    @staticjit
    def _rhs(n, p, z, t, a, d, e, g, h, r, s):
        ndot = (
            r * (1 - e * np.sin(z)) * n
            - r * (n ** 2)
            - g * (n ** 2) / (n ** 2 + h ** 2)
            - a * n * p / (n + d)
        )
        pdot = s * (1 - e * np.sin(z)) * p - s * (p ** 2) / n
        zdot = 2 * np.pi
        return ndot, pdot, zdot

    @staticjit
    def _postprocessing(x, y, z):
        return x, y, np.sin(z)


class StickSlipOscillator(DynSys):
    def _t(self, v):
        return self.t0 * np.sign(v) - self.alpha * v + self.beta * v ** 3

    @staticjit
    def _rhs(x, v, th, t, a, alpha, b, beta, eps, gamma, t0, vs, w):
        tq = t0 * np.sign(v - vs) - alpha * v + beta * (v - vs) ** 3
        xdot = v
        vdot = eps * (gamma * np.cos(th) - tq) + a * x - b * x ** 3
        thdot = w
        return xdot, vdot, thdot

    @staticjit
    def _postprocessing(x, v, th):
        return x, v, np.cos(th)


class HastingsPowell(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a1, a2, b1, b2, d1, d2):
        xdot = x * (1 - x) - y * a1 * x / (1 + b1 * x)
        ydot = y * a1 * x / (1 + b1 * x) - z * a2 * y / (1 + b2 * y) - d1 * y
        zdot = z * a2 * y / (1 + b2 * y) - d2 * z
        return xdot, ydot, zdot


class CellularNeuralNetwork(DynSys):
    @staticjit
    def f(x):
        return 0.5 * (np.abs(x + 1) - np.abs(x - 1))

    def rhs(self, X, t):
        x, y, z = X
        xdot = -x + self.d * self.f(x) - self.b * self.f(y) - self.b * self.f(z)
        ydot = -y - self.b * self.f(x) + self.c * self.f(y) - self.a * self.f(z)
        zdot = -z - self.b * self.f(x) + self.a * self.f(y) + self.f(z)
        return (xdot, ydot, zdot)


class BeerRNN(DynSys):
    @staticjit
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def rhs(self, X, t):
        Xdot = (-X + np.matmul(self.w, self._sig(X + self.theta))) / self.tau
        return Xdot


class Torus(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, n, r):
        xdot = (-a * n * np.sin(n * t)) * np.cos(t) - (r + a * np.cos(n * t)) * np.sin(
            t
        )
        ydot = (-a * n * np.sin(n * t)) * np.sin(t) + (r + a * np.cos(n * t)) * np.cos(
            t
        )
        zdot = a * n * np.cos(n * t)
        return xdot, ydot, zdot


class CaTwoPlusQuasiperiodic(CaTwoPlus):
    pass


class Hopfield(DynSys):
    def f(self, x):
        return (1 + np.tanh(x)) / 2

    def rhs(self, X, t):
        Xdot = -X / self.tau + self.f(self.eps * np.matmul(self.k, X)) - self.beta
        return Xdot


class MacArthur(DynSys):
    def growth_rate(self, rr):
        u0 = rr / (self.k.T + rr)
        u = self.r * u0.T
        return np.min(u.T, axis=1)

    def rhs(self, X, t):
        nn, rr = X[:5], X[5:]
        mu = self.growth_rate(rr)
        nndot = nn * (mu - self.m)
        rrdot = self.d * (self.s - rr) - np.matmul(self.c, (mu * nn))
        return np.hstack([nndot, rrdot])


class ItikBanksTumor(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a12, a13, a21, a31, d3, k3, r2, r3):
        xdot = x * (1 - x) - a12 * x * y - a13 * x * z
        ydot = r2 * y * (1 - y) - a21 * x * y
        zdot = r3 * x * z / (x + k3) - a31 * x * z - d3 * z
        return xdot, ydot, zdot


## Doesn't match described dynamics
# class CosmologyFriedmann(DynSys):
#     @staticjit
#     def _rhs(x, y, z, t, a, b, c, d, p):
#         xdot = y
#         ydot = -a * y**2 / x - b * x - c * x**3 + d * p * x
#         zdot = 3 * (y / x) * (p + z)
#         return xdot, ydot, zdot


## Doesn't match described dynamics
# class MixMasterUniverse(DynSys):
#     def rhs(self, X, t):
#         a, b, g, adot_, bdot_, gdot_ = X
#         adot = adot_
#         bdot = bdot_
#         gdot = gdot_
#         addot = (np.exp(2*b) - np.exp(2*g))**2 - np.exp(4*a)
#         bddot = (np.exp(2*g) - np.exp(2*a))**2 - np.exp(4*b)
#         gddot = (np.exp(2*a) - np.exp(2*b))**2 - np.exp(4*g)
#         return (adot, bdot, gdot, addot, bddot, gddot)

## Doesn't match described dynamics
# class Universe(DynSys):
#     def rhs(self, X, t):
#         #Xdot = X * np.matmul(self.a, 1 - X)
#         Xdot = self.r * X * (1 - np.matmul(self.a, X))
#         return Xdot

# class SeasonalSEIR:
#     """
#     This is extremely unstable for some reason
#     Olsen, Schaffer. Science 1990

#     eq = SeasonalSEIR()
#     tpts = np.linspace(0, 1000, 100000)
#     ic = (.579, .02, .001, 1e-6)
#     sol = integrate_dyn(eq, ic, tpts)

#     plt.plot(sol[0], sol[2], '.k', markersize=.1)

#     """
#     def __init__(self):
#         pass

#     def __call__(self, X, t):
#         (s, e, i, th) = X

#         ## measles
#         m = 0.02
#         a = 35.84
#         g = 100
#         b0 = 1800
#         b1 = 0.28

# #         ## chickenpox
# #         m = 0.02
# #         a = 36
# #         g = 34.3
# #         b0 = 537
# #         b1 = 0.3

#         b = b0*(1 + b1*np.cos(th))
#         sdot = m*(1 - s) - b*s*i
#         edot = b*s*i - (m + a)*e
#         idot = a*e - (m + g)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
#     Zhang, Q., Liu, C., & Zhang, X. (2012). Analysis and Control of an SEIR Epidemic System with Nonlinear Transmission Rate. Lecture Notes in Control and Information Sciences, 203–225. doi:10.1007/978-1-4471-2303-3_14
#     """
#     def __init__(self, b=0.02, beta0=1800, alpha=35.84, gamma=100.0, beta1=0.28):
#         self.b, self.beta0, self.alpha, self.gamma, self.beta1 = b, beta0, alpha, gamma, beta1
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         alpha is a
#         b is mu
#         """
#         s, e, i, th = X
#         beta = self.beta0*(1 + self.beta1*np.cos(th)) # seasonal forcing
#         sdot = self.b - self.b*s - beta*s*i
#         edot = beta*s*i - (self.alpha + self.b)*e
#         idot = self.alpha*e - (self.gamma + self.b)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
#     Zhang, Q., Liu, C., & Zhang, X. (2012). Analysis and Control of an SEIR Epidemic System with Nonlinear Transmission Rate. Lecture Notes in Control and Information Sciences, 203–225. doi:10.1007/978-1-4471-2303-3_14
#     """
#     def __init__(self, b=0.02, beta0=1800, alpha=35.84, gamma=100.0, beta1=0.28):
#         self.b, self.beta0, self.alpha, self.gamma, self.beta1 = b, beta0, alpha, gamma, beta1
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         alpha is a
#         b is mu
#         """
#         s, e, i, th = X
#         beta = self.beta0*(1 + self.beta1*np.cos(th)) # seasonal forcing
#         sdot = self.b - self.b*s - beta*s*i
#         edot = beta*s*i - (self.alpha + self.b)*e
#         idot = self.alpha*e - (self.gamma + self.b)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
#     """
#     def __init__(self, mu=0.02, b0=1800, a=35.84, g=100.0, d=0.28):
#         self.mu, self.b0, self.a, self.g, self.d = mu, b0, a, g, d
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         alpha is a
#         b is mu
#         """
#         s, e, i, th = X
#         b = self.b0*(1 + self.d*np.cos(th)) # seasonal forcing
#         sdot = self.mu - self.mu*s - b*s*i
#         edot = b*s*i - (self.mu + self.a)*e
#         idot = self.a*e - (self.mu + self.g)*i
# #         edot = 0
# #         idot = 0
#         thdot = 2*np.pi

#         return (sdot, edot, idot, thdot)

# class SeasonalSEIR:
#     """
#     Seasonally forced SEIR model
# Bifurcation analysis of periodic SEIR and SIR epidemic models
#     """
#     def __init__(self, mu=0.02, b0=1884.95*5, a=35.842, g=100.0, d=0.255):
#         self.mu, self.b0, self.a, self.g, self.d = mu, b0, a, g, d
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         s, e, i, th = X
#         b = self.b0*(1 + self.d*np.cos(th)) # seasonal forcing
#         sdot = self.mu - self.mu*s - b*s*i
#         edot = b*s*i - (self.mu + self.a)*e
#         idot = self.a*e - (self.mu + self.g)*i
#         thdot = 2*np.pi
#         return (sdot, edot, idot, thdot)


# class SeasonalSIR:
#     """
#     Seasonally forced SEIR model
#     """
#     def __init__(self, mu=0.02, b0=1884.95, a=35.842, g=100, d=0.255):
#         self.mu, self.b0, self.a, self.g, self.d = mu, b0, a, g, d
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         s, i, th = X
#         b = self.b0*(1 + self.d*np.sin(th)) # seasonal forcing
#         sdot = self.mu - self.mu*s - b*s*i
#         idot =  b*s*i - (self.mu + self.g)*i
#         thdot = 2*np.pi
#         return (sdot, idot, thdot)


# class Robinson:
#     """
#     C Robinson 1989 Nonlinearity
#     Was unable to find published parameters for which this has a stable attractor,
#     it may only be transient
#     """
#     def __init__(self, a=0.71, b=1.8587, v=1.0, gamma=0.7061, delta=0.1):
#         self.a, self.b, self.v, self.gamma, self.delta = a, b, v, gamma, delta
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         x, y, z = X
#         xdot = y
#         ydot = x - 2*x**3 - self.a*y + (self.b*x**2)*y - self.v*y*z
#         zdot = -self.gamma*z + self.delta*x**2
#         return (xdot, ydot, zdot)

# Sato: Cardiac model analogous to HH
# https://www.sciencedirect.com/science/article/pii/S1007570416300016
# hundreds of equations


# class HodgkinHuxley:
#     def __init__(self, i=7.92197):
#         self.i = i

#     def phi(self, x):
#         return x/(np.exp(x) - 1)

#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         v, m, n, h = X
#         i = self.i
#         vdot = i - (120*m**3*h*(v + 115) + 36*n**4*(v - 12) + 0.3*(v + 10.599))
#         mdot = (1 - m)*self.phi((v + 25)/10) - m*(4*np.exp(v/18))
#         ndot = (1 - n)*0.1*self.phi((v + 10)/10) - n*0.125*np.exp(v/80)
#         hdot = (1 - h)*0.07*np.exp(v/20) - h/(1 + np.exp((v + 30)/10))
#         return (vdot, mdot, ndot, hdot)

##############################
##
## Quasiperiodic systems
##
##############################

# class SymmetricKuramoto(DynSys):
#     def coupling(self, x, n=4):
#         k = 1 + np.arange(n)
#         return np.sum(self.a[:, None, None] * np.cos(k[:, None, None]*x[None, ...] - self.eta[:, None, None]), axis=0)
#     def rhs(self, X, t):
#         phase_diff = X[:, None] - X[None, :]
#         Xdot = self.w + np.mean(self.coupling(phase_diff), axis=0) ## need to apply coupling element-wise
#         return Xdot

#     "SymmetricKuramoto": {
#         "initial_conditions": [
#             0.1,
#             0.01,
#             0.01,
#             0.01
#         ],
#         "dt": 0.01,
#         "parameters": {
#             "w" : 0.0,
#             "a": [-2, -2, -1,  -0.88],
#             "eta": [0.1104, -0.1104, 0.669, 0.669]
#         },
#         "citation": "Bick, Christian, et al. Chaos in symmetric phase oscillator networks. Physical review letters 107.24 (2011): 244101.",
#         "period": 7737.7
#     }

# class InterfacialFlight:
#     """
#     """
#     def __init__(self):
#         pass
#     def __call__(self, X, t):
#         x, y, z = X
#         rclaw = 57 #um
#         m = 2.2
#         cl = 1.55
#         ly = 73

#         l0 = 137*1e6 # convert from uN/mg into um/s^2
#         f = 116
#         r = 0.15
#         phi0 = np.pi/2

#         cdleg = 3
#         rhow = 1e-9 # water density in mg/um^3

#         sigma = 72800 # water surface tension mg/s^2

#         kinv = 2709 # um
#         hclaw = rclaw*np.log(2*kinv/rclaw)

#         sech = lambda pp : 1 / np.cosh(pp)

#         phi_arg = 2*np.pi*f*t + phi0
#         sin_term = np.sin(phi_arg)
#         hsin_term = np.heaviside(sin_term, 0)
#         zdot = (l0/m)*np.cos(phi_arg)*(hsin_term + r*(1 - hsin_term))
#         zdot += -(8*np.pi*sigma*rclaw/m)*sech((x - hclaw)/rclaw)*np.sign(x)
#         zdot += -(2*rhow*cdleg*np.pi*rclaw**2/m)*y*np.sign(y)

#         xdot = 1
#         ydot = x
#         return (xdot, ydot, zdot)

## Not chaotic
# class Robinson(DynSys):
#     @staticjit
#     def _rhs(x, y, z, t, a, b, c, d, v):
#         xdot = y
#         ydot = x - 2 * x**3 - a * y + b * x**2 * y - v * y * z
#         zdot = -c * z + d * x**2
#         return (xdot, ydot, zdot)
