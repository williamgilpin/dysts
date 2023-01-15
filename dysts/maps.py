"""
Low-dimensional chaotic maps in Python

# (M, T, D) convention

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)
+ jax (optional, for faster integration)

Resources:
http://www.3d-meier.de/tut19/Seite1.html

DEV: 
+ Database: How to load a function object from a string? exec does not do this
+ Add a function compute_dt that finds the timestep based on fft
+ Set standard integration timestep based on this value
+ Add a function that rescales outputs to the same interval
+ Add a function that finds initial conditions on the attractor
"""

import numpy as np
from .base import DynMap, staticjit

class Logistic(DynMap):
    @staticjit
    def _rhs(x, r):
        return r * x * (1 - x)
    
class Ricker(DynMap):
    @staticjit
    def _rhs(x, a):
        return x * np.exp(a - x)
    
class Tent(DynMap):
    @staticjit
    def _rhs(x, mu):
        return mu * (1 - 2 * np.abs(x - 0.5))

class Gauss(DynMap):
    @staticjit
    def _rhs(x, a, b):
        return np.exp(-a * x**2) + b

class Chebyshev(DynMap):
    @staticjit
    def _rhs(x, a):
        return np.cos(a * np.arccos(x))
    
class Bogdanov(DynMap):
    @staticjit
    def _rhs(x, y, eps, k, mu):
        yp = (1 + eps) * y + k * x * (x - 1) + mu * x * y
        xp = x + yp
        return xp, yp

class Baker(DynMap):
    @staticjit
    def _rhs(x, y, a):
        eps2 = 2.0 - 1e-10
        x_flr = (eps2 * x) // 1
        xp = eps2 * x - x_flr
        yp = (a * y + x_flr) / 2
        return xp, yp
    
    @staticjit
    def _rhs_inv(xp, yp, a):
        ### Bug here
        eps2 = 2.0 - 1e-10
        if yp > 0.5:
            xflr = 0.5 + yp * a / 2
        else:
            xflr = yp * a / 2
        x = (xp + xflr) / eps2
        y = (2 * yp - xflr) / a
        return x, y 
    
class Circle(DynMap):
    @staticjit
    def _rhs(theta, k, omega):
        thetap = theta + omega + (k / (2 * np.pi)) * np.sin(2 * np.pi * theta)
        return thetap

class DeJong(DynMap):
    @staticjit
    def _rhs(x, y, a, b, c, d):
        xp = np.sin(a * y) - np.cos(b * x)
        yp = np.sin(c * x) - np.cos(d * y)
        return xp, yp

class Svensson(DynMap):
    @staticjit
    def _rhs(x, y, a, b, c, d):
        xp = d * np.sin(a * x) - np.sin(b * y)
        yp = c * np.cos(a * x) + np.cos(b * y)
        return xp, yp
    
class Bedhead(DynMap):
    @staticjit
    def _rhs(x, y, a, b):
        xp = np.sin(x * y / b) * y + np.cos(a * x - y)
        yp =  x + np.sin( y ) / b
        return xp, yp
    
# class ZeraouliaSprott(DynMap):
#     @staticjit
#     def _rhs(x, y, a, b):
#         xp = - a * x / (1 + y**2)
#         yp = x + b * y
#         return xp, yp
    
class GumowskiMira(DynMap):
    @staticjit
    def _rhs(x, y, a, b):
        fx = a * x + 2 * (1 - a) * x**2 / (1 + x**2)
        xp = b * y + fx
        fx1 = a * xp + 2 * (1 - a) * xp**2 / (1 + xp**2)
        yp = fx1 - x
        return xp, yp
    
class Hopalong(DynMap):
    @staticjit
    def _rhs(x, y, a, b, c):
        xp = y - 1 - np.sqrt(np.abs(b * x - 1 - c))*np.sign(x - 1)
        yp = a - x - 1
        return xp, yp
    
class Ikeda(DynMap):
    @staticjit
    def _rhs(x, y, u):
        t = 0.4 - 6 / (1 + x**2 + y**2)
        xp = 1 + u * (x * np.cos(t) - y * np.sin(t))
        yp = u * (x * np.sin(t) + y * np.cos(t))
        return xp, yp

class Tinkerbell(DynMap):
    @staticjit
    def _rhs(x, y, a, b, c, d):
        xp = x**2 - y**2 + a * x + b * y
        yp = 2 * x * y + c * x + d * y
        return xp, yp
    
class Pickover(DynMap):
    @staticjit
    def _rhs(x, y, a, b, c, d):
        xp = np.sin(a * y) + c * np.cos(a * x)
        yp = np.sin(b * x) + d * np.cos(b * y)
        return xp, yp
    
class MaynardSmith(DynMap):
    @staticjit
    def _rhs(x, y, a, b):
        xp = y
        yp = a * y + b - x**2
        return xp, yp

class KaplanYorke(DynMap):
    @staticjit
    def _rhs(x, y, alpha):
        xp = (2 * x) % 0.99999995
        yp = alpha * y + np.cos(4 * np.pi * x)
        return xp, yp

class Gingerbreadman(DynMap):
    @staticjit
    def _rhs(x, y):
        xp = 1 - y + np.abs(x)
        yp = x
        return xp, yp
    
class Duffing(DynMap):
    @staticjit
    def _rhs(x, y, a, b):
        xp = y
        yp = -b * x + a * y - y**3
        return xp, yp
    
# class Zaslavskii(DynMap):
#     @staticjit
#     def _rhs(x, y, eps, nu, r):
#         mu = (1 - np.exp(-r)) / r
#         xp = x + nu * (1 + mu * y) + eps * nu * mu * np.cos(2 * np.pi * x)
#         xp = xp % 0.99999995 
#         yp = np.exp(-r) * (y + eps * np.cos(2 * np.pi * x))
#         return xp, yp

    
class Chirikov(DynMap):
    @staticjit
    def _rhs(p, x, k):
        pp = p + k * np.sin(x)
        xp = x + pp 
        return pp, xp

    @staticjit
    def _rhs_inv(pp, xp, k):
        x = xp - pp
        p = pp - k * np.sin(xp - pp)
        return p, x

class Henon(DynMap):

    @staticjit
    def _rhs(x, y, a, b):
        xp = 1 - a * x**2 + y
        yp = b * x
        return xp, yp

    @staticjit
    def _rhs_inv(xp, yp, a, b):
        x = yp / b
        y = - xp - 1 + a * x**2
        return x, y



from scipy.optimize import fsolve
class BlinkingVortexMap(DynMap):
    
    def __post_init__(self):
        from scipy.optimize import fsolve
    
    def smoothstep(self, x):
        y = 1 - 1 / (1 + np.exp(10**self.p * np.sin(x)))
        return y
    
    def _param_update(self):
        self.b = -self.b
        
    @staticjit
    def make_parameters(rt, tt, a, b, t):
        a2b = a**2 / b
        lam2 = b**2 + rt**2 - 2 * b * rt * np.cos(tt)
        lam2 /= a2b**2 + rt**2 - 2 * a2b * rt * np.cos(tt)
        lam = np.sqrt(lam2)
        etac = (b - lam2 * a2b)/(1 - lam2)
        rho = np.abs((lam / (1 - lam2)) * (a2b - b))
        return lam2, lam, etac, rho
    
    @staticjit
    def root(qq, t, rt, tt, lam2, lam, etac, rho, gamma):
        tp = np.arctan2(rt * np.sin(tt), -etac + rt * np.cos(tt))
        fac = (2 * lam)/(1 + lam2)
        tlam = (2 * np.pi)**2 * ((rho**2)/gamma) * (1 + lam2) / (1 - lam2)
        out = qq - fac * np.sin(qq) - (tp - fac * np.sin(tp)) - 2 * np.pi * t / tlam
        return out

    def _rhs(self, rt, tt, a, b, gamma, t):
        
        lam2, lam, etac, rho = self.make_parameters(rt, tt, a, b, t)
        
        # parallel case
        if len(np.asarray(rt).shape) > 0:
            thetat = list()
            for arg_set in zip(rt, tt, lam2, lam, etac, rho):
                root_term = lambda x : self.root(x, t, *arg_set, gamma)
                thetat_i = fsolve(root_term, 0.01)[0]
                thetat.append(thetat_i)
            thetat = np.array(thetat)
        else:
            root_term = lambda x : self.root(x, t, rt, tt, lam2, lam, etac, rho, gamma)
            thetat = fsolve(root_term, 0.01)[0]
        rout = np.sqrt(rho**2 + etac**2 + 2 * rho * etac * np.cos(thetat))
        thout = np.arctan2(rho * np.sin(thetat), etac + rho * np.cos(thetat))
        
        self._param_update()
        
        return rout, thout
        
    def _rhs_inv(self, rt, tt, a, b, gamma, t):
        return self._rhs(rt, tt, a, -gamma, -b, t)