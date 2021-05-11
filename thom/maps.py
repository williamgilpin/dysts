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
http://www.chaoscope.org/doc/attractors.htm
http://nbodyphysics.com/chaoticmotion/html/_hadley_8cs_source.html
https://chaoticatmospheres.com/mathrules-strange-attractors
https://matousstieber.wordpress.com/2016/01/12/strange-attractors/

DEV: 
+ Database: How to load a function object from a string? exec does not do this
+ Add a function compute_dt that finds the timestep based on fft
+ Set standard integration timestep based on this value
+ Add a function that rescales outputs to the same interval
+ Add a function that finds initial conditions on the attractor
"""

from dataclasses import dataclass, field, asdict
import warnings
import json

# Ikeda, Pickover

from importlib import import_module
import os
import sys
curr_path = sys.path[0]
# print(curr_path)

import pkg_resources
data_path = pkg_resources.resource_filename('thom', 'data/discrete_maps.json')
# print(data_path)

import numpy as np


try:
    from numba import jit, njit
    has_jit = True
except ModuleNotFoundError:
    import numpy as np
    has_jit = False
    # Define placeholder functions
    def jit(func):
        return func
    njit = jit

staticjit = lambda func: staticmethod(njit(func)) # Compose staticmethod and jit decorators

@dataclass(init=False)
class DynMap:
    """
    A dynamical system base class, which loads and assigns parameter
    values from a file
    - params : list, parameter values for the differential equations
    
    DEV: A function to look up additional metadata, if requested
    """
    name : str = None
    params : dict = field(default_factory=dict)
    
    def __init__(self, **entries):
        self.name = self.__class__.__name__
        self._load_data()
        dfac = lambda : self._load_data()["parameters"]
        self.params = self._load_data()["parameters"]
        self.params.update(entries)
        # Cast all parameter arrays to numpy
        for key in self.params:
            if not np.isscalar(self.params[key]):
                self.params[key] = np.array(self.params[key])
        self.__dict__.update(self.params)
        self.ic = self._load_data()["initial_conditions"]
        
    def _load_data(self):
        """Load data from a JSON file"""
        # with open(os.path.join(curr_path, "chaotic_attractors.json"), "r") as read_file:
        #     data = json.load(read_file)
        with open(data_path, "r") as read_file:
            data = json.load(read_file)
        try:
            return data[self.name]
        except KeyError:
            print(f"No metadata available for {self.name}")
            return {"parameters" : None}

    def rhs(self, X):
        """The right hand side of a dynamical map"""
        param_list = [getattr(self, param_name) for param_name in self.params.keys()]
        xin = list()
        for i in range(X.shape[-1]):
            xin.append(X[..., i])
        out = self._rhs(*xin, *param_list)
        return np.vstack(out).T
    
    def rhs_inv(self, Xp):
        """The inverse of the right hand side of a dynamical map"""
        param_list = [getattr(self, param_name) for param_name in self.params.keys()]
        xpin = list()
        for i in range(Xp.shape[-1]):
            xpin.append(Xp[..., i])
        out = self._rhs_inv(*xpin, *param_list)
        return np.vstack(out).T
    
    def __call__(self, X):
        """Wrapper around right hand side"""
        return self.rhs(X)
    
    def make_trajectory(self, n, inverse=False, **kwargs):
        """
        Generate a fixed-length trajectory with default timestep,
        parameters, and initial condition(s)
        - n : int, the length of each trajectory
        - inverse : bool, whether to reverse a trajectory
        """
        
        m = len(np.array(self.ic).shape)
        
        if m < 1: m = 1
        
        if m == 1:
            curr = np.array(self.ic)[None, :] # (M, D)
        else:
            curr = np.array(self.ic)
            
        traj = np.copy(curr)[:, None, :] # (M, T, D)
    
        if inverse:
            propagator = self.rhs_inv
        else:
            propagator = self.rhs
        
        for i in range(n):
            curr = propagator(curr)
            traj = np.concatenate([traj, curr[:, None, :]], axis=1)
        return traj


class Logistic(DynMap):
    @staticjit
    def _rhs(x, r):
        return r * x * (1 - x)
    
class Tent(DynMap):
    @staticjit
    def _rhs(x, mu):
        return mu * (1 - 2 * np.abs(x - 0.5))

class Gauss(DynMap):
    @staticjit
    def _rhs(x, a, b):
        return np.exp(-a * x**2) + b

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

class DeJong(DynMap):
    @staticjit
    def _rhs(x, y, a, b, c, d):
        xp = np.sin(a * y) - np.cos(b * x)
        yp = np.sin(c * x) - np.cos(d * y)
        return xp, yp

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
class BlinkingVortex(DynMap):
    
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
    
    def _rhs(self, rt, tt, a, gamma, b, t):
        
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
        
    def _rhs_inv(self, rt, tt, a, gamma, b, t):
        return self._rhs(rt, tt, a, -gamma, -b, t)