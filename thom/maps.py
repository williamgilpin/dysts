    
"""
Various low-dimensional chaotic mapsin Python

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
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

import os
import sys
curr_path = sys.path[0]
# print(curr_path)

import pkg_resources
data_path = pkg_resources.resource_filename('thom', 'data/discrete_maps.json')
# print(data_path)

import numpy as np


try:
    from numba import jit
    has_jit = True
except ModuleNotFoundError:
    import numpy as np
    has_jit = False
    # Define placeholder functions
    def jit(func):
        return func

## Compose staticmethod and jit decorators
staticjit = lambda func: staticmethod(jit(func))


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
        """The right hand side of a dynamical equation"""
        return X
    
    def __call__(self, X):
        """Wrapper around right hand side"""
        return self.rhs(X)
    
    def make_trajectory(self, n, **kwargs):
        """
        Generate a fixed-length trajectory with default timestep,
        parameters, and initial conditions
        - n : int, the number of trajectory points
        """
        curr = self.ic
        if len(curr) == 1:
            curr = self.ic[0]
        else:
            curr = self.ic
            
        traj = np.array([curr])
        for i in range(n):
            curr = self.rhs(curr)
            traj = np.vstack([traj, curr])
        return traj


class Logistic(DynMap):
    @staticjit
    def _rhs(X, r):
        return r * X * (1 - X)
    def rhs(self, X):
        return self._rhs(X, self.r)
    
class Tent(DynMap):
    @staticjit
    def _rhs(X, mu):
        return mu * (1 - 2 * np.abs(X - 0.5))
    def rhs(self, X):
        return self._rhs(X, self.mu)

class Gauss(DynMap):
    @staticjit
    def _rhs(X, a, b):
        return np.exp(-a * X**2) + b

    def rhs(self, X):
        return self._rhs(X, self.a, self.b)

class Baker(DynMap):
    @staticjit
    def _rhs(X, a):
        x, y = X
        eps2 = 2.0 - 1e-10
        x_flr = (eps2 * x) // 1
        xp = eps2 * x - x_flr
        yp = (a * y + x_flr) / 2
        return xp, yp
    
    @staticjit
    def _rhs_inv(X, a):
        ### Bug here
        xp, yp = X
        eps2 = 2.0 - 1e-10
        if yp > 0.5:
            xflr = 0.5 + yp * a / 2
        else:
            xflr = yp * a / 2
        x = (xp + xflr) / eps2
        y = (2 * yp - xflr) / a - 10000
        return x, y 

    def rhs(self, X):
        return self._rhs(X, self.a)
    
    def rhs_inv(self, X):
        return self._rhs_inv(X, self.a)

class DeJong(DynMap):
    @staticjit
    def _rhs(X, a, b, c, d):
        x, y = X
        xp = np.sin(a * y) - np.cos(b * x)
        yp = np.sin(c * x) - np.cos(d * y)
        return (xp, yp)

    def rhs(self, X):
        return self._rhs(X, self.a, self.b, self.c, self.d)

class Chirikov(DynMap):
    @staticjit
    def _rhs(X, k):
        p, x = X
        pp = p + k * np.sin(x)
        xp = x + pp 
        return pp, xp

    @staticjit
    def _rhs_inv(X, k):
        pp, xp = X
        x = xp - pp
        p = pp - k * np.sin(xp - pp)
        return p, x
    
    def rhs(self, X):
        return self._rhs(X, self.k)

    def rhs_inv(self, X):
        return self._rhs_inv(X, self.k)

class Henon(DynMap):

    @staticjit
    def _rhs(X, a, b):
        x, y = X
        xp = 1 - a * x**2 + y
        yp = b * x
        return xp, yp

    @staticjit
    def _rhs_inv(X, a, b):
        xp, yp = X
        x = yp / b
        y = - xp - 1 + a * x**2
        return x, y
    
    def rhs(self, X):
        return self._rhs(X, self.a, self.b)

    def rhs_inv(self, X):
        return self._rhs_inv(X, self.a, self.b)


        

        
