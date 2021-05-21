"""
Dynamical systems in Python

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
https://matousstieber.wordpress.com/2016/01/12/strange-attractors/

DEV: 
+ Add a function compute_dt that finds the timestep based on fft
+ Set standard integration timestep based on this value
+ Add a function that rescales outputs to the same interval
+ Add a function that finds initial conditions on the attractor
"""


from dataclasses import dataclass, field, asdict
import warnings
import json

import os
import sys
curr_path = sys.path[0]
# print(curr_path)

import pkg_resources
data_path_continuous = pkg_resources.resource_filename('thom', 'data/chaotic_attractors.json')
# print(data_path)
data_path_discrete = pkg_resources.resource_filename('thom', 'data/discrete_maps.json')

import numpy as np

from .utils import integrate_dyn

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
class BaseDyn:
    """
    A base class for dynamical systems
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
        
    def get_param_names(self):
        return sorted(self.params.keys())
        
    def _load_data(self):
        """Load data from a JSON file"""
        # with open(os.path.join(curr_path, "chaotic_attractors.json"), "r") as read_file:
        #     data = json.load(read_file)
        with open(self.data_path, "r") as read_file:
            data = json.load(read_file)
        try:
            return data[self.name]
        except KeyError:
            print(f"No metadata available for {self.name}")
            return {"parameters" : None}
        
    @staticmethod
    def bound_trajectory(traj):
        """Bound a trajectory within a periodic domain"""
        return np.mod(traj, 2*np.pi)

from scipy.integrate import solve_ivp
class DynSys(BaseDyn):
    """
    A dynamical system base class, which loads and assigns parameter
    values from a file
    - params : list, parameter values for the differential equations
    
    DEV: A function to look up additional metadata, if requested
    """
    def __init__(self):
        self.data_path = data_path_continuous
        super().__init__()
        self.dt = self._load_data()["dt"]	
        self.period = self._load_data()["period"]
    
    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        param_list = [getattr(self, param_name) for param_name in self.get_param_names()]
        out = self._rhs(*X.T, t, *param_list)
        return out
    
    def __call__(self, X, t):
        """Wrapper around right hand side"""
        return self.rhs(X, t)
    
    def make_trajectory(self, n, method="RK45", resample=False, pts_per_period=100):
        """
        Generate a fixed-length trajectory with default timestep,
        parameters, and initial conditions
        
        Args:
            n (int): the total number of trajectory points
            method (str): the integration method
            resample (bool): whether to resample trajectories to have matching dominant 
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            
        """
        tpts = np.arange(n)*self.dt
        
        if resample:
#         print((self.period * self.dt))
            tlim = (self.period) * (n / pts_per_period)
            upscale_factor = (tlim/self.dt)/n
            if n > 10: warnings.warn(f"Excess integration required; scale factor {upscale_factor}")
            tpts = np.linspace(0, tlim, n)
        
        m = len(np.array(self.ic).shape)
        if m < 1: m = 1
        if m == 1:
            sol = integrate_dyn(self, self.ic, tpts, first_step=self.dt, method=method)
        else:
            sol = list()
            for ic in self.ic:
                sol.append(integrate_dyn(self, ic, tpts, first_step=self.dt, method=method))
            sol = np.transpose(np.array(sol), (0, 2, 1))
            
        return sol
        



# # Ikeda, Pickover

# @dataclass(init=False)
class DynMap(BaseDyn):
    """
    A dynamical system base class, which loads and assigns parameter
    values from a file
    - params : list, parameter values for the differential equations
    
    DEV: A function to look up additional metadata, if requested
    """
    
    def __init__(self):
        self.data_path = data_path_discrete
        super().__init__()

    def rhs(self, X):
        """The right hand side of a dynamical map"""
        param_list = [getattr(self, param_name) for param_name in self.get_param_names()]
        out = self._rhs(*X.T, *param_list)
        return np.vstack(out).T
    
    def rhs_inv(self, Xp):
        """The inverse of the right hand side of a dynamical map"""
        param_list = [getattr(self, param_name) for param_name in self.get_param_names()]
        out = self._rhs_inv(*Xp.T, *param_list)
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