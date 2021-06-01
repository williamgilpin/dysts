"""
Dynamical systems in Python

(M, T, D) or (T, D) convention for outputs

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)

"""


from dataclasses import dataclass, field, asdict
import warnings
import json

import os
import sys
curr_path = sys.path[0]

import pkg_resources
data_path_continuous = pkg_resources.resource_filename('dysts', 'data/chaotic_attractors.json')
data_path_discrete = pkg_resources.resource_filename('dysts', 'data/discrete_maps.json')

import numpy as np

from .utils import integrate_dyn

try:
    from numba import jit, njit
#     from jax import jit
#     njit = jit
    
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
    """A base class for dynamical systems
    
    Attributes:
        name (str): The name of the system
        params (dict): The parameters of the system.
        random_state (int): The seed for the random number generator. Defaults to None
        
    Development
        Add a function to look up additional metadata, if requested
    """
    name : str = None
    params : dict = field(default_factory=dict)
    random_state : int = None
    
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
        np.random.seed(self.random_state)
        
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
    A continuous dynamical system base class, which loads and assigns parameter
    values from a file
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
    
    def make_trajectory(self, n, method="Radau", resample=False, pts_per_period=100,
                       return_times=False):
        """
        Generate a fixed-length trajectory with default timestep, parameters, and initial conditions
        
        Args:
            n (int): the total number of trajectory points
            method (str): the integration method
            resample (bool): whether to resample trajectories to have matching dominant 
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            return_times (bool): Whether to return the timepoints at which the solution 
                was computed
            
        """
        tpts = np.arange(n) * self.dt
        
        if resample:
#         print((self.period * self.dt))
            tlim = (self.period) * (n / pts_per_period)
            upscale_factor = (tlim/self.dt)/n
            if upscale_factor > 1e3: warnings.warn(f"Excessive integration required; scale factor {upscale_factor}")
            tpts = np.linspace(0, tlim, n)
        
        m = len(np.array(self.ic).shape)
        if m < 1: m = 1
        if m == 1:
            sol = integrate_dyn(self, self.ic, tpts, first_step=self.dt, method=method).T
        else:
            sol = list()
            for ic in self.ic:
                sol.append(integrate_dyn(self, ic, tpts, first_step=self.dt, method=method))
            sol = np.transpose(np.array(sol), (0, 2, 1))
        
        if return_times:
            return tpts, sol
        else:
            return sol
        

class DynMap(BaseDyn):
    """
    A dynamical system base class, which loads and assigns parameter
    values from a file
    
    Args:
        params : list, parameter values for the differential equations
    
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
    
    def make_trajectory(self, n, inverse=False, return_times=False, **kwargs):
        """
        Generate a fixed-length trajectory with default timestep,
        parameters, and initial condition(s)
        
        Args:
            n (int): the length of each trajectory
            inverse (bool): whether to reverse a trajectory
            return_times (bool): Whether to return the timepoints at which the solution 
                was computed
        """
        
        m = len(np.array(self.ic).shape)
        
        if m < 1: m = 1
        
        if m == 1:
            curr = np.array(self.ic)[None, :] # (M, D)
        else:
            curr = np.array(self.ic)
    
        if inverse:
            propagator = self.rhs_inv
        else:
            propagator = self.rhs
        
        traj = np.zeros((curr.shape[0], n, curr.shape[-1]))
#         traj[:, 0, :] = curr
        for i in range(n):
            curr = propagator(curr)
            traj[:, i, :] = curr
            
#         traj = np.copy(curr)[:, None, :] # (M, T, D)
#         for i in range(n):
#             curr = propagator(curr)
#             traj = np.concatenate([traj, curr[:, None, :]], axis=1)
        sol = np.squeeze(traj)

        if return_times:
            return np.arange(len(sol)), sol
        else:
            return sol
    
import collections  
class DynSysDelay(DynSys):
    """
    A delayed differential equation object. Defaults to using Euler integration scheme
    The delay timescale is assumed to be the "tau" field. The embedding dimension is set 
    by default to ten, but delay equations are infinite dimensional.
    Uses a double-ended queue for memory efficiency
    
    Todo:
        Treat previous delay values as a part of the dynamical variable in rhs
    
        Currently, only univariate delay equations and single initial conditons 
        are supported
    """
    def __init__(self): 
        super().__init__()
        #self.history = collections.deque(1.3 * np.random.rand(1 + mem_stride))
        self.__call__ = self.rhs
        
    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        X, Xprev = X[0], X[1]
        param_list = [getattr(self, param_name) for param_name in self.get_param_names()]
        out = self._rhs(X, Xprev, t, *param_list)
        return out
        
    def make_trajectory(self, n, d=10, method="Euler", noise=0.0, 
                        resample=False, pts_per_period=100,
                        return_times=False):
        """
        Generate a fixed-length trajectory with default timestep,
        parameters, and initial conditions
        
        Args:
            n (int): the total number of trajectory points
            d (int): the number of embedding dimensions to return
            method (str): Not used. Currently Euler is the only option here
            noise (float): The amplitude of brownian forcing
            resample (bool): whether to resample trajectories to have matching dominant 
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            return_times (bool): Whether to return the timepoints at which the solution 
                was computed
            
        Development:
            Support for multivariate and multidelay equations with multiple deques
            Support for multiple initial conditions
            
        """
        np.random.seed(self.random_state)
        
        mem_stride = int(np.ceil(self.tau / self.dt)) # stride
#         clipping = mem_stride
#         n += 2 * clipping
        
        if resample:
            nt = int(np.ceil((self.period / self.dt) * (n / pts_per_period)))
        else:
            nt = n
        
        # remove transient at front and back
        clipping = int(np.ceil(mem_stride / (nt / n)))
        n0 = n
        n += (d + 1) * clipping
        nt += (d + 1) * mem_stride
        
        if len(self.ic) >= mem_stride:
            history = collections.deque(self.ic[-mem_stride:])
        else:
            history = collections.deque(self.ic[-1] * (1 + 0.2 * np.random.rand(1 + mem_stride)) )

        tpts = np.arange(nt) * self.dt
        tlim = tpts[-1]
        save_inds = np.linspace(0, nt, n).astype(int)
        
        # pre-allocate solution
        sol = np.zeros(n)
        sol[0] = self.ic[-1]
        x_next = sol[0]
        
        ## Pre-compute noise
        noise_vals = np.random.normal(size=nt, loc=0.0, scale=np.sqrt(self.dt))
        for i, t in enumerate(tpts):
            if i == 0:
                continue
            dt = tpts[i] - tpts[i - 1]
            
            x_next = x_next + self.rhs([x_next, history.pop()], t) * self.dt + noise * noise_vals[i]
            
            if i in save_inds:
                sol[save_inds == i] = x_next
            history.appendleft(x_next)
        
        ## now stack to create an embedding
        sol_embed = list()
        embed_stride = int((n / nt) * mem_stride)
        for i in range(d):
            sol_embed.append(sol[i * embed_stride : -(d - i) * embed_stride])
        sol0 = np.vstack(sol_embed)[:, clipping:(n0 + clipping)].T
    
        if return_times:
            return tpts, sol0
        else:
            return sol0

    
def get_attractor_list(model_type="continuous"):
    """
    Returns the names of all models in the package
    
    Args:
        model_type (str): "continuous" (default) or "discrete"
        
    Returns:
        attractor_list (list of str): The names of all attractors in database
    """
    if model_type == "continuous":
        data_path = data_path_continuous
    else:
        data_path = data_path_discrete
    with open(data_path, "r") as file:
        data = json.load(file)
    attractor_list = sorted(list(data.keys()))
    return attractor_list
