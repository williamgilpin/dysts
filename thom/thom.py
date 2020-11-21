    
"""
Various low-dimensional chaotic dynamical systems in Python

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
+ Load the attractor information from a database. How to build load a function
object from a string? exec does not do this
+ add methods that initialize standard chaotic values
+ Add a function compute_dt that finds the timestep based on fft
+ Set standard integration timestep based on this value
+ Add a function that rescales outputs to the same interval
+ Add a function that finds initial conditions on the attractor
"""

from dataclasses import dataclass, field, asdict
import warnings
from numba import jit
import json

from utils import integrate_dyn

# try:
#     import jax.numpy as np
#     has_jax = True
# except ModuleNotFoundError:
#     import numpy as np
#     has_jax = False
#     warnings.warn("JAX not found, falling back to numpy.")
    
import numpy as np

@dataclass(init=False)
class DynSys:
    """
    A dynamical system class, which loads and assigns parameter
    values
    - params : list, parameter values for the differential equations
    
    DEV: A function to look up additional metadata, if requested
    """
    name : str = None
    params : dict = field(default_factory=dict)
    
    def __init__(self, **entries):
        self.name = self.__class__.__name__
        dfac = lambda : _load_data(self.name)["parameters"]
        self.params = self._load_data(self.name)["parameters"]
        self.params.update(entries)
        # Cast all arrays to numpy
        for key in self.params:
            if not np.isscalar(self.params[key]):
                self.params[key] = np.array(self.params[key])
        self.__dict__.update(self.params)
#         self.__dict__.update(entries) # can probably be removed
        self.dt = self._load_data(self.name)["dt"]
        self.ic = self._load_data(self.name)["initial_conditions"]
        

    @staticmethod
    def _load_data(name):
        with open("chaotic_attractors.json", "r") as read_file:
            data = json.load(read_file)
        try:
            return data[name]
        except KeyError:
            print(f"No metadata available for {name}")
            return {"parameters" : None}
          
    def rhs(self, X, t):
        return X
    
    def __call__(self, X, t):
        return self.rhs(X, t)
    
    def make_trajectory(self, n, **kwargs):
        """
        Generate a fixed-length trajectory with default timestep,
        parameters, and initial conditions
        - n : int, the number of trajectory points
        - kwargs : dict, arguments passed to integrate_dyn
        """
        tpts = np.arange(n)*self.dt
        sol = integrate_dyn(self, self.ic, tpts, **kwargs)
        return sol

class Lorenz(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.sigma*(y - x)
        ydot = x*(self.rho - z) - y
        zdot = x*y - self.beta*z
        return (xdot, ydot, zdot)

class LorenzBounded(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        f = 1 - (x**2 + y**2 + z**2)/self.r**2
        xdot = self.sigma*(y - x)*f
        ydot = (x*(self.rho - z) - y)*f
        zdot = (x*y - self.beta*z)*f
        return (xdot, ydot, zdot)
        
class LorenzCoupled(DynSys):
    def rhs(self, X, t):
        x1, y1, z1, x2, y2, z2 = X
        x1dot = self.sigma*(y1 - x1)
        y1dot = x1*(self.rho1 - z1) - y1
        z1dot = x1*y1 - self.beta*z1
        x2dot = self.sigma*(y2 - x2) + self.eps*(x1 - x2)
        y2dot = x2*(self.rho2 - z2) - y2
        z2dot = x2*y2 - self.beta*z2
        return (x1dot, y1dot, z1dot, x2dot, y2dot, z2dot)

class Rossler(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -y - z
        ydot = x + self.a*y
        zdot = self.b + z*(x - self.c)
        return (xdot, ydot, zdot)

class Thomas(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -self.a*x + self.b*np.sin(y)
        ydot = -self.a*y + self.b*np.sin(z)
        zdot = -self.a*z + self.b*np.sin(x)
        return (xdot, ydot, zdot)

class DoublePendulum(DynSys):
    def rhs(self, X, t):
        th1, th2, p1, p2 = X
        g = 9.82
        m, d = self.m, self.d
        pre = (6/(m*d**2))
        denom = 16 - 9*np.cos(th1 - th2)**2
        th1_dot = pre*(2*p1 - 3*np.cos(th1 - th2)*p2)/denom
        th2_dot = pre*(8*p2 - 3*np.cos(th1 - th2)*p1)/denom
        p1_dot = -0.5*(m*d**2)*(th1_dot*th2_dot*np.sin(th1 - th2) + 3*(g/d)*np.sin(th1))
        p2_dot = -0.5*(m*d**2)*(-th1_dot*th2_dot*np.sin(th1 - th2) + 3*(g/d)*np.sin(th2))
        return (th1_dot, th2_dot, p1_dot, p2_dot)

class Halvorsen(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -self.a*x - self.b*(y + z) - y**2
        ydot = -self.a*y - self.b*(z + x) - z**2
        zdot = -self.a*z - self.b*(x + y) - x**2
        return (xdot, ydot, zdot)

class Chua(DynSys):
    def diode(self, x):
        return self.m1*x + (0.5)*(self.m0 - self.m1)*(np.abs(x + 1) - np.abs(x - 1))
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.alpha*(y - x - self.diode(x))
        ydot = x - y + z
        zdot = -self.beta*y
        return (xdot, ydot, zdot)
    
class MultiChua(DynSys):
    def diode(self, x):
        m, c = self.m, self.c
        total = m[-1]*x
        for i in range(1, 6):
            total += 0.5*(m[i-1] - m[i])*(np.abs(x + c[i]) - np.abs(x - c[i]))
        return total
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*(y - self.diode(x))
        ydot = x - y + z
        zdot = -self.b*y
        return (xdot, ydot, zdot)

class Duffing(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = -self.delta*y - self.beta*x - self.alpha*x**3 + self.gamma*np.cos(z)
        zdot = self.omega
        return (xdot, ydot, zdot)

    
    


class MackeyGlass(object):
    """
    Simulate the dynamics of the Mackey-Glass time delay model
    Inputs
    - tau : float, the delay parameter
    """
    def __init__(self, tau, beta, gamma, dt, n=10):
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.dt = dt
        
        self.mem = int(np.ceil(tau/dt))
        self.history = deque(1.2 + (np.random.rand(self.mem) - 0.5))

    def __call__(self, x, t):
        xt = self.history.pop()
        xdot = self.beta*(xt/(1 + xt**self.n)) - self.gamma*x
        return xdot
    
    def integrate(self, x0, tpts):
        x_series = np.zeros_like(tpts)
        x_series[0] = x0
        self.history.appendleft(x0)
        for i, t in enumerate(tpts):
            if i==0:
                continue
            
            dt = tpts[i] - tpts[i-1]
            x_nxt = x_series[i-1] + self(x_series[i-1], t)*self.dt
            
            x_series[i] = x_nxt 
            self.history.appendleft(x_nxt)
        
        return x_series

class DoubleGyre(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        a = self.eps*np.sin(z)
        b = 1 - 2*self.eps*np.sin(z)
        f = a*x**2 + b*x
        dx = -self.a * np.pi*np.sin(np.pi*f)*np.cos(np.pi*y)
        dy = self.a * np.pi*np.cos(np.pi*f)*np.sin(np.pi*y)*(2*a*x + b)
        dz = self.omega
        return np.stack([dx, dy, dz]).T

class HindmarshRose:    
    """
    Marhl, Perc. Chaos, Solitons, Fractals 2005
    """
    def __init__(self, tx=0.03, tz=0.8, a=0.49, b=1.0, d=1.0, s=1.0, c=.0322):
        self.tx, self.tz, self.a, self.b, self.d, self.s, self.c = tx, tz, a, b, d, s, c
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        tx, tz, a, b, d, s, c = self.tx, self.tz, self.a, self.b, self.d, self.s, self.c
        xdot = - tx*x + y - a*x**3 + b*x**2 + z
        ydot = -a*x**3  - (d - b)*x**2 + z
        zdot = -s*x - z + c
        return (xdot/tx, ydot, zdot/tz)
    
    
class JerkCircuit(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -z - x  - self.eps*(np.exp(y/self.y0) - 1)
        return (xdot, ydot, zdot)

    
class ForcedBrusselator(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a + x**2*y - (self.b + 1)*x + self.f*np.cos(z)
        ydot = self.b*x - x**2*y
        zdot = self.w
        return (xdot, ydot, zdot)


class WindmiReduced(DynSys):
    def rhs(self, X, t):
        i, v, p = X
        idot = self.a1*(self.vsw - v)
        vdot = self.b1*i - self.b2*p**1/2 - self.b3*v
        pdot = self.vsw**2 - p**(5/4)*self.vsw**(1/2)*(1 + np.tanh(self.d1*(i-1)))/2
        return (idot, vdot, pdot)  

class MooreSpiegel(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = self.a*z
        zdot = -z + self.eps*y - y*x**2 - self.b*x
        return (xdot, ydot, zdot)

class CoevolvingPredatorPrey(DynSys):
    def rhs(self, X, t):
        x, y, alpha = X
        xdot = x*(-((self.a3*y)/(1 + self.b2*x)) + (self.a1*alpha*(1 - self.k1*x*(-alpha + alpha*self.delta)))/(1 + self.b1*alpha) - self.d1*(1 - self.k2*(-alpha**2 + (alpha*self.delta)**2) + self.k4*(-alpha**4 + (alpha*self.delta)**4))) 
        ydot = (-self.d2 + (self.a2*x)/(1 + self.b2*x))*y
        alphadot = self.vv*(-((self.a1*self.k1*x*alpha*self.delta)/(1 + self.b1*alpha)) - self.d1*(-2*self.k2*alpha*self.delta**2 + 4*self.k4*alpha**3*self.delta**4))
        return (xdot, ydot, alphadot)

class KawczynskiStrizhak(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.gamma*(y - x**3 + 3*self.mu*x)
        ydot = -2*self.mu*x - y - z + self.beta
        zdot = self.kappa*(x - z)
        return (xdot, ydot, zdot)

    
class BelousovZhabotinsky(DynSys):
    def rhs(self, X, t):    
        x, z, v = X
        ybar = (1/self.y0)*self.yb1*z*v/(self.yb2*x + self.yb3 + self.kf)
        rf = (self.ci - self.z0*z)*np.sqrt(x)
        xdot = self.c1*x*ybar + self.c2*ybar + self.c3*x**2 + self.c4*rf + self.c5*x*z - self.kf*x
        zdot = (self.c6/self.z0)*rf + self.c7*x*z + self.c8*z*v + self.c9*z - self.kf*z
        vdot = self.c10*x*ybar + self.c11*ybar + self.c12*x**2 + self.c13*z*v - self.kf*v
        return (xdot*self.t0, zdot*self.t0, vdot*self.t0)
    
class RabinovichFabrikant(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y*(z - 1 +x**2) + self.g*x
        ydot = x*(3*z + 1 - x**2) + self.g*y
        zdot = -2*z*(self.a + x*y)
        return (xdot, ydot, zdot)
    
class NoseHoover(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = -x + y*z
        zdot = self.a - y**2
        return (xdot, ydot, zdot)

class Dadras(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y - self.p*x + self.o*y*z
        ydot = self.r*y - x*z + z
        zdot = self.c*x*y - self.e*z
        return (xdot, ydot, zdot)
    
class SprottTorus(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y + 2*x*y + x*z
        ydot = 1 - 2*x**2 + y*z
        zdot = x - x**2 - y**2
        return (xdot, ydot, zdot)
    
    
class SprottB(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y*z
        ydot = x - y
        zdot = 1 - x*y
        return (xdot, ydot, zdot)
    
class SprottC(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y*z
        ydot = x -y
        zdot = 1 - x**2
        return (xdot, ydot, zdot)
    
class SprottD(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -y
        ydot = x + z
        zdot = x*z + 3*y**2
        return (xdot, ydot, zdot)
    
class SprottE(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y*z
        ydot = x**2 - y
        zdot = 1 - 4*x
        return (xdot, ydot, zdot)
    
class SprottF(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y + z
        ydot = -x + self.a*y
        zdot = x**2 - z
        return (xdot, ydot, zdot)
    
class SprottG(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*x + z
        ydot = x*z - y
        zdot = -x + y
        return (xdot, ydot, zdot)
    
class SprottH(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -y + z**2
        ydot = x + self.a*y
        zdot = x - z
        return (xdot, ydot, zdot)
    
class SprottI(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -self.a*y
        ydot = x + z
        zdot = x + y**2 - z
        return (xdot, ydot, zdot)
    
class SprottJ(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = 2*z
        ydot = -2*y + z
        zdot = -x + y + y**2
        return (xdot, ydot, zdot)

class SprottK(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = x*y - z
        ydot = x - y
        zdot = x + self.a*z
        return (xdot, ydot, zdot)    

class SprottL(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y + self.b*z
        ydot = self.a*x**2 - y
        zdot = 1 - x
        return (xdot, ydot, zdot)

class SprottM(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -z
        ydot = -x**2 - y
        zdot = self.a*(1 + x) + y
        return (xdot, ydot, zdot)

class SprottN(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -2*y
        ydot = x + z**2
        zdot = 1 + y - 2*z
        return (xdot, ydot, zdot)

class SprottO(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = x - z
        zdot = x + x*z + self.a*y
        return (xdot, ydot, zdot)

class SprottP(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*y + z
        ydot = -x + y**2
        zdot = x + y
        return (xdot, ydot, zdot)
    
class SprottQ(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -z
        ydot = x - y
        zdot = self.a*x + y**2 + self.b*z
        return (xdot, ydot, zdot)
    
class SprottR(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a - y
        ydot = self.b + z
        zdot = x*y - z
        return (xdot, ydot, zdot)
    
class SprottS(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -x - 4*y
        ydot = x + z**2
        zdot = 1 + x
        return (xdot, ydot, zdot)

class Arneodo(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -self.a*x - self.b*y - self.c*z  + self.d*x**3
        return (xdot, ydot, zdot)
    
class Rucklidge(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = - self.a*x + self.b*y - y*z
        ydot = x
        zdot = -z + y**2
        return (xdot, ydot, zdot)

class Sakarya(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*x + self.h*y + self.s*y*z
        ydot = -self.b*y - self.p*x + self.q*x*z
        zdot = self.c*z - self.r*x*y
        return (xdot, ydot, zdot)
    
class LiuChen(Sakarya):
    """
    Liu, Chen. Int J Bifurcat Chaos. 2004: 1395-1403.
    """
    def __init__(self):
        super().__init__(a=0.4, h=0.0, b=12.0, p=0.0, q=-1.0, c=-5.0, r=1.0)
        

class RayleighBenard(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = self.r*y - x*z
        zdot = x*y - self.b*z
        return (xdot, ydot, zdot)
    
    
class Finance(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = (1/self.b - self.a)*x + z + x*y
        ydot = -self.b*y - x**2
        zdot = -x - self.c*z
        return (xdot, ydot, zdot)

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


class Bouali2(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*x*(self.y0 - y) - self.b*z
        ydot = -self.g*y*(1 - x**2)
        zdot = -self.m*x*(1.5 - self.bb*z) - self.c*z
        return (xdot, ydot, zdot)
    
class Bouali(Bouali2):
    def __init__(self):
        super().__init__(y0=4, a=1, b=-0.3, g=1, m=1, bb=1.0, c=0.05)
# class Bouali:
#     """
#     Bouali 2012
#     """
#     def __init__(self, a=0.3, b=1.0, c=0.05):
#         self.a, self.b, self.c = a, b, c
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         x, y, z = X
#         xdot = x*(4 - y) + self.a*z
#         ydot = -y*(1 - x**2)
#         zdot = -x*(1.5 - self.b*z) - self.c*z
#         return (xdot, ydot, zdot)

class LuChenCheng(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -(self.a * self.b)/(self.a + self.b)*x - y*z + self.c
        ydot = self.a*y + x*z
        zdot = self.b*z + x*y
        return (xdot, ydot, zdot)

class LuChen:
    """
    A system that switches shapes between the Lorenz and Chen 
    attractors as the parameter c is varied from 12.7 to 28.5
    Also called the Chen Celikovsky attractor
    Lu, Chen. Int J Bifurcat Chaos. 2002: 659-661
    """
    def __init__(self, a=36, b=3, c=18):
        self.a, self.b, self.c = a, b, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = -x*z + self.c*y
        zdot = x*y - self.b*z
        return (xdot, ydot, zdot)


class QiChen:
    """
    Qiet al. Chaos, Solitons & Fractals 2008 
    """
    def __init__(self, a=38, b=2.666, c=80):
        self.a, self.b, self.c = a, b, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - x) + y*z
        ydot = self.c*x + y - x*z
        zdot = x*y - self.b*z
        return (xdot, ydot, zdot)
    
class ZhouChen(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*x + self.b*y + y*z
        ydot = self.c*y - x*z + self.d*y*z
        zdot = self.e*z - x*y
        return (xdot, ydot, zdot)
    
class BurkeShaw(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -self.n*(x + y)
        ydot = y - self.n*x*z
        zdot = self.n*x*y + self.e
        return (xdot, ydot, zdot)    

    
class YuWang:
    """
    Yu,  Wang. Eng Techn& Applied Science Research 2012
    """
    def __init__(self, a=10, b=30, c=2, d=2.5):
        self.a, self.b, self.c, self.d = a, b, c, d
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = self.b*x - self.c*x*z
        zdot = np.cosh(x*y) - self.d*z
        return (xdot, ydot, zdot)
    
class Chen:
    """
    Chen, Ueta. Int J Bifurcat Chaos. 2004: 1465-1466
    Chen. Proc. First Int. Conf. Control of Oscillations and Chaos. 1997: 181–186
    """
    def __init__(self, a=35, b=3, c=28):
        self.a, self.b, self.c = a, b, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = (self.c - self.a)*x - x*z + self.c*y
        zdot = x*y - self.b*z
        return (xdot, ydot, zdot)
    
    
class ChenLee:
    def __init__(self, a=5.0, b=-10.0, c=-0.38):
        self.a, self.b, self.c = a, b, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*x - y*z
        ydot = self.b*y + x*z
        zdot = self.c*z + x*y/3
        return (xdot, ydot, zdot)
    
    
class WangSun:
    """
    
    """
    def __init__(self, a=0.2, b=-0.01, d=-0.4, e=-1.0, f=-1.0, q=1.0):
        self.a, self.b, self.d, self.e, self.f, self.q = a, b, d, e, f, q
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*x + self.q*y*z
        ydot = self.b*x + self.d*y - x*z
        zdot = self.e*z + self.f*x*y
        return (xdot, ydot, zdot)
    
class YuWang:
    """
    
    """
    def __init__(self, a=10, b=40, c=2, d=2.5):
        self.a, self.b, self.c, self.d = a, b, c, d
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = self.b*x - self.c*x*z
        zdot = np.exp(x*y) - self.d*z
        return (xdot, ydot, zdot)

class SanUmSrisuchinwong(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y - x
        ydot = -z*np.tanh(x)
        zdot = -self.a + x*y + np.abs(y)
        return (xdot, ydot, zdot)

class DequanLi:
    """
    Li, Phys Lett A. 2008: 387-393
    Sometimes called the "Three Scroll" unified attractor TSUCS-2
    """
    def __init__(self, a=40, c=1.833, d=0.16, eps=0.65, k=55, f=20):
        (self.a, self.c, self.d, 
         self.eps, self.k, self.f) = a ,c, d, eps, k, f
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - x) + self.d*x*z
        ydot = self.k*x + self.f*y - x*z
        zdot = self.c*z + x*y - self.eps*x**2
        return (xdot, ydot, zdot)

class PanXuZhou(DequanLi):
    """
    A named attractor related to the DequanLi attractor
    """
    def __init__(self):
        super().__init__(a=10, d=0, k=16, f=0, c=-8/3, eps=0)

class Tsucs2(DequanLi):
    """
    A named attractor related to the DequanLi attractor
    """
    def __init__(self):
        super().__init__(d=0.5, k=0, c=0.833)

        
class NewtonLiepnik:
    """
    """
    def __init__(self, a=0.4, b=0.175):
        self.a, self.b = a, b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -self.a*x + y + 10*y*z
        ydot = -x - 0.4*y + 5*x*z
        zdot = self.b*z - 5*x*y
        return (xdot, ydot, zdot)

class HyperRossler(DynSys):
    def rhs(self, X, t):
        x, y, z, w = X
        xdot = - y - z
        ydot = x + self.a*y + w
        zdot = self.b + x*z
        wdot = -self.c*z + self.d*w
        return (xdot, ydot, zdot, wdot)

class SaltonSea(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.r*x*(1 - (x + y)/self.k) - self.lam*x*y
        ydot = self.lam*x*y - self.m*y*z/(y + self.a) - self.mu*y
        zdot = self.th*y*z/(y + self.a) - self.d*z
        return (xdot, ydot, zdot)
    
# class CoupledOscillators:
#     """
#     Pazó, Montbrió. PRX 2014
#     """
# class NonlinearSchrodinger:
#     """
#     Nozake, Bekki. Physica D 1986
    
#     Was not able to replicate
#     """

class CaTwoPlus(DynSys):
    def rhs(self, X, t):
        z, y, a = X
        Vin = self.V0 + self.V1*self.beta
        V2 = self.Vm2*(z**2)/(self.K2**2 + z**2)
        V3 = (self.Vm3*(z**self.m)/(self.Kz**self.m + z**self.m) )*(y**2/(self.Ky**2 + y**2) )*(a**4/(self.Ka**4 + a**4))
        V5 = self.Vm5*(a**self.p/(self.K5**self.p + a**self.p))*(z**self.n/(self.Kd**self.n + z**self.n))
        zdot = Vin - V2 + V3 + self.kf*y - self.k*z
        ydot = V2 - V3 - self.kf*y
        adot = self.beta*self.V4 - V5 - self.eps*a
        return (zdot, ydot, adot)

class CaTwoPlusQuasiperiodic(CaTwoPlus):
    """
    Intracellular calcium ion oscillations
    with quasiperiodic parameter values
    + paper also includes full parameters for limit cycle
    Houart, Dupont, Goldbeter. Bull Math Biol 1999
    """
    def __init__(self):
        CaTwoPlus.__init__(self, beta=0.51, K5=0.3, Ka=0.2, Kd=0.5, Ky=0.2, 
                     Kz=0.5, eps=.1, p=2, Vm3=20, V4=5, Vm5=30)

class CellCycle(DynSys):
    def rhs(self, X, t):
        c1, m1, x1, c2, m2, x2 = X
        Vm1, Um1= 2*[self.Vm1]
        vi1, vi2 = 2*[self.vi]
        H1, H2, H3, H4 = 4*[self.K]
        K1, K2, K3, K4 = 4*[self.K]
        V2, U2 = 2*[self.V2]
        Vm3, Um3 = 2*[self.Vm3]
        V4, U4 = 2*[self.V4]
        Kc1, Kc2 = 2*[self.Kc]
        vd1, vd2 = 2*[self.vd]
        Kd1, Kd2 = 2*[self.Kd1]
        kd1, kd2 = 2*[self.kd1]
        Kim1, Kim2 = 2*[self.Kim]
        V1 = Vm1*c1/(Kc1 + c1)
        U1 = Um1*c2/(Kc2 + c2)
        V3 = m1*Vm3
        U3 = m2*Um3
        c1dot = vi1*Kim1/(Kim1 + m2) - vd1*x1*c1/(Kd1 + c1) - kd1*c1
        c2dot = vi2*Kim2/(Kim2 + m1) - vd2*x2*c2/(Kd2 + c2) - kd2*c2
        m1dot = V1*(1 - m1)/(K1 + (1 - m1)) - V2*m1/(K2 + m1)
        m2dot = U1*(1 - m2)/(H1 + (1 - m2)) - U2*m2/(H2 + m2)
        x1dot = V3*(1 - x1)/(K3 + (1 - x1)) - V4*x1/(K4 + x1)
        x2dot = U3*(1 - x2)/(H3 + (1 - x2)) - U4*x2/(H4 + x2)
        return  c1dot, m1dot, x1dot, c2dot, m2dot, x2dot

class CircadianRhythm(DynSys):
    def rhs(self, X, t):
        m, fc, fs, fn, th = X
        vs = 2.5*((0.5 + 0.5*np.cos(th)) + self.vmin)*(self.vmax - self.vmin)
        mdot = vs*(self.Ki**self.n)/(self.Ki**self.n + fn**self.n) - self.vm*m/(self.km + m)
        fcdot = self.ks*m - self.k1*fc + self.k2*fn - self.k*fc
        fsdot = self.k*fc - self.vd*fs/(self.kd + fs)
        fndot = self.k1*fc - self.k2*fn - self.vdn*fn/(self.kdn + fn)
        thdot = 2*np.pi/24
        return (mdot, fcdot, fsdot, fndot, thdot)
    

class FluidTrampoline(DynSys):
    def rhs(self, X, t):
        x, y, th = X
        xdot = y
        ydot = -1 - np.heaviside(-x, 0)*(x + self.psi*y*np.abs(y)) + self.gamma*np.cos(th)
        thdot = self.w  
        return (xdot, ydot, thdot)

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

class ItikBanksTumor(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = x*(1 - x) - self.a12*x*y - self.a13*x*z
        ydot = self.r2*y*(1 - y) - self.a21*x*y
        zdot = self.r3*x*z/(x + self.k3) - self.a31*x*z - self.d3*z
        return (xdot, ydot, zdot)

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
    
# class DequanLi:
#     """
#     Use subclassing
#     Li, Phys Lett A. 2008: 387-393
#     Sometimes called the "Three Scroll" unified attractor TCUCS-1
#     """
#     def __init__(self, a=40, c=1.833, d=0.16, eps=0.65, k=55, f=20):
#         (self.a, self.c, self.d, 
#          self.eps, self.k, self.f) = a ,c, d, eps, k, f
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         x, y, z = X
#         xdot = self.a*(y - x) + self.d*x*z
#         ydot = self.k*x + self.f*y - x*z
#         zdot = self.c*z + x*y - self.eps*x**2
#         return (xdot, ydot, zdot)

class Aizawa(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        xdot = (z - b)*x - d*y
        ydot = d*x + (z - b)*y
        zdot = c + a*z - z**3/3 - (x**2 + y**2)*(1 + e*z) + f*z*x**3
        return (xdot, ydot, zdot)  
    
class AnishchenkoAstakhov(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        mu, eta = self.mu, self.eta
        xdot = mu*x + y - x*z
        ydot = -x
        zdot = -eta*z + eta*np.heaviside(x, 0)*x**2
        return (xdot, ydot, zdot)  
    
class ShimizuMorioka(DynSys):
    def rhs(self, X, t):
        x, y, z = X  
        xdot = y
        ydot = x - self.a*y - x*z
        zdot = -self.b*z + x**2
        return (xdot, ydot, zdot)
    
class GenesioTesi(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -self.c*x - self.b*y - self.a*z + x**2
        return (xdot, ydot, zdot)    

# class Pickover:
#     def __init__(self, a=-0.759494, b=2.449367, c=1.253165, d=1.5):
#         self.a, self.b, self.c, self.d = a, b, c, d
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         x, y, z = X
#         xdot = np.sin(self.a*y) - z*np.cos(self.b*x)
#         ydot = z*np.sin(self.c*x) - np.cos(self.d*y)
#         zdot = np.sin(x)
#         return (xdot, ydot, zdot)    

class Hadley(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -y**2 - z**2 - self.a*x + self.a*self.f
        ydot = x*y - self.b*x*z - y + self.g
        zdot = self.b*x*y + x*z - z
        return (xdot, ydot, zdot)   


class ForcedVanDerPol:
    """
    The forced van der pol oscillator
    """
    def __init__(self, mu=8.53, a=1.2, w=0.63):
        self.mu, self.a, self.w = mu, a, w
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        ydot = self.mu *(1 - x**2)*y - x + self.a*np.sin(z)
        xdot = y
        zdot = self.w
        return (xdot, ydot, zdot)

class ForcedFitzHugh:
    """
    A forced FitzHugh-Nagumo oscillator
    """
    def __init__(self, curr=0.5, a=0.7, b=0.8, gamma=0.08, omega=1.1, f=0.25):
        self.curr, self.a, self.b, self.gamma, self.omega, self.f = curr, a, b, gamma, omega, f
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        v, w, z = X
        vdot = v - v**3/3 - w + self.curr + self.f*np.sin(z)
        wdot = self.gamma*(v + self.a - self.b*w)
        zdot = self.omega
        return (vdot, wdot, zdot)      


class Colpitts:
    """
    An electrical circuit used as a signal generator
    Kennedy. IEEE Trans Circuits & Systems. 1994: 771-774.
    Li, Zhou, Yang. Chaos, Solitons, & Fractals. 2007
    """
    def __init__(self, a=30, b=0.8, c=20, d=0.08, e=10):
        self.a, self.b, self.c, self.d, self.e = a, b, c, d, e
        
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X

        u = z - (self.e - 1)
        fz = -u*(1 - np.heaviside(u, 0))
        xdot = y - self.a*fz
        ydot = self.c - x - self.b*y - z
        zdot = y - self.d*z
        return (xdot, ydot, zdot)   

class Blasius(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        a, alpha1, k1, b, alpha2, k2, c, zs = self.a, self.alpha1, self.k1, self.b, self.alpha2, self.k2, self.c, self.zs
        xdot = a*x - alpha1*x*y/(1 + k1*x)
        ydot = -b*y + alpha1*x*y/(1 + k1*x) - alpha2*y*z/(1 + k2*y)
        zdot = -c*(z - zs) + alpha2*y*z/(1 + k2*y)
        return (xdot, ydot, zdot)       
    
class TurchinHanski(DynSys):
    def rhs(self, X, t):
        n, p, z = X
        ndot = self.r*(1 - self.e*np.sin(z))*n - self.r*(n**2) - self.g*(n**2)/(n**2 + self.h**2) - self.a*n*p/(n + self.d)
        pdot = self.s*(1 - self.e*np.sin(z))*p - self.s*(p**2)/n
        zdot = 2*np.pi
        return (ndot, pdot, zdot) 
    
class HastingsPowell(DynSys):
    def f(self, x):
        return 0.5*(np.abs(x + 1) - np.abs(x - 1))
    def rhs(self, X, t):
        x, y, z = X
        a1, b1, d1, a2, b2, d2 = self.a1, self.b1, self.d1, self.a2, self.b2, self.d2
        xdot = x*(1 - x) - y*a1*x/(1 + b1*x)
        ydot = y*a1*x/(1 + b1*x) - z*a2*y/(1 + b2*y) - d1*y
        zdot = z*a2*y/(1 + b2*y) - d2*z
        return (xdot, ydot, zdot)   
    
class CellularNeuralNetwork:
    """
    Arena, Caponetto, Fortuna, and Porto., Int J Bifurcat Chaos. 1998:  1527-1539.
    """
    def __init__(self, a=4.4, b=3.21, c=1.1, d=1.24):
        self.a, self.b, self.c, self.d = a, b, c, d
    
    def f(self, x):
        return 0.5*(np.abs(x + 1) - np.abs(x - 1))
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -x + self.d*self.f(x) - self.b*self.f(y) - self.b*self.f(z)
        ydot = -y - self.b*self.f(x) + self.c*self.f(y) - self.a*self.f(z)
        zdot = -z - self.b*self.f(x) + self.a*self.f(y) + self.f(z)
        return (xdot, ydot, zdot)   

class BeerRNN(DynSys):
    def _sig(self, x):
        return 1.0/(1. + np.exp(-x))
    def rhs(self, X, t):
        Xdot = (-X + np.matmul(self.w, self._sig(X + self.theta)))/self.tau
        return Xdot 


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
## Non-chaotic systems
##
############################## 



class Torus2(object):
    """
    Simulate a minimal quasiperiodic flow on a torus
    """
    def __init__(self, r=1.0, a=0.5, n=15.3):
        """
        - r : the toroid radius
        - a : the (smaller) cross sectional radius
        - n : the number of turns per turn. Any non-integer
                value produces a quasiperiodic toroid
        """
        self.r = r
        self.a = a
        self.n = n
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        
        x, y, z = X
        
        xdot = (-self.a*self.n*np.sin(self.n*t))*np.cos(t) - (self.r + self.a*np.cos(self.n*t))*np.sin(t)
        ydot = (-self.a*self.n*np.sin(self.n*t))*np.sin(t) + (self.r + self.a*np.cos(self.n*t))*np.cos(t)
        zdot = self.a*self.n*np.cos(self.n*t)
        return (xdot, ydot, zdot)

# class Universe:
#     """
#     A simplification of the Friedman-Robertson-Walker equations
#     that admits chaotic solutions
#     Aydiner, Scientific Reports 2018
#     """
#     def __init__(self, a=None):
#         if a==None:
#             self.a = np.array([
#                 [-0.5, -0.1, 0.1],
#                 [0.5, 0.5, 0.1],
#                 [1.39, 0.1, 0.1]
#             ])
#         else:
#             pass
        
#     def __call__(self, X, t):
#         """
#         - X : tuple of length 3, corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         Xdot = X*np.matmul(self.a, 1 - X)
#         return Xdot
    
class Hopfield(DynSys):
    def f(self, x):
        return (1 + np.tanh(x))/2
    def rhs(self, X, t):
        Xdot = -X/self.tau + self.f(self.eps*np.matmul(self.k, X)) - self.beta
        return Xdot
    
class MacArthur(DynSys):
    def growth_rate(self, rr):
        """
        Liebig's law of the maximum
        r : np.ndarray, a vector of resource abundances
        """
        u0 = rr/(self.k.T + rr)
        u = self.r * u0.T
        return np.min(u.T, axis=1)
        
    def rhs(self, X, t):
        nn, rr = X[:5], X[5:]
        mu = self.growth_rate(rr)
        nndot = nn*(mu - self.m)
        rrdot = self.d*(self.s - rr) - np.matmul(self.c, (mu*nn))
        return np.hstack([nndot, rrdot])



### Non autonomous


## Maps

class deJong2D(object):
    """
    Simulate the dynamics of the 2D de Jong attractor
    Parameters default to a known chaotic case
    """
    def __init__(self, a=1.641, b=1.902, c=0.316, d=1.525):
        """
        Inputs
        - a : float
        - b : float
        - c : float
        - d : float
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 2, corresponding to xn, yn
        - t : float (the current time)
        """
        
        x, y = X
        x_next = np.sin(self.a*y) - np.cos(self.b*x)
        y_next = np.sin(self.c*x) - np.cos(self.d*y)
        return (x_next, y_next)

    def integrate(self, X0, tpts):
        """
        X0 : 2-tuple, the initial values of the coordinates
        tpts : np.array, the time mesh
        """
        x0, y0 = X0
        curr = (x0, y0)
        all_vals = [curr]
        for t in tpts:
            curr = self(curr, t)
            all_vals.append(curr)
        sol = np.array(all_vals)
        return sol.T


class Henon(object):
    """
    Simulate the dynamics of the Henon attractor
    Parameters default to a known chaotic case
    """
    def __init__(self, a=1.4, b=0.3):
        """
        Inputs
        - a : float
        - b : float
        """
        self.a = a
        self.b = b

    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 2, corresponding to xn, yn
        - t : float (the current time)
        """
        
        x, y = X
        x_next = 1-self.a*x**2 + y
        y_next = self.b*x
        return (x_next, y_next)

    def integrate(self, X0, tpts):
        """
        X0 : 2-tuple, the initial values of the coordinates
        tpts : np.array, the time mesh
        """
        x0, y0 = X0
        curr = (x0, y0)
        all_vals = [curr]
        for t in tpts:
            curr = self(curr, t)
            all_vals.append(curr)
        sol = np.array(all_vals)
        return sol.T

# Ikeda, Pickover, Logistic