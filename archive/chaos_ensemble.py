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

# try:
#     import jax.numpy as np
#     from jax import jit
#     has_jax = True
# except:
#     import numpy as np
#     has_jax = False
# from functools import partial
import numpy as np

import warnings
from collections import deque


## Numba support removed until jitclass API stabilizes
# try:
#     from numba import jit, njit, jitclass
#     has_numba = True
# except ModuleNotFoundError:
#     warnings.warn("Numba installation not found, numerical integration will run slower.")
#     has_numba = False
#     # Define placeholder functions
#     def jit(func):
#         return func
#     def njit(func):
#         return func
#     def autojit(func):
#         return func
#     def jitclass(c):
#         return c

## Try a more concise form
# from abc import abstractmethod, ABC
# class DynSys(ABC):
#     dt: float

#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)

#     @abstractmethod
#     def __call__(self, X, t):
#         ...
        
# class LorenzAltForm(DynSys):
#     dt = .01
#     sigma=10 
#     rho=28
#     beta=2.667

#     def __call__(self, X, t):
#         x, y, z = X
#         xdot = self.sigma*(y - x)
#         ydot = x*(self.rho - z) - y
#         zdot = x*y - self.beta*z
#         return (xdot, ydot, zdot)  
        
class Lorenz(object):
    """
    Simulate the dynamics of the Lorenz equations
    
    Inputs
    - sigma : float, the Prandtl number
    - rho : float, the Rayleigh number
    - beta : float, the spatial scale
    """
    def __init__(self, sigma=10, rho=28, beta=2.667):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    #@partial(jit, static_argnums=(0,))
    def __call__(self, X, t):
        x, y, z = X
        xdot = self.sigma*(y - x)
        ydot = x*(self.rho - z) - y
        zdot = x*y - self.beta*z
        return (xdot, ydot, zdot)
    
    def integrate(self, X0, tpts):
        """
        X0 : 3-tuple, the initial values of the three coordinates
        tpts : np.array, the time mesh
        """
        x0, y0, z0 = X0
        sol = odeint(self, (x0, y0, z0), tpts)
        return sol.T
    
class LorenzBounded:
    """
    A modified version of the Lorenz system that has a finite basin of attraction
    Sprott & Xiong, Chaos 2015
    """
    def __init__(self, sigma=10, beta=8/3, rho=28, r=64):
        self.sigma, self.beta, self.rho, self.r = sigma, beta, rho, r
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        f = 1 - (x**2 + y**2 + z**2)/self.r**2
        xdot = self.sigma*(y - x)*f
        ydot = (x*(self.rho - z) - y)*f
        zdot = (x*y - self.beta*z)*f
        return (xdot, ydot, zdot)
        
class CoupledLorenz:
    """
    Two coupled Lorenz maps
    """
    def __init__(self, sigma=10, beta=2.667, rho1=35, rho2=1.15, eps=2.85):
        """
        """
        self.sigma, self.beta, self.rho1, self.rho2, self.eps = sigma, beta, rho1, rho2, eps
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x1, y1, z1, x2, y2, z2 = X
        
        x1dot = self.sigma*(y1 - x1)
        y1dot = x1*(self.rho1 - z1) - y1
        z1dot = x1*y1 - self.beta*z1
        x2dot = self.sigma*(y2 - x2) + self.eps*(x1 - x2)
        y2dot = x2*(self.rho2 - z2) - y2
        z2dot = x2*y2 - self.beta*z2
        return (x1dot, y1dot, z1dot, x2dot, y2dot, z2dot)
  

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
  

class Rossler(object):  
    """
    Simulate the dynamics of the Rossler attractor
    Inputs
    - a : float
    - b : float
    - c : float
    """
    def __init__(self, a=.2, b=.2, c=5.7):
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        
        x, y, z = X
        
        xdot = -y - z
        ydot = x + self.a*y
        zdot = self.b + z*(x - self.c)
        return (xdot, ydot, zdot)
    
    def integrate(self, X0, tpts):
        """
        X0 : 3-tuple, the initial values of the three coordinates
        tpts : np.array, the time mesh
        """
        x0, y0, z0 = X0
        sol = odeint(self, (x0, y0, z0), tpts)
        return sol.T

    
class Thomas(object):  
    """
    Simulate the dynamics of the Thomas attractor
    """
    def __init__(self, a=1.85, b=10.):
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
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        
        x, y, z = X
        
        xdot = -self.a*x + self.b*np.sin(y)
        ydot = -self.a*y + self.b*np.sin(z)
        zdot = -self.a*z + self.b*np.sin(x)
        return (xdot, ydot, zdot)


class DoublePendulum:
    """
    The dynamics of a double pendulum
    """
    def __init__(self, m=1.0, d=1.0):
        """
        Inputs
        - m1 : float
        - m1 : float
        """
        self.m = m
        self.d = d
        
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
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
    
class Halvorsen(object):  
    """
    Simulate the dynamics of the Halvorsen attractor
    """
    def __init__(self, a=1.4, b=4.):
        """
        Inputs
        - a : float
        - b : float
        """
        self.a = a
        self.b = b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """ 
        x, y, z = X
        xdot = -self.a*x - self.b*(y + z) - y**2
        ydot = -self.a*y - self.b*(z + x) - z**2
        zdot = -self.a*z - self.b*(x + y) - x**2
        return (xdot, ydot, zdot)

class MultiChua:
    """
    A coupled version of Chua's circuit
    Cellular Neural Networks, Multi-Scroll Chaos And Synchronization
    Müstak E. Yalcin, Johan A. K. Suykens, Joos Vandewalle
    A New Four-Scroll Chaotic Attractor Consisted of Two-Scroll Transient Chaotic and Two-Scroll Ultimate Chaotic
    Yuhua Xu, Bing Li, Yuling Wang, Wuneng Zhou and Jian-an Fang
    """
    def __init__(self, a=9, b=14.286, 
                 m=[-1/7, 2/7, -4/7, 2/7, -4/7, 2/7],
                c=[0, 1.0, 2.15, 3.6, 8.2, 13.0]
                ):
        self.a = a
        self.b = b
        self.m, self.c = m, c
    
    def diode(self, x):
        m, c = self.m, self.c
        total = m[-1]*x
        for i in range(1, 6):
            total += 0.5*(m[i-1] - m[i])*(np.abs(x + c[i]) - np.abs(x - c[i]))
        return total
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - self.diode(x))
        ydot = x - y + z
        zdot = -self.b*y
        return (xdot, ydot, zdot)

class DoubleGyre:
    """
    range [0,2], [0,1]
    """
    def __init__(self, a=.1, eps=.1, omega=np.pi/5):
        self.a = a
        self.eps = eps
        self.omega = omega
        
    def __call__(self, X, t):
        x, y, z = X
        a = self.eps*np.sin(z)
        b = 1 - 2*self.eps*np.sin(z)
        f = a*x**2 + b*x
        dx = -self.a * np.pi*np.sin(np.pi*f)*np.cos(np.pi*y)
        dy = self.a * np.pi*np.cos(np.pi*f)*np.sin(np.pi*y)*(2*a*x + b)
        dz = self.omega
        return np.stack([dx, dy, dz]).T

class Chua:
    """
    Simulate the dynamics of Chua's circuit
    """
    def __init__(self, alpha=15.6, beta=28.0, m0=-8/7., m1=-5/7):
        """
        """
        self.alpha = alpha
        self.beta = beta
        self.m0 = m0
        self.m1 = m1
        self.diode = lambda x: m1*x + (0.5)*(m0 - m1)*(np.abs(x + 1) - np.abs(x - 1))
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.alpha*(y - x - self.diode(x))
        ydot = x - y + z
        zdot = -self.beta*y
        return (xdot, ydot, zdot)



class Duffing(object):
    """
    The Duffing-Ueda oscillator
    """
    def __init__(self, alpha=1.6, beta=0.0, gamma=7.5, delta=0.05, omega=1.):
        """
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma= gamma
        self.delta = delta
        self.omega = omega
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        
        x, y, z = X
        
        xdot = y
        ydot = -self.delta*y - self.beta*x - self.alpha*x**3 + self.gamma*np.cos(z)
        zdot = self.omega
        return (xdot, ydot, zdot)


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
    
    
class JerkCircuit:    
    """
    Sprott IEEE Circ. Sys. 2011
    """
    def __init__(self, y0=0.026, eps=1e-9):
        self.y0, self.eps = y0, eps
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -z - x  - self.eps*(np.exp(y/self.y0) - 1)
        return (xdot, ydot, zdot)

    
class ForcedBrusselator:
    def __init__(self, a=0.4, b=1.2, f=0.05, w=0.81):
        self.a, self.b, self.f, self.w = a, b, f, w
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a + x**2*y - (self.b + 1)*x + self.f*np.cos(z)
        ydot = self.b*x - x**2*y
        zdot = self.w
        return (xdot, ydot, zdot)
    
# def Reiterer:

# class Windmi:
#     """
#     The Windmi model of the Earth's magnetosphere
#     the parameter vsw controls a bifurcation
#     """
#     def __init__(self, vsw=4.5, a1=.247, a2=.391, b1=10.8, b2=0.0752, b3=1.06, d1=2200, f1=2.47, g1=1080, g2=4, g3=3.79):
#         self.a1, self.vsw, self.a2, self.b1, self.b2, self.b3, self.d1, self.f1, self.g1, self.g2, self.g3 = a1, vsw, a2, b1, b2, b3, d1, f1, g1, g2, g3
    
#     def __call__(self, X, t):
#         """
#         The dynamical equation for the system
#         - X : tuple corresponding to the three coordinates
#         - t : float (the current time)
#         """
#         i, v, p, kp, i1, vi = X
#         idot = self.a1*(self.vsw - v) + self.a2*(v - vi)
#         vdot = self.b1*(i - i1) - self.b2*p**1/2 - self.b3*v
#         pdot = v**2 - np.sqrt(kp)*p*(1 + np.tanh(self.d1*(i-1)))/2
#         kpdot = v*p**(1/2) - kp
#         i1dot = self.a2*(self.vsw - v) + self.f1*(v - vi)
#         vidot = self.g1*i1 - self.g2*vi - self.g3*(i1*vi**3)**(1/2)
#         return (idot, vdot, pdot, kpdot, i1dot, vidot)  


class WindmiReduced:
    """
    A simplified form of the Windmi model of the Earth's magnetosphere. The parameter
    vsw controls the onset of chaos
    the parameter vsw controls a bifurcation
    Smith, Thiffeault, Horton. J Geophys Res. 2000
    Horton, Weigel, Sprott. Chaos 2001
    """
    def __init__(self, vsw=5, a1=.247, b1=10.8, b2=0.0752, b3=1.06, d1=2200):
        self.a1, self.vsw, self.b1, self.b2, self.b3, self.d1 = a1, vsw, b1, b2, b3, d1
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        i, v, p = X
        idot = self.a1*(self.vsw - v)
        vdot = self.b1*i - self.b2*p**1/2 - self.b3*v
        pdot = self.vsw**2 - p**(5/4)*self.vsw**(1/2)*(1 + np.tanh(self.d1*(i-1)))/2
        return (idot, vdot, pdot)  
    


class MooreSpiegel:
    def __init__(self, a=10, b=4, eps=9):
        self.a, self.b, self.eps = a, b, eps
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y
        ydot = self.a*z
        zdot = -z + self.eps*y - y*x**2 - self.b*x
        return (xdot, ydot, zdot)

class CoevolvingPredatorPrey:
    """
    A system of predator-prey equations with co-evolving prey
    Gilpin, Feldman. PLOS Comp Biol 2017
    """
    def __init__(self, a1=5/2, a2=0.05, a3=0.4, delta=1, d1=0.16, d2=0.004, 
        b1=6.0, b2=1.333, k1=6.0, k2=9.0, k4=9.0, vv=1/3):
        self.a1, self.a2, self.a3, self.delta, self.d1, self.d2, self.b1, self.b2, self.k1, self.k2, self.k4, self.vv = a1, a2, a3, delta, d1, d2, b1, b2, k1, k2, k4, vv
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, alpha = X
        xdot = x*(-((self.a3*y)/(1 + self.b2*x)) + (self.a1*alpha*(1 - self.k1*x*(-alpha + alpha*self.delta)))/(1 + self.b1*alpha) - self.d1*(1 - self.k2*(-alpha**2 + (alpha*self.delta)**2) + self.k4*(-alpha**4 + (alpha*self.delta)**4))) 
        ydot = (-self.d2 + (self.a2*x)/(1 + self.b2*x))*y
        alphadot = self.vv*(-((self.a1*self.k1*x*alpha*self.delta)/(1 + self.b1*alpha)) - self.d1*(-2*self.k2*alpha*self.delta**2 + 4*self.k4*alpha**3*self.delta**4))

        return (xdot, ydot, alphadot)

class KawczynskiStrizhak:
    """
    A chemical oscillator model describing mixed-modes in the BZ equations
    P. E. Strizhak and A. L. Kawczynski, J. Phys. Chem. 99, 10830 (1995).
    """
    def __init__(self, kappa=.2, gamma=.49, mu=2.1, beta=-0.4):
        self.gamma, self.mu, self.beta, self.kappa = gamma, mu, beta, kappa
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.gamma*(y - x**3 + 3*self.mu*x)
        ydot = -2*self.mu*x - y - z + self.beta
        zdot = self.kappa*(x - z)
        return (xdot, ydot, zdot)

    
class BelousovZhabotinsky:
    """
    A reduced-order model of the BZ reaction that exhibits period doubling
    The bifurcation parameter for controlling the onset of chaos is kf. The system undergoes
    regular cycling when kf=3e-4, and chaotic oscillations when kf=3.5e-4
    from 
    "A three-variable model of deterministic chaos in the Belousov–Zhabotinsky reaction"
    Györgyi, Field. Nature 1992
    """
    def __init__(self, 
                 y0=7.72571e-6, yb1=6.92813e-7, yb2=2.00869, yb3=0.01352, kf=3.5e-4, z0=8.33e-6,
                 c1=-8.03474,c2=0.05408,c3=-0.0115886,c4=832.587,c5=-0.029155,c6=0.00321617,
                 c7=-0.01352,c8=-0.0831709,c9=-0.0199985, c10=0.0223915,c11= 7.53559e-5,c12=8.07384e-6,
                 c13=-0.000499825,
                 ci=0.000833, t0=2308.62
                 ):
        """
        """
        self.vars  = [y0, yb1, yb2, yb3, kf, z0, c1, 
                      c2, c3, c4, c5, c6, c7, c8, c9, 
                      c10, c11, c12, c13, ci, t0]
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """      
        x, z, v = X
        
        [y0, yb1, yb2, yb3, kf, z0, 
         c1, c2, c3, c4, c5, c6, c7, 
         c8, c9, c10, c11, c12, c13, ci, t0] = self.vars
        
        ybar = (1/y0)*yb1*z*v/(yb2*x + yb3 + kf)
        rf = (ci - z0*z)*np.sqrt(x)
        xdot = c1*x*ybar + c2*ybar + c3*x**2 + c4*rf + c5*x*z - kf*x
        zdot = (c6/z0)*rf + c7*x*z + c8*z*v + c9*z - kf*z
        vdot = c10*x*ybar + c11*ybar + c12*x**2 + c13*z*v - kf*v
        return (xdot*t0, zdot*t0, vdot*t0)
    
class RabinovichFabrikant(object):
    """
    """
    def __init__(self, g=0.87, a=1.1):
        self.g = g
        self.a = a
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        
        xdot = y*(z - 1 +x**2) + self.g*x
        ydot = x*(3*z + 1 - x**2) + self.g*y
        zdot = -2*z*(self.a + x*y)
        return (xdot, ydot, zdot)
    
class NoseHoover:
    def __init__(self, a=1.5):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y
        ydot = -x + y*z
        zdot = self.a - y**2
        return (xdot, ydot, zdot)

class Dadras:
    def __init__(self, p=3.0, o=2.7, r=1.7, c=2., e=9.):
        self.p, self.o, self.r, self.c, self.e = p, o, r, c, e
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y - self.p*x + self.o*y*z
        ydot = self.r*y - x*z + z
        zdot = self.c*x*y - self.e*z
        return (xdot, ydot, zdot)
    
class SprottTorus:
    """
    Depending on the initial conditions, this is a torus
    or a complex attractor
    Sprott Physics Letters A 2014
    """
    def __init__(self):
        pass
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        
        xdot = y + 2*x*y + x*z
        ydot = 1 - 2*x**2 + y*z
        zdot = x - x**2 - y**2
        return (xdot, ydot, zdot)
    
    
class SprottB:
    def __init__(self):
        pass
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y*z
        ydot = x - y
        zdot = 1 - x*y
        return (xdot, ydot, zdot)
    
class SprottC:
    def __init__(self):
        pass
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y*z
        ydot = x -y
        zdot = 1 - x**2
        return (xdot, ydot, zdot)
    
class SprottD:
    def __init__(self):
        pass
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -y
        ydot = x + z
        zdot = x*z + 3*y**2
        return (xdot, ydot, zdot)
    
class SprottE:
    def __init__(self):
        pass
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y*z
        ydot = x**2 - y
        zdot = 1 - 4*x
        return (xdot, ydot, zdot)
    
class SprottF:
    def __init__(self, a=0.5):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y + z
        ydot = -x + self.a*y
        zdot = x**2 - z
        return (xdot, ydot, zdot)
    
class SprottG:
    def __init__(self, a=0.4):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        
        xdot = self.a*x + z
        ydot = x*z - y
        zdot = -x + y
        return (xdot, ydot, zdot)
    
class SprottH:
    def __init__(self, a=0.5):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -y + z**2
        ydot = x + self.a*y
        zdot = x - z
        return (xdot, ydot, zdot)
    
class SprottI:
    def __init__(self, a=0.2):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -self.a*y
        ydot = x + z
        zdot = x + y**2 - z
        return (xdot, ydot, zdot)
    
class SprottJ:
    def __init__(self):
        pass
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = 2*z
        ydot = -2*y + z
        zdot = -x + y + y**2
        return (xdot, ydot, zdot)

class SprottK:
    def __init__(self, a=0.3):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = x*y - z
        ydot = x - y
        zdot = x + self.a*z
        return (xdot, ydot, zdot)    

class SprottL:
    def __init__(self, a=0.9, b=3.9):
        self.a, self.b = a, b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y + self.b*z
        ydot = self.a*x**2 - y
        zdot = 1 - x
        return (xdot, ydot, zdot)

class SprottM:
    def __init__(self, a=1.7):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -z
        ydot = -x**2 - y
        zdot = self.a*(1 + x) + y
        return (xdot, ydot, zdot)

class SprottN:
    def __init__(self):
        pass
    
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        
        xdot = -2*y
        ydot = x + z**2
        zdot = 1 + y - 2*z
        return (xdot, ydot, zdot)

class SprottO:
    def __init__(self, a=2.7):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y
        ydot = x - z
        zdot = x + x*z + self.a*y
        return (xdot, ydot, zdot)

class SprottP:
    def __init__(self, a=2.7):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*y + z
        ydot = -x + y**2
        zdot = x + y
        return (xdot, ydot, zdot)
    
class SprottQ:
    def __init__(self, a=3.1, b=0.5):
        self.a, self.b = a, b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -z
        ydot = x - y
        zdot = self.a*x + y**2 + self.b*z
        return (xdot, ydot, zdot)
    
class SprottR:
    def __init__(self, a=0.9, b=0.4):
        self.a, self.b = a, b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a - y
        ydot = self.b + z
        zdot = x*y - z
        return (xdot, ydot, zdot)
    
class SprottS:
    def __init__(self):
        pass
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = -x - 4*y
        ydot = x + z**2
        zdot = 1 + x
        return (xdot, ydot, zdot)

class Arneodo:
    """
    Aka the "ACT" attractor
    """
    def __init__(self, a=-5.5, b=4.5, c=1.0, d=-1.0):
        self.a, self.b, self.c, self.d = a, b, c, d
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -self.a*x - self.b*y - self.c*z  + self.d*x**3
        return (xdot, ydot, zdot)
    
class Rucklidge:
    def __init__(self, a=2.0, b=6.7):
        self.a, self.b = a, b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = - self.a*x + self.b*y - y*z
        ydot = x
        zdot = -z + y**2
        return (xdot, ydot, zdot)

class Sakarya:
    """
    
    """
    def __init__(self, a=-1.0, b=1.0, c=1.0, r=0.3, p=1.0, q=0.4, h=1.0, s=1.0):
        self.a, self.b, self.c, self.h, self.r, self.p, self.q, self.s = a, b, c, h, r, p, q, s
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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
        

class RayleighBenard:
    """
    A low dimensional model of a convection cell
    """
    def __init__(self, a=30.0, r=18.0, b=5.0):
        self.a, self.r, self.b = a, r, b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = self.r*y - x*z
        zdot = x*y - self.b*z
        return (xdot, ydot, zdot)
    
    
class Finance:
    """
    Guoliang Cai,Juanjuan Huang
    International Journal of Nonlinear Science, Vol. 3 (2007)
    """
    def __init__(self, a=0.001, b=0.2, c=1.1):
        self.a, self.b, self.c = a, b, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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


class Bouali2:
    """
    Bouali 2012
    Further described here:
    https://www.ioc.ee/~dima/YFX1520/LectureNotes_9.pdf
    """
    def __init__(self, a=3.0, b=2.2, g=1.0, m=-0.004/1.5, y0=1.0, bb=0, c=0):
        self.a, self.b, self.g, self.m, self.y0, self.bb, self.c = a, b, g, m, y0, bb, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*x*(self.y0 - y) - self.b*z
        ydot = -self.g*y*(1 - x**2)
        zdot = -self.m*x*(1.5 - self.bb*z) - self.c*z
        return (xdot, ydot, zdot)
    
class Bouali(Bouali2):
    """
    A named attractor related to the DequanLi attractor
    """
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

class LuChenCheng:
    """
    Lu, Chen, Cheng. Int J Bifurcat Chaos. 2004: 1507–1537.
    """
    def __init__(self, a=-10.0, b=-4.0, c=18.1):
        self.a, self.b, self.c = a, b, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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
    
class ZhouChen:
    """
    Zhou, Chen. Int J Bifurcat Chaos, 2004
    """
    def __init__(self, a=2.97, b=0.15, c=-3.0, d=1, e=-8.78):
        self.a, self.b, self.c, self.d, self.e = a, b, c, d, e
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        xdot = self.a*x + self.b*y + y*z
        ydot = self.c*y - x*z + self.d*y*z
        zdot = self.e*z - x*y
        return (xdot, ydot, zdot)
    
class BurkeShaw:
    """
    Shaw, Robert. Zeitschrift für Naturforschung (1981): 80-112.
    """
    def __init__(self, n=10, e=13):
        self.n, self.e = n, e
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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
    
    
class SanUmSrisuchinwong:
    """
    San-Um, Srisuchinwong. J. Comp 2012
    """
    def __init__(self, a=2.0):
        self.a = a
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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

class HyperRossler:
    """
    Rössler, 1979
    """
    def __init__(self, a=0.25, b=3.0, c=0.5, d=0.05):
        self.a, self.b, self.c, self.d = a, b, c, d
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z, w = X
        xdot = - y - z
        ydot = x + self.a*y + w
        zdot = self.b + x*z
        wdot = -self.c*z + self.d*w
        return (xdot, ydot, zdot, wdot)

class SaltonSea:
    """
    Upadhyay, Bairagi, Kundu, Chattopadhyay, 2007
    """
    def __init__(self, r=22, k=400, lam=0.06, m=15.5, a=15, mu=3.4, th=10.0, d=8.3):
        self.r, self.k, self.lam, self.m, self.a, self.mu, self.th, self.d = r, k, lam, m, a, mu, th, d
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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

class CaTwoPlus:
    """
    Intracellular calcium ion oscillations
    Houart, Dupont, Goldbeter. Bull Math Biol 1999
    """
    def __init__(self, beta=0.65, K2=0.1, K5=0.3194, Ka=0.1, Kd=1, Ky=0.3, 
                     Kz=0.6, k=10, kf=1, eps=13, n=4, m=2, p=1, V0=2, 
                     V1=2, Vm2=6, Vm3=30, V4=3, Vm5=50):
        self.params = [beta, K2, K5, Ka, Kd, Ky, 
                     Kz, k, kf, eps, n, m, p, V0, 
                     V1, Vm2, Vm3, V4, Vm5]
        
    def __call__(self, X, t):
        z, y, a = X
        [beta, K2, K5, Ka, Kd, 
         Ky, Kz, k, kf, eps, n, 
         m, p, V0, V1, Vm2, Vm3, V4, Vm5] = self.params
        Vin = V0 + V1*beta
        V2 = Vm2*(z**2)/(K2**2 + z**2)
        V3 = ( Vm3*(z**m)/(Kz**m + z**m) )*( y**2/(Ky**2 + y**2) )*(a**4/(Ka**4 + a**4))
        V5 = Vm5*(a**p/(K5**p + a**p))*(z**n/(Kd**n + z**n))
        
        zdot = Vin - V2 + V3 + kf*y - k*z
        ydot = V2 - V3 - kf*y
        adot = beta*V4 - V5 - eps*a
        return (zdot, ydot, adot)

class CellCycle:
    """
    A simplified model of the cell cycle
    The parameter Kim controls the bifurcation
    
    Romond, Rustici, Gonze, Goldbeter. 1999
    """
    
    def __init__(self, Kim=0.65, K=0.01, Vm1=0.3, vi=0.05, V2=0.15, Vm3=0.1, 
                 V4=0.05, Kd1=0.02, kd1=0.001,Kc=0.5, vd=0.025):
        [self.Kim, self.K, self.Vm1, self.vi, self.V2, 
         self.Vm3, self.V4, self.Kd1, self.kd1, self.Kc, self.vd] = [Kim, K, Vm1, vi, V2, 
                                                                     Vm3, V4, Kd1, kd1, Kc, vd]
    def __call__(self, X, t):
        
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
    
    
# from scipy.signal import square
class CircadianRhythm:
    """
    The Drosophila circadian rhythm under periodic light/dark forcing
    Leloup, Gonze, Goldbeter. 1999
    Gonze, Leloup, Goldbeter. 2000
    """
    def __init__(self, vs=6, vm=0.7, vd=6, vdn=1.5, ks=1, k=0.5, 
                 k1=0.3, k2=0.15, km=0.4, Ki=1, kd=1.4, kdn=0.4, n=4, vmin=1.0, vmax=4.7):
        self.params = [vs, vm, vd, vdn, ks, k, k1, k2, km, Ki, kd, kdn, n, vmin, vmax]
        
    def __call__(self, X, t):
        m, fc, fs, fn, th = X
        [vs, vm, vd, vdn, ks, k, k1, k2, km, Ki, kd, kdn, n, vmin, vmax] = self.params
        
        #vmin, vmax = 1.6, 4.7
        #vs = ((0.5 + 0.5*np.cos(th)) + vmin)*(vmax - vmin)

        #vmin, vmax = 1.0, 4.7
        vs = 2.5*((0.5 + 0.5*np.cos(th)) + vmin)*(vmax - vmin)
        mdot = vs*(Ki**n)/(Ki**n + fn**n) - vm*m/(km + m)
        fcdot = ks*m - k1*fc + k2*fn - k*fc
        fsdot = k*fc - vd*fs/(kd + fs)
        fndot = k1*fc - k2*fn - vdn*fn/(kdn + fn)
        thdot = 2*np.pi/24
        return (mdot, fcdot, fsdot, fndot, thdot)
    

class FluidTrampoline:
    """
    A droplet bouncing on a horizontal soap film
    Gilet, Bush. JFM 2009
    """
    def __init__(self, psi=0.01019, gamma=1.82,w=1.21):
        self.psi, self.gamma, self.w = psi, gamma, w
    def __call__(self, X, t):
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

class ItikBanksTumor:
    """
    A model of cancer cell populations
    Itik, Banks. Int J Bifurcat Chaos 2010
    """
    def __init__(self, a12=1, a13=2.5, r2=0.6, a21=1.5, r3=4.5, k3=1, a31=0.2, d3=0.5):
        self.a12, self.a13, self.r2, self.a21, self.r3,self.k3, self.a31, self.d3 =  a12, a13, r2, a21, r3, k3, a31, d3
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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

class Aizawa:
    def __init__(self, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        xdot = (z - b)*x - d*y
        ydot = d*x + (z - b)*y
        zdot = c + a*z - z**3/3 - (x**2 + y**2)*(1 + e*z) + f*z*x**3
        return (xdot, ydot, zdot)  
    
class AnishchenkoAstakhov:
    def __init__(self, mu=1.2, eta=0.5):
        self.mu, self.eta = mu, eta
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        mu, eta = self.mu, self.eta
        xdot = mu*x + y - x*z
        ydot = -x
        zdot = -eta*z + eta*np.heaviside(x, 0)*x**2
        return (xdot, ydot, zdot)  
    
class ShimizuMorioka:
    """
    Shimizu, Morioka. Phys Lett A. 1980: 201-204
    """
    def __init__(self, a=0.85, b=0.5):
        self.a, self.b = a, b
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        
        xdot = y
        ydot = x - self.a*y - x*z
        zdot = -self.b*z + x**2
        return (xdot, ydot, zdot)
    
class GenesioTesi:
    def __init__(self, a=0.44, b=1.1, c=1):
        self.a, self.b, self.c = a, b, c
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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

class Hadley:
    """
    The Hadley convective cell  circulation model
    the parameters b and f strongly influence attractor shape
    """
    def __init__(self, a=0.2, b=4, f=9, g=1):
        self.a, self.b, self.f, self.g = a, b, f, g
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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

class Blasius:
    """
    A chaotic food web
    Blasius, Huppert, Stone. Nature 1999
    """
    def __init__(self, a=1, b=1, c=10, alpha1=0.2, alpha2=1, k1=0.05, k2=0, zs=.006):
        self.a, self.alpha1, self.k1, self.b, self.alpha2, self.k2, self.c, self.zs = a, alpha1, k1, b, alpha2, k2, c, zs
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        x, y, z = X
        a, alpha1, k1, b, alpha2, k2, c, zs = self.a, self.alpha1, self.k1, self.b, self.alpha2, self.k2, self.c, self.zs
        xdot = a*x - alpha1*x*y/(1 + k1*x)
        ydot = -b*y + alpha1*x*y/(1 + k1*x) - alpha2*y*z/(1 + k2*y)
        zdot = -c*(z - zs) + alpha2*y*z/(1 + k2*y)
        return (xdot, ydot, zdot)       
    
class TurchinHanski:
    """
    A chaotic three species food web
    Turchin, Hanski. The American Naturalist 1997
    Turchin, Hanski. Ecology 2000
    """
    def __init__(self, r=4.06, s=1.25, e=0.5, g=0.1, a=8, d=0.04, h=0.8):
        self.r, self.s, self.e, self.g, self.a, self.d, self.h = r, s, e, g, a, d, h
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
        n, p, z = X
        ndot = self.r*(1 - self.e*np.sin(z))*n - self.r*(n**2) - self.g*(n**2)/(n**2 + self.h**2) - self.a*n*p/(n + self.d)
        pdot = self.s*(1 - self.e*np.sin(z))*p - self.s*(p**2)/n
        zdot = 2*np.pi
        return (ndot, pdot, zdot) 
    
class HastingsPowell:
    """
    A chaotic three species food web
    Hastings, Powell. Ecology 1991
    """
    def __init__(self, a1=5.0, b1=3.0, d1=0.4, a2=0.1, b2=2.0, d2=0.01):
        self.a1, self.b1, self.d1, self.a2, self.b2, self.d2 = a1, b1, d1, a2, b2, d2
    
    def f(self, x):
        return 0.5*(np.abs(x + 1) - np.abs(x - 1))
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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

class BeerRNN:
    """
    Beer, R. D. (1995).
     On the dynamics of small continuous-time recurrent neural networks. 
     Adapt. Behav., 3(4), 469–509. http://doi.org/10.1177/105971239500300405
    """
    def __init__(self, 
        tau=np.array([1.0, 2.5, 1.0]), 
        theta=np.array([-4.108, -2.787, -1.114]), 
        w=None
        ):
        if not w:
            self.w = np.array([[5.422,  -0.018,  2.75],
                               [-0.24, 4.59, 1.21],
                               [0.535, -2.25, 3.885]
                              ]
                             )
            
        else:
            self.w = w
        self.theta = theta
        self.tau = tau
    
    def _sig(self, x):
        return 1.0/(1. + np.exp(-x))
    def __call__(self, X, t):
        """
        The dynamical equation for the system
        - X : tuple corresponding to the three coordinates
        - t : float (the current time)
        """
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
    
    def integrate(self, X0, tpts):
        """
        X0 : 3-tuple, the initial values of the three coordinates
        tpts : np.array, the time mesh
        """
        x0, y0, z0 = X0
        sol = odeint(self, (x0, y0, z0), tpts)
        return sol.T

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
    
class Hopfield:
    """
    Small chaotic Hopfield network with frustrated connectivity,
    as described by Lewis & Glass, Neur Comp (1992)
    """
    def __init__(self, tau=2.5, eps=10, beta=0.5, k=None):
        self.tau, self.eps, self.beta = tau, eps, beta
        if k == None:
            self.k = self.set_default_coupling()
        else:
            pass
        self.n = self.k.shape[0]
    
    def set_default_coupling(self):
        return np.array([
            [0, -1, 0, 0, -1, -1],
            [0, 0, 0, -1, -1, -1],
            [-1, -1, 0, 0, -1, 0],
            [-1, -1, -1, 0, 0, 0],
            [-1, -1, 0, -1, 0, 0],
            [0, -1, -1, -1, 0, 0]
        ])
    
    def f(self, x):
        return (1 + np.tanh(x))/2
    
    def __call__(self, X, t):
        """
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        Xdot = -X/self.tau + self.f(self.eps*np.matmul(self.k, X)) - self.beta
        return Xdot
    
class MacArthur(object):
    """
    Simulate the dynamics of the modified MacArthur resource competition model,
    as studied by Huisman & Weissing, Nature 1999
    """
    def __init__(self, r=None, k=None, c=None, d=None, s=None, m=None):
        """
        Inputs
        """
        
        if r==None:
            self.set_defaults()
        else:
            assert len(s) == k.shape[0], "vector \'s\' has improper dimensionality"
            assert k.shape == c.shape, "K and C matrices must have matching dimensions"
            self.r = r
            self.k = k
            self.c = c
            self.d = d
            self.s = s
            self.m = m
            
        self.n_resources, self.n_species = self.k.shape
                
    def set_defaults(self):
        """
        Set default values for parameters. Taken from Fig. 4 of 
        Huisman & Weissing. Nature 1999
        """
        
        self.k = np.array([[0.39,0.34,0.30,0.24,0.23,0.41,0.20,0.45,0.14,0.15,0.38,0.28],
                           [0.22,0.39,0.34,0.30,0.27,0.16,0.15,0.05,0.38,0.29,0.37,0.31],
                           [0.27,0.22,0.39,0.34,0.30,0.07,0.11,0.05,0.38,0.41,0.24,0.25],
                           [0.30,0.24,0.22,0.39,0.34,0.28,0.12,0.13,0.27,0.33,0.04,0.41],
                           [0.34,0.30,0.22,0.20,0.39,0.40,0.50,0.26,0.12,0.29,0.09,0.16]])
        self.c = np.array([[0.04,0.04,0.07,0.04,0.04,0.22,0.10,0.08,0.02,0.17,0.25,0.03],
                           [0.08,0.08,0.08,0.10,0.08,0.14,0.22,0.04,0.18,0.06,0.20,0.04],
                           [0.10,0.10,0.10,0.10,0.14,0.22,0.24,0.12,0.03,0.24,0.17,0.01],
                           [0.05,0.03,0.03,0.03,0.03,0.09,0.07,0.06,0.03,0.03,0.11,0.05],
                           [0.07,0.09,0.07,0.07,0.07,0.05,0.24,0.05,0.08,0.10,0.02,0.04]])
        self.s = np.array([6, 10, 14, 4, 9])
        self.d = 0.25
        self.r = 1
        self.m = 0.25
        
        # 5 species, 5 resources
        self.k = self.k[:,:5]
        self.c = self.c[:,:5]
    
    def set_ic(self):
        """
        Get default initial conditions from Huisman & Weissing. Nature 1999
        """
        if self.n_species<=5:
            ic_n = np.array([0.1 + i/100 for i in range(1,self.n_species+1)])
        else:
            ic_n = np.hstack([np.array([0.1 + i/100 for i in range(1,5+1)]), np.zeros(n_species-5)])
        ic_r = np.copy(self.s)
        return (ic_n, ic_r)
    
    def growth_rate(self, rr):
        """
        Calculate growth rate using Liebig's law of the maximum
        r : np.ndarray, a vector of resource abundances
        """
        u0 = rr/(self.k.T + rr)
        u = self.r * u0.T
        return np.min(u.T, axis=1)
        
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : vector of length n_species + n_resources, corresponding to all dynamic variables
        - t : float (the current time)
        """
        
        nn, rr = X[:self.n_species], X[self.n_species:]
        
        mu = self.growth_rate(rr)
        nndot = nn*(mu - self.m)
        rrdot = self.d*(self.s - rr) - np.matmul(self.c, (mu*nn))
        return np.hstack([nndot, rrdot])

    def integrate(self, X0, tpts):
        """
        X0 : 2-tuple of vectors, the initial values of the species and resources
        tpts : np.array, the time mesh
        """
        if not X0:
            X0 = self.set_ic()
        else:
            pass
        
        sol = odeint(self, np.hstack(X0), tpts)
        return sol.T


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