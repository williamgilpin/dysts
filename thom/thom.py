    
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
+ Database: How to load a function object from a string? exec does not do this
+ Add a function compute_dt that finds the timestep based on fft
+ Set standard integration timestep based on this value
+ Add a function that rescales outputs to the same interval
+ Add a function that finds initial conditions on the attractor
"""

import numpy as np
from .base import DynSys, staticjit

class Lorenz(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, rho, sigma):
        xdot = sigma * (y - x)
        ydot = x * (rho - z) - y
        zdot = x * y - beta * z
        return xdot, ydot, zdot

class LorenzBounded(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, r, rho, sigma):
        f = 1 - (x**2 + y**2 + z**2) / r**2
        xdot = sigma*(y - x)*f
        ydot = (x*(rho - z) - y)*f
        zdot = (x*y - beta*z)*f
        return xdot, ydot, zdot
        
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

class Lorenz96(DynSys):
    def rhs(self, X, t):
        Xdot = np.zeros_like(X)
        Xdot[0] = (X[1] - X[-2]) * X[-1] - X[0] + self.f
        Xdot[1] = (X[2] - X[-1]) * X[0] - X[1] + self.f
        Xdot[-1] = (X[0] - X[-3]) * X[-2] - X[-1] + self.f
        Xdot[2:-1] = (X[3:] - X[:-3])*X[1:-2] - X[2:-1] + self.f
        return Xdot
    
class Lorenz84(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, g):
        xdot = -a * x - y**2 - z**2 + a * f
        ydot = -y + x * y - b * x * z + g
        zdot = -z + b * x * y + x * z
        return xdot, ydot, zdot
    
class Rossler(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = - y - z
        ydot = x + a * y
        zdot = b + z * (x - c)
        return xdot, ydot, zdot

class Thomas(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a*x + b*np.sin(y)
        ydot = -a*y + b*np.sin(z)
        zdot = -a*z + b*np.sin(x)
        return xdot, ydot, zdot
    
class ThomasLabyrinth(Thomas):
    pass

class DoublePendulum(DynSys):
    @staticjit
    def _rhs(th1, th2, p1, p2, t, d, m):
        g = 9.82
        pre = (6/(m*d**2))
        denom = 16 - 9*np.cos(th1 - th2)**2
        th1_dot = pre*(2*p1 - 3*np.cos(th1 - th2)*p2)/denom
        th2_dot = pre*(8*p2 - 3*np.cos(th1 - th2)*p1)/denom
        p1_dot = -0.5*(m*d**2)*(th1_dot*th2_dot*np.sin(th1 - th2) + 3*(g/d)*np.sin(th1))
        p2_dot = -0.5*(m*d**2)*(-th1_dot*th2_dot*np.sin(th1 - th2) + 3*(g/d)*np.sin(th2))
        return th1_dot, th2_dot, p1_dot, p2_dot
  
class HenonHeiles(DynSys):  
    @staticjit
    def _rhs(x, y, px, py, t, lam):
        xdot = px
        ydot = py
        pxdot = -x - 2 * lam * x * y
        pydot = -y - lam * (x**2 - y**2)
        return xdot, ydot, pxdot, pydot

class Halvorsen(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -a*x - b*(y + z) - y**2
        ydot = -a*y - b*(z + x) - z**2
        zdot = -a*z - b*(x + y) - x**2
        return xdot, ydot, zdot

class Chua(DynSys):
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, m0, m1):
        ramp_x = m1 * x +  0.5 * (m0 - m1)*(np.abs(x + 1) - np.abs(x - 1))
        xdot = alpha*(y - x - ramp_x)
        ydot = x - y + z
        zdot = -beta*y
        return xdot, ydot, zdot
    
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
    @staticjit
    def _rhs(x, y, z, t, alpha, beta, delta, gamma, omega):
        xdot = y
        ydot = -delta * y - beta * x - alpha * x**3 + gamma * np.cos(z)
        zdot = omega
        return xdot, ydot, zdot

## Can this be incorporated in a DynSys object __call___ method?
# class MackeyGlass(object):
#     """
#     Simulate the dynamics of the Mackey-Glass time delay model
#     Inputs
#     - tau : float, the delay parameter
#     """
#     def __post_init__(self, tau, beta, gamma, dt, n=10):
#         self.tau = tau
#         self.beta = beta
#         self.gamma = gamma
#         self.n = n
#         self.dt = dt
        
#         self.mem = int(np.ceil(self.tau/self.dt))
#         self.history = deque(1.2 + (np.random.rand(self.mem) - 0.5))

#     def __call__(self, x, t):
#         xt = self.history.pop()
#         xdot = self.beta*(xt/(1 + xt**self.n)) - self.gamma*x
#         return xdot
    
#     def integrate(self, x0, tpts):
#         x_series = np.zeros_like(tpts)
#         x_series[0] = x0
#         self.history.appendleft(x0)
#         for i, t in enumerate(tpts):
#             if i==0:
#                 continue
            
#             dt = tpts[i] - tpts[i-1]
#             x_nxt = x_series[i-1] + self(x_series[i-1], t)*self.dt
            
#             x_series[i] = x_nxt 
#             self.history.appendleft(x_nxt)
        
#         return x_series

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
    
class BlinkingRotlet(DynSys):
    @staticjit
    def _rotlet(r, theta, a, b, bc):
        """A rotlet velocity field"""
        kappa = a ** 2 + (b ** 2 * r ** 2) / a ** 2 - 2 * b * r * np.cos(theta)
        gamma = (1 - r ** 2 / a ** 2) * (a ** 2 - (b ** 2 * r ** 2) / a ** 2)
        iota = (b ** 2 * r) / a ** 2 - b * np.cos(theta)
        zeta = b ** 2 + r ** 2 - 2 * b * r * np.cos(theta)
        nu = a**2 + b**2 - (2*b**2*r**2)/a**2
        vr = b*np.sin(theta)*(- bc * (gamma/kappa**2) - 1/kappa + 1/zeta)
        vth = bc * (gamma*iota)/kappa**2 + bc * r*nu/(a**2*kappa) + iota/kappa - (r - b*np.cos(theta))/zeta
        return vr, vth
    
    @staticjit
    def _protocol(t, tau, stiffness=20):
        return  0.5 + 0.5 * np.tanh(tau * stiffness * np.sin(2 * np.pi * t / tau))
    
    def rhs(self, X, t):
        r, theta = X
        weight = self._protocol(t, self.tau) 
        dr1, dth1 = self._rotlet(r, theta, self.a, self.b, self.bc)
        dr2, dth2 = self._rotlet(r, theta, self.a, -self.b, self.bc)
        dr = weight * dr1 + (1 - weight) * dr2
        dth = (weight * dth1 + (1 - weight) * dth2) / r
        return self.sigma * dr, self.sigma * dth
    
class BlinkingVortex(BlinkingRotlet):
    pass
        
class OscillatingFlow(DynSys):
    @staticjit
    def _rhs(x, y, z, t, b, k, omega, u):
        #x, y = np.mod(x, 2 * np.pi / self.k), np.mod(y, 2 * np.pi / self.k)
        f = x + b * np.sin(z)
        dx = u * np.cos(k * y) * np.sin(k * f)
        dy = -u * np.sin(k * y) * np.cos(k * f)
        dz = omega
        return dx, dy, dz

class BickleyJet(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        sechy = 1/np.cosh(y/self.ell)
        u = [self.k[i]*(x - t*self.sigma[i]) for i in range(3)]
        dx = self.u*sechy**2*(-1 - 2*(np.cos(u[0])*self.eps[0] + np.cos(u[1])*self.eps[1] + np.cos(u[2])*self.eps[2])*np.tanh(y/self.ell))
        dy = self.ell*self.u*sechy**2*(self.eps[0]*self.k[0]*np.sin(u[0]) + self.eps[1]*self.k[1]*np.sin(u[1]) + self.eps[2]*self.k[2]*np.sin(u[2]))
        dz = self.omega
        return np.stack([dx, dy, dz]).T

class ArnoldBeltramiChildress(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        dx = self.a*np.sin(z) + self.c*np.cos(y)
        dy = self.b*np.sin(x) + self.a*np.cos(z)
        dz = self.c*np.sin(y) + self.b*np.cos(x)
        return (dx, dy, dz)

class JerkCircuit(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y
        ydot = z
        zdot = -z - x  - self.eps*(np.exp(y/self.y0) - 1)
        return (xdot, ydot, zdot)

class ForcedBrusselator(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, f, w):
        xdot = a + x**2*y - (b + 1)*x + f*np.cos(z)
        ydot = b*x - x**2*y
        zdot = w
        return xdot, ydot, zdot

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
        ydot = a*z
        zdot = -z + eps * y - y * x**2 - b * x
        return xdot, ydot, zdot

class CoevolvingPredatorPrey(DynSys):
    def rhs(self, X, t):
        x, y, alpha = X
        xdot = x*(-((self.a3*y)/(1 + self.b2*x)) + (self.a1*alpha*(1 - self.k1*x*(-alpha + alpha*self.delta)))/(1 + self.b1*alpha) - self.d1*(1 - self.k2*(-alpha**2 + (alpha*self.delta)**2) + self.k4*(-alpha**4 + (alpha*self.delta)**4))) 
        ydot = (-self.d2 + (self.a2*x)/(1 + self.b2*x))*y
        alphadot = self.vv*(-((self.a1*self.k1*x*alpha*self.delta)/(1 + self.b1*alpha)) - self.d1*(-2*self.k2*alpha*self.delta**2 + 4*self.k4*alpha**3*self.delta**4))
        return (xdot, ydot, alphadot)

class KawczynskiStrizhak(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, gamma, kappa, mu):
        xdot = gamma * (y - x**3 + 3 * mu * x)
        ydot = -2 * mu * x - y - z + beta
        zdot = kappa * (x - z)
        return xdot, ydot, zdot

class BelousovZhabotinsky(DynSys):
    def rhs(self, X, t):    
        x, z, v = X
        ybar = (1/self.y0)*self.yb1*z*v/(self.yb2*x + self.yb3 + self.kf)
        rf = (self.ci - self.z0*z)*np.sqrt(x)
        xdot = self.c1*x*ybar + self.c2*ybar + self.c3*x**2 + self.c4*rf + self.c5*x*z - self.kf*x
        zdot = (self.c6/self.z0)*rf + self.c7*x*z + self.c8*z*v + self.c9*z - self.kf*z
        vdot = self.c10*x*ybar + self.c11*ybar + self.c12*x**2 + self.c13*z*v - self.kf*v
        return xdot*self.t0, zdot*self.t0, vdot*self.t0

class IsothermalChemical(DynSys):
    def rhs(self, X, t):
        alpha, beta, gamma = X
        alphadot = self.mu * (self.kappa + gamma) - alpha * beta**2 - alpha
        betadot = (alpha * beta**2 + alpha - beta) / self.sigma
        gammadot = (beta - gamma) / self.delta
        return alphadot, betadot, gammadot
    
class VallisElNino(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.b * y - self.c * (x + self.p)
        ydot = -y + x * z
        zdot = -z - x * y + 1
        return (xdot, ydot, zdot)

## Not chaotic
# class Robinson(DynSys):
#     @staticjit
#     def _rhs(x, y, z, t, a, b, c, d, v):
#         xdot = y
#         ydot = x - 2 * x**3 - a * y + b * x**2 * y - v * y * z
#         zdot = -c * z + d * x**2
#         return (xdot, ydot, zdot)
    
class RabinovichFabrikant(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, g):
        xdot = y*(z - 1 + x**2) + g * x
        ydot = x*(3 * z + 1 - x**2) + g * y
        zdot = -2 * z * (a + x * y)
        return (xdot, ydot, zdot)
    
class NoseHoover(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = -x + y*z
        zdot = a - y**2
        return xdot, ydot, zdot

class Dadras(DynSys):
    @staticjit
    def _rhs(x, y, z, t, c, e, o, p, r):
        xdot = y - p * x + o * y *z
        ydot = r * y - x * z + z
        zdot = c * x * y - e * z
        return xdot, ydot, zdot
    
class RikitakeDynamo(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, mu):
        xdot = - mu * x + y * z
        ydot = - mu * y + x * (z - a)
        zdot = 1 - x * y
        return xdot, ydot, zdot
    
class SprottTorus(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y + 2*x*y + x*z
        ydot = 1 - 2*x**2 + y*z
        zdot = x - x**2 - y**2
        return xdot, ydot, zdot
    
class SprottJerk(DynSys):
    @staticjit
    def _rhs(x, y, z, t, mu):
        xdot = y
        ydot = z
        zdot = -x  + y**2 - mu*z
        return xdot, ydot, zdot
    
## Not chaotic
# class JerkCircuit(DynSys):
#     def rhs(self, X, t):
#         x, y, z = X
#         xdot = y
#         ydot = z
#         zdot = -z - x  - self.eps*(np.exp(y/self.y0) - 1)
#         return (xdot, ydot, zdot)
    
class SprottB(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x - y
        zdot = 1 - x * y
        return xdot, ydot, zdot
    
class SprottC(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y * z
        ydot = x -y
        zdot = 1 - x**2
        return xdot, ydot, zdot
    
class SprottD(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -y
        ydot = x + z
        zdot = x * z + 3 * y**2
        return xdot, ydot, zdot
    
class SprottE(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = y*z
        ydot = x**2 - y
        zdot = 1 - 4*x
        return xdot, ydot, zdot
    
class SprottF(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y + z
        ydot = -x + a*y
        zdot = x**2 - z
        return xdot, ydot, zdot
    
class SprottG(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = a * x + z
        ydot = x * z - y
        zdot = -x + y
        return xdot, ydot, zdot
    
class SprottH(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -y + z**2
        ydot = x + a*y
        zdot = x - z
        return xdot, ydot, zdot
    
class SprottI(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -a * y
        ydot = x + z
        zdot = x + y**2 - z
        return xdot, ydot, zdot
    
class SprottJ(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = 2 * z
        ydot = -2 * y + z
        zdot = -x + y + y**2
        return (xdot, ydot, zdot)

class SprottK(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = x * y - z
        ydot = x - y
        zdot = x + a*z
        return xdot, ydot, zdot

class SprottL(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = y + b * z
        ydot = a * x**2 - y
        zdot = 1 - x
        return xdot, ydot, zdot

class SprottM(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = -z
        ydot = -x**2 - y
        zdot = a * (1 + x) + y
        return xdot, ydot, zdot

class SprottN(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -2 * y
        ydot = x + z**2
        zdot = 1 + y - 2 * z
        return xdot, ydot, zdot

class SprottO(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = y
        ydot = x - z
        zdot = x + x * z + a * y
        return xdot, ydot, zdot

class SprottP(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a):
        xdot = a * y + z
        ydot = -x + y**2
        zdot = x + y
        return xdot, ydot, zdot
    
class SprottQ(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = -z
        ydot = x - y
        zdot = a * x + y**2 + b * z
        return (xdot, ydot, zdot)
    
class SprottR(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = a - y
        ydot = b + z
        zdot = x * y - z
        return xdot, ydot, zdot
    
class SprottS(DynSys):
    @staticjit
    def _rhs(x, y, z, t):
        xdot = -x - 4 * y
        ydot = x + z**2
        zdot = 1 + x
        return xdot, ydot, zdot

class Arneodo(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = y
        ydot = z
        zdot = -a * x - b * y - c * z  + d * x**3
        return xdot, ydot, zdot

class Coullet(Arneodo):
    pass
    
class Rucklidge(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b):
        xdot = - a*x + b*y - y*z
        ydot = x
        zdot = -z + y**2
        return xdot, ydot, zdot

class Sakarya(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, h, p, q, r, s):
        xdot = a*x + h*y + s*y*z
        ydot = -b*y - p*x + q*x*z
        zdot = c*z - r*x*y
        return xdot, ydot, zdot
    
class LiuChen(Sakarya):
    pass
        
class RayleighBenard(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, r):
        xdot = a * (y - x)
        ydot = r * y - x * z
        zdot = x * y - b * z
        return xdot, ydot, zdot
    
    
class Finance(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = (1 / b - a) * x + z + x * y
        ydot = -b * y - x**2
        zdot = -x - c * z
        return xdot, ydot, zdot

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
    @staticjit
    def _rhs(x, y, z, t, a, b, bb, c, g, m, y0):
        xdot = a * x * (y0 - y) - b * z
        ydot = -g * y * (1 - x**2)
        zdot = -m * x*(1.5 - bb * z) - c * z
        return xdot, ydot, zdot
    
class Bouali(Bouali2):
    pass

class LuChenCheng(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = -(self.a * self.b)/(self.a + self.b)*x - y*z + self.c
        ydot = self.a*y + x*z
        zdot = self.b*z + x*y
        return (xdot, ydot, zdot)

class LuChen(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = -x*z + self.c*y
        zdot = x*y - self.b*z
        return (xdot, ydot, zdot)


class QiChen(DynSys):
    def rhs(self, X, t):
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
    
class Chen(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = (self.c - self.a)*x - x*z + self.c*y
        zdot = x*y - self.b*z
        return (xdot, ydot, zdot)
    
class ChenLee(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*x - y*z
        ydot = self.b*y + x*z
        zdot = self.c*z + x*y/3
        return (xdot, ydot, zdot)

class WangSun(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*x + self.q*y*z
        ydot = self.b*x + self.d*y - x*z
        zdot = self.e*z + self.f*x*y
        return (xdot, ydot, zdot)
    
class YuWang(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = self.b*x - self.c*x*z
        zdot = np.exp(x*y) - self.d*z
        return (xdot, ydot, zdot)

class YuWang2(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*(y - x)
        ydot = self.b*x - self.c*x*z
        zdot = np.cosh(x*y) - self.d*z
        return (xdot, ydot, zdot)

class SanUmSrisuchinwong(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = y - x
        ydot = -z*np.tanh(x)
        zdot = -self.a + x*y + np.abs(y)
        return (xdot, ydot, zdot)

class DequanLi(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a*(y - x) + self.d*x*z
        ydot = self.k*x + self.f*y - x*z
        zdot = self.c*z + x*y - self.eps*x**2
        return (xdot, ydot, zdot)

class PanXuZhou(DequanLi):
    pass

class Tsucs2(DequanLi):
    pass

class ArnoldWeb(DynSys):
    def rhs(self, X, t):
        p1, p2, x1, x2, z = X
        denom = 4 + np.cos(t) + np.cos(x1) + np.cos(x2)
        p1dot = -self.mu * np.sin(x1) / denom**2
        p2dot = -self.mu * np.sin(x2) / denom**2
        x1dot = p1
        x2dot = p2
        zdot = self.w
        return (p1dot, p2dot, x1dot, x2dot, zdot)
        
class NewtonLiepnik(DynSys):
    def rhs(self, X, t):
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
        
        vdot = self.gi * minf**3 * hinf * (self.vi - v) + self.gkv * n**4 * (self.vk - v) + self.gkc * ca * (self.vk - v) + self.gl * (self.vl - v)
        ndot = (ninf - n) / tau
        cdot = self.rho * (minf**3 * hinf * (self.vc - v) - self.kc * c)
        return vdot, ndot, cdot

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
    @staticmethod
    def _rhs(x, y, th, t, gamma, psi, w):
        xdot = y
        ydot = -1 - np.heaviside(-x, 0)*(x + psi*y*np.abs(y)) + gamma*np.cos(th)
        thdot = w  
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
    @staticjit
    def _rhs(x, y, z, t, a, b, f, g):
        xdot = -y**2 - z**2 - a*x + a*f
        ydot = x*y - b*x*z - y + g
        zdot = b*x*y + x*z - z
        return xdot, ydot, zdot


class ForcedVanDerPol(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        ydot = self.mu *(1 - x**2)*y - x + self.a*np.sin(z)
        xdot = y
        zdot = self.w
        return (xdot, ydot, zdot)

class ForcedFitzHughNagumo(DynSys):
    def rhs(self, X, t):
        v, w, z = X
        vdot = v - v**3/3 - w + self.curr + self.f*np.sin(z)
        wdot = self.gamma*(v + self.a - self.b*w)
        zdot = self.omega
        return (vdot, wdot, zdot)  
    
class HindmarshRose(DynSys):    
    def rhs(self, X, t):
        x, y, z = X
        tx, tz, a, b, d, s, c = self.tx, self.tz, self.a, self.b, self.d, self.s, self.c
        xdot = - tx*x + y - a*x**3 + b*x**2 + z
        ydot = -a*x**3  - (d - b)*x**2 + z
        zdot = -s*x - z + c
        return (xdot/tx, ydot, zdot/tz)

class Colpitts(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        u = z - (self.e - 1)
        fz = -u*(1 - np.heaviside(u, 0))
        xdot = y - self.a*fz
        ydot = self.c - x - self.b*y - z
        zdot = y - self.d*z
        return (xdot, ydot, zdot)   
    
class Laser(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = self.a * (y - x) + self.b * y * z**2
        ydot = self.c * x + self.d * x * z**2
        zdot = self.h * z + self.k * x**2
        return (xdot, ydot, zdot)   
        
#         b, p21, p23, p31, d21, d23, z = X
#         bdot = -self.sigma*b + self.g*(1 + self.f * np.sin(z))*p23
#         p21dot = -p21 - b*p31 + self.a*d21
#         p23dot = -p23  + b*d23- self.a*p31
#         p31dot = -p31 + b*p21 + self.a*p23
#         d21dot = -self.bb*(d21 - self.d210) - 4*self.a*p21 - 2*b*p23
#         d23dot = -self.bb*(d23 - self.d230) - 2*self.a*p21 - 4*b*p23
#         zdot = self.omega
#         return (bdot, p21dot, p23dot, p31dot, d21dot, d23dot, zdot)   


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

class StickSlipOscillator(DynSys):
    def _t(self, v):
        return self.t0*np.sign(v) - self.alpha*v + self.beta*v**3
    def rhs(self, X, t):
        x, v, th = X
        xdot = v
        vdot = self.eps*(self.gamma*np.cos(th) - self._t(v - self.vs) ) + self.a*x - self.b*x**3
        thdot = self.w
        return (xdot, vdot, thdot) 
    
class HastingsPowell(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        a1, b1, d1, a2, b2, d2 = self.a1, self.b1, self.d1, self.a2, self.b2, self.d2
        xdot = x*(1 - x) - y*a1*x/(1 + b1*x)
        ydot = y*a1*x/(1 + b1*x) - z*a2*y/(1 + b2*y) - d1*y
        zdot = z*a2*y/(1 + b2*y) - d2*z
        return (xdot, ydot, zdot)   
    
class CellularNeuralNetwork(DynSys):
    @staticjit
    def f(x):
        return 0.5*(np.abs(x + 1) - np.abs(x - 1))
    def rhs(self, X, t):
        x, y, z = X
        xdot = -x + self.d*self.f(x) - self.b*self.f(y) - self.b*self.f(z)
        ydot = -y - self.b*self.f(x) + self.c*self.f(y) - self.a*self.f(z)
        zdot = -z - self.b*self.f(x) + self.a*self.f(y) + self.f(z)
        return (xdot, ydot, zdot)   

class BeerRNN(DynSys):
    @staticjit
    def _sig(x):
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
## Quasiperiodic systems
##
############################## 

class Torus(DynSys):
    def rhs(self, X, t):
        x, y, z = X
        xdot = (-self.a*self.n*np.sin(self.n*t))*np.cos(t) - (self.r + self.a*np.cos(self.n*t))*np.sin(t)
        ydot = (-self.a*self.n*np.sin(self.n*t))*np.sin(t) + (self.r + self.a*np.cos(self.n*t))*np.cos(t)
        zdot = self.a*self.n*np.cos(self.n*t)
        return (xdot, ydot, zdot)

class CaTwoPlusQuasiperiodic(CaTwoPlus):
    pass

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
    
class Hopfield(DynSys):
    def f(self, x):
        return (1 + np.tanh(x))/2
    def rhs(self, X, t):
        Xdot = -X/self.tau + self.f(self.eps*np.matmul(self.k, X)) - self.beta
        return Xdot
    
class MacArthur(DynSys):
    def growth_rate(self, rr):
        u0 = rr/(self.k.T + rr)
        u = self.r * u0.T
        return np.min(u.T, axis=1)
    def rhs(self, X, t):
        nn, rr = X[:5], X[5:]
        mu = self.growth_rate(rr)
        nndot = nn*(mu - self.m)
        rrdot = self.d*(self.s - rr) - np.matmul(self.c, (mu*nn))
        return np.hstack([nndot, rrdot])

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
        
