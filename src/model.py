# imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import pandas as pd
from typing import Callable, Union, Any, Dict, Tuple
from tqdm import tqdm
from scipy.optimize import root, newton
import os, sys
from matplotlib import animation
plt.rcParams.update({
    "font.size": 20,        # default font size
    "axes.titlesize": 20,   # title font size
    "axes.labelsize": 20,   # x and y label font size
    "xtick.labelsize": 20,  # x tick labels
    "ytick.labelsize": 20,  # y tick labels
    "legend.fontsize": 20   # legend text
})
class ShallowWaterModel:
    def __init__(self, 
                    dirname: str = "sol",
                    xmin: float = -5.0,
                    xmax: float = 5.0,
                    T: float = 2.0,
                    g: float = 9.8,
                    Nx: int = 201,
                    Nt: int = 20001,
                    k_start: int = 0,
                    method: str = "FTCS",
                    bc_h: str = "von Neumann",
                    bc_h_vals: NDArray = np.array([0, 0]),
                    bc_hu: str = "dirichlet",
                    bc_hu_vals: NDArray = np.array([0, 0])) -> None:

        self.dirname = dirname
        if not os.path.exists(f"./data/{self.dirname}/"):
            os.makedirs(f"./data/{self.dirname}/")

        self.xmin = xmin
        self.xmax = xmax
        self.T = T
        self.g = g
        self.Nx = Nx
        self.Nt = Nt
        self.k_start = k_start

        self.L = self.xmax - self.xmin
        self.dx = self.L / (self.Nx - 1)
        self.dt = self.T / (self.Nt - 1)

        self.x = np.linspace(self.xmin, self.xmax, self.Nx)
        self.t = np.linspace(0, self.T, self.Nt)

        self.method = method
        self._methods = {
            "FTCS": self.ftcs,
            "LAX_FRIEDRICHS": self.lax_friedrichs,
            "SYMPLECTIC": self.symplectic,
            "BTCS": self.btcs,
            "MUSCL": self.muscl,
            "MUSCL_HYDROSTATIC": self.kurganov_petrova_v2,
        }
        if method.upper() not in self._methods:
            raise ValueError(f"Method {method} not recognized. Available methods: {list(self._methods.keys())}")
        self._method = self._methods[method.upper()]

        self.bc_h = bc_h
        self.bc_h_vals = bc_h_vals
        self.bc_hu = bc_hu
        self.bc_hu_vals = bc_hu_vals

        if self.bc_h not in ["dirichlet", "von Neumann"]:
            raise ValueError(f"Boundary condition type {self.bc_h} for h not recognized. Available types: 'dirichlet', 'von Neumann'")

        if self.bc_hu not in ["dirichlet", "von Neumann"]:
            raise ValueError(f"Boundary condition type {self.bc_hu} for hu not recognized. Available types: 'dirichlet', 'von Neumann'")

    def make_bottom_profile(self, wave:bool = True, override: Union[None, NDArray] = None) -> None:
        match (wave, override):
            case (False, None):
                self.a = np.zeros(self.Nx)-1  # Flat bottom
            case (True, None):
                self.a = np.cos(np.pi/2 * (self.x+5)/5)-2  # Wavy bottom
            case (False, override):
                self.a = override
            case (True, override):
                self.a = override   

    def create_solution_matrix(self) -> None:
        self.h = np.empty((self.Nx, self.Nt))
        self.h_u = np.empty((self.Nx, self.Nt))
    
    def assign_initial_conditions(self, h_override=None, h_u_override=None) -> None:
        if h_override is not None:
            self.h[:,0] = h_override
        else:
            self.h[:,0] = 0 - self.a[:] + 2/5 * np.exp(-self.x**2)  # Initial water height
        if h_u_override is not None:
            self.h_u[:,0] = h_u_override
        else:
            self.h_u[:,0] = 0.0

        self.enforce_bc(self.h[:,0], bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(self.h_u[:,0], bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

    def enforce_bc(self, arr: NDArray, bc_type: str = "dirichlet", bc_vals: NDArray = np.array([0,0])) -> None:
        match bc_type:
            case "dirichlet":
                '''Set the value to bc_vals'''
                arr[0] = bc_vals[0]
                arr[-1] = bc_vals[1]
            case "von Neumann":
                '''Set the first derivative to bc_vals''' # To second order accuracy
                arr[0] = 1/3 * (4*arr[1] - arr[2] - bc_vals[0]*2*self.dx) 
                arr[-1] = 1/3 * (4*arr[-2] - arr[-3] - bc_vals[1]*2*self.dx)

    def ftcs(self, k: int) -> None:
        h_next = self.h[:,k]
        h_u_next = self.h_u[:,k]
        h_curr = self.h[:,k-1]
        h_u_curr = self.h_u[:,k-1]

        # Main
        energy_vec = np.divide(h_u_curr**2, h_curr, out=np.zeros_like(h_u_curr, dtype=float), where=h_curr!=0)  + self.g/2 * h_curr**2
        for i in range(1, self.Nx-1):
            # Using order-1 ftcs scheme (Lax-Friedrichs)

            # partial_t h = -partial_x ( (h + h)* u )
            step_size = (h_u_curr[i+1] - h_u_curr[i-1]) / (2*self.dx)
            h_next[i] = h_curr[i] - self.dt * step_size

            # partial_t (h*u) = - partial_x ( h*u^2 + 0.5*g*h^2 )
            step_size =  (energy_vec[i+1] - energy_vec[i-1])/(2*self.dx) + h_curr[i] * (self.a[i+1] - self.a[i-1])/(2*self.dx) * self.g

            h_u_next[i] = h_u_curr[i] - self.dt * step_size

        self.enforce_bc(h_next, bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(h_u_next, bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

    def lax_friedrichs(self, k: int) -> None:
        h_next = self.h[:,k]
        h_u_next = self.h_u[:,k]
        h_curr = self.h[:,k-1]
        h_u_curr = self.h_u[:,k-1]

        # Main
        self.enforce_bc(h_curr, bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(h_u_curr, bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

        energy_vec = np.divide(h_u_curr**2, h_curr, out=np.zeros_like(h_u_curr, dtype=float), where=h_curr!=0)  + self.g/2 * h_curr**2
        for i in range(1, self.Nx-1):
            # Using order-1 ftcs scheme (Lax-Friedrichs)

            # partial_t h = -partial_x ( (h + h)* u )
            step_size = (h_u_curr[i+1] - h_u_curr[i-1]) / (2*self.dx)
            h_next[i] = (h_curr[i-1] + h_curr[i+1]) / 2 - self.dt * step_size

            # partial_t (h*u) = - partial_x ( h*u^2 + 0.5*g*h^2 )
            step_size =  (energy_vec[i+1] - energy_vec[i-1])/(2*self.dx) + h_curr[i] * (self.a[i+1] - self.a[i-1])/(2*self.dx) * self.g

            h_u_next[i] = (h_u_curr[i-1] + h_u_curr[i+1]) / 2 - self.dt * step_size

        self.enforce_bc(h_next, bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(h_u_next, bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

    def symplectic(self, k: int) -> None:
            h_next = self.h[:,k]
            h_u_next = self.h_u[:,k]
            h_curr = self.h[:,k-1]
            h_u_curr = self.h_u[:,k-1]
            
            energy_vec = np.divide(h_u_curr**2, h_curr, out=np.zeros_like(h_u_curr, dtype=float), where=h_curr!=0)  + self.g/2 * h_curr**2
            h_u_next[1:-1] = h_u_curr[1:-1] - self.dt * ( (energy_vec[2:] - energy_vec[:-2])/(2*self.dx) + h_curr[1:-1] * (self.a[2:] - self.a[:-2])/(2*self.dx) * self.g )
            self.enforce_bc(h_u_next, bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)
            h_next[1:-1] = h_curr[1:-1] - self.dt * ( (h_u_next[2:] - h_u_next[:-2]) / (2*self.dx) )

            self.enforce_bc(h_next, bc_type=self.bc_h, bc_vals=self.bc_h_vals)

    def btcs(self, k: int) -> None:
        h_curr = self.h[:, k-1]
        h_u_curr = self.h_u[:, k-1]
        y_curr = np.concatenate((h_curr, h_u_curr))

        def f(y_next):
            # Initialize residuals to zero to avoid returning uninitialised values
            y_change = np.zeros_like(y_next, dtype=float)
            h_next = y_next[:self.Nx]
            h_u_next = y_next[self.Nx:]

            # energy computed from the unknown next-level variables (fully implicit)
            energy_vec = np.divide(h_u_next**2, h_next, out=np.zeros_like(h_u_next, dtype=float), where=h_next!=0)  + self.g/2 * h_next**2

            # interior continuity residuals: use h_u_next (unknown)
            for i in range(1, self.Nx-1):
                y_change[i] = -((h_u_next[i+1] - h_u_next[i-1]) / (2*self.dx))

            # interior momentum residuals: use energy_vec (from next) and h_next for bathymetry term
            for i in range(1, self.Nx-1):
                y_change[self.Nx + i] = - ((energy_vec[i+1] - energy_vec[i-1])/(2*self.dx) + h_next[i] * (self.a[i+1] - self.a[i-1])/(2*self.dx) * self.g)

            # enforce boundary residuals consistent with BC types so solver pins them
            if self.bc_h == "dirichlet":
                y_change[0] = h_next[0] - self.bc_h_vals[0]
                y_change[self.Nx-1] = h_next[-1] - self.bc_h_vals[1]
            else:  # von Neumann: enforce discrete derivative condition used in enforce_bc
                y_change[0] = h_next[0] - (1/3 * (4*h_next[1] - h_next[2] - self.bc_h_vals[0]*2*self.dx))
                y_change[self.Nx-1] = h_next[-1] - (1/3 * (4*h_next[-2] - h_next[-3] - self.bc_h_vals[1]*2*self.dx))

            if self.bc_hu == "dirichlet":
                y_change[self.Nx + 0] = h_u_next[0] - self.bc_hu_vals[0]
                y_change[self.Nx + self.Nx - 1] = h_u_next[-1] - self.bc_hu_vals[1]
            else:
                y_change[self.Nx + 0] = h_u_next[0] - (1/3 * (4*h_u_next[1] - h_u_next[2] - self.bc_hu_vals[0]*2*self.dx))
                y_change[self.Nx + self.Nx - 1] = h_u_next[-1] - (1/3 * (4*h_u_next[-2] - h_u_next[-3] - self.bc_hu_vals[1]*2*self.dx))

            return y_change

        sol = newton(lambda y_next: y_next - y_curr - self.dt * f(y_next), x0=y_curr, maxiter=1000)

        self.h[:, k] = sol[:self.Nx]
        self.h_u[:, k] = sol[self.Nx:]


        self.enforce_bc(self.h[:, k], bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(self.h_u[:, k], bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

    def muscl(self, k: int) -> None:
        """ MUSCL scheme without proper source term handling
        """
        eps = 1e-12

        def flux(U: NDArray) -> NDArray:
            h = U[0]
            hu = U[1]
            return np.array([hu, np.divide(hu * hu, h, out=np.zeros_like(hu, dtype=float), where=h != 0) + 0.5 * self.g * h ** 2])

        h_curr = self.h[:, k - 1].copy()
        hu_curr = self.h_u[:, k - 1].copy()

        Nx = self.Nx
        dx = self.dx

        # padded arrays with two ghost cells on each side
        U_pad = np.zeros((2, Nx + 4), dtype=float)
        U_pad[0, 2 : 2 + Nx] = h_curr
        U_pad[1, 2 : 2 + Nx] = hu_curr

        # set ghost cells equal to the second-order boundary values (two ghosts on each side)
        U_pad[0, 0] = h_curr[0]
        U_pad[0, 1] = h_curr[0]
        U_pad[1, 0] = hu_curr[0]
        U_pad[1, 1] = hu_curr[0]

        U_pad[0, -2] = h_curr[-1]
        U_pad[0, -1] = h_curr[-1]
        U_pad[1, -2] = hu_curr[-1]
        U_pad[1, -1] = hu_curr[-1]

        # Build padded bottom
        a_pad = np.zeros(Nx + 4, dtype=float)
        a_pad[2 : 2 + Nx] = self.a
        a_pad[0] = self.a[0]
        a_pad[1] = self.a[0]
        a_pad[-2] = self.a[-1]
        a_pad[-1] = self.a[-1]

        # Compute slopes for w = h + a and hu
        w_pad = U_pad[0, :] + a_pad
        hu_pad = U_pad[1, :]

        # minmod limiter: sigma = minmod(delta_minus, delta_plus)
        def minmod(a: NDArray, b: NDArray) -> NDArray:
            prod = a * b
            return np.where(prod <= 0, 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))

        s_w = np.zeros_like(w_pad)
        s_hu = np.zeros_like(hu_pad)
        for p in range(1, w_pad.size - 1):
            dm = w_pad[p] - w_pad[p - 1]
            dp = w_pad[p + 1] - w_pad[p]
            sigma = minmod(dm, dp)
            s_w[p] = 0.5 * sigma

            dm_hu = hu_pad[p] - hu_pad[p - 1]
            dp_hu = hu_pad[p + 1] - hu_pad[p]
            sigma_hu = minmod(dm_hu, dp_hu)
            s_hu[p] = 0.5 * sigma_hu

        # interface loop: compute reconstructed states, fluxes, and hydrostatic h*
        F_star = np.zeros((2, Nx + 1), dtype=float)
        h_if = np.zeros(Nx + 1, dtype=float)
        a_if = np.zeros(Nx + 1, dtype=float)

        for i in range(Nx + 1):
            pL = i + 1
            pR = i + 2
            # reconstruct w and hu at each side
            wL = w_pad[pL] + s_w[pL]
            wR = w_pad[pR] - s_w[pR]
            huL = hu_pad[pL] + s_hu[pL]
            huR = hu_pad[pR] - s_hu[pR]

            aL = a_pad[pL]
            aR = a_pad[pR]

            # hydrostatic reconstruction: compute non-negative left/right depths
            hL_star = max(0.0, wL - max(aL, aR))
            hR_star = max(0.0, wR - max(aL, aR))

            # compute velocities from reconstructed hu where possible
            uL = huL / (max(hL_star, eps)) if hL_star > eps else 0.0
            uR = huR / (max(hR_star, eps)) if hR_star > eps else 0.0

            # reconstructed conserved variables for flux
            U_L_star = np.array([hL_star, hL_star * uL])
            U_R_star = np.array([hR_star, hR_star * uR])

            # fluxes based on hydrostatic reconstructed states
            F_L = flux(U_L_star)
            F_R = flux(U_R_star)

            # wave speed estimate
            a_wave = max(abs(uL) + np.sqrt(self.g * hL_star), abs(uR) + np.sqrt(self.g * hR_star))

            F_star[:, i] = 0.5 * (F_L + F_R) - 0.5 * a_wave * (U_R_star - U_L_star)

            # interface averaged depths and bottoms for source term
            h_if[i] = 0.5 * (hL_star + hR_star)
            a_if[i] = 0.5 * (aL + aR)

        # cell update loop
        U_curr = np.vstack((h_curr, hu_curr))
        U_next = np.zeros((2, Nx), dtype=float)
        for i in range(Nx):
            U_next[:, i] = U_curr[:, i] - (self.dt / dx) * (F_star[:, i + 1] - F_star[:, i])

            # source term using hydrostatic reconstructed interfaces
            hbar = 0.5 * (h_if[i] + h_if[i + 1])
            delta_a = a_if[i + 1] - a_if[i]
            S_i = -self.g * hbar * delta_a
            U_next[1, i] += self.dt / self.dx * S_i

        # write back
        h_next = self.h[:, k]
        h_u_next = self.h_u[:, k]
        h_next[:] = U_next[0, :]
        h_u_next[:] = U_next[1, :]

        # enforce boundary conditions on physical arrays after update
        self.enforce_bc(h_next, bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(h_u_next, bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

    def kurganov_petrova(self, k: int) -> None:
        h_next = self.h[:,k]
        h_u_next = self.h_u[:,k]
        h_curr = self.h[:,k-1]
        h_u_curr = self.h_u[:,k-1]

        pad_h_curr = np.zeros(self.Nx + 4)
        pad_h_curr[2:-2] = h_curr
        pad_h_curr[0] = h_curr[0]
        pad_h_curr[1] = h_curr[0]
        pad_h_curr[-2] = h_curr[-1]
        pad_h_curr[-1] = h_curr[-1]

        pad_h_u_curr = np.zeros(self.Nx + 4)
        pad_h_u_curr[2:-2] = h_u_curr
        pad_h_u_curr[0] = h_u_curr[0]
        pad_h_u_curr[1] = h_u_curr[0]
        pad_h_u_curr[-2] = h_u_curr[-1]
        pad_h_u_curr[-1] = h_u_curr[-1]

        pad_a = np.zeros(self.Nx + 4)
        pad_a[2:-2] = self.a
        pad_a[0] = self.a[0]
        pad_a[1] = self.a[0]
        pad_a[-2] = self.a[-1]
        pad_a[-1] = self.a[-1]

        def F(U: NDArray, a: float) -> NDArray:
            w = U[0]
            hu = U[1]
            return np.array([hu, hu**2 / (w-a) + 0.5 * self.g * (w-a)**2])

        for i in range(0, self.Nx):
            pad_idx = i + 2



            U_plus_two = np.array([pad_h_curr[pad_idx+2]+pad_a[pad_idx+2], pad_h_u_curr[pad_idx+2]])
            U_plus_one = np.array([pad_h_curr[pad_idx+1]+pad_a[pad_idx+1], pad_h_u_curr[pad_idx+1]])
            U_curr = np.array([pad_h_curr[pad_idx]+pad_a[pad_idx], pad_h_u_curr[pad_idx]]) # Here U = [w, hu]
            U_minus_one = np.array([pad_h_curr[pad_idx-1]+pad_a[pad_idx-1], pad_h_u_curr[pad_idx-1]])
            U_minus_two = np.array([pad_h_curr[pad_idx-2]+pad_a[pad_idx-2], pad_h_u_curr[pad_idx-2]])

            theta = 1
            def minmod(*args: NDArray) -> NDArray:
                """Minmod limiter function"""
                a = args[0]
                for b in args[1:]:
                    prod = a * b
                    a = np.where(prod <= 0, 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))
                return a

            Udiff_curr = minmod(theta * (U_curr - U_minus_one), theta*0.5 * (U_plus_one - U_minus_one), theta * (U_plus_one - U_curr))
            Udiff_plus_one = minmod(theta * (U_plus_one - U_curr), theta*0.5 * (U_plus_two - U_curr), theta * (U_plus_two - U_plus_one))
            Udiff_minus_one = minmod(theta * (U_minus_one - U_minus_two), theta*0.5 * (U_curr - U_minus_two), theta * (U_curr - U_minus_one))

            U_plus_plus =  U_plus_one - 1 /2 * Udiff_plus_one
            U_plus_minus = U_curr + 1 /2 * Udiff_curr

            U_minus_plus = U_curr - 1 /2 * Udiff_curr
            U_minus_minus = U_minus_one + 1 /2 * Udiff_minus_one

            depth_plus = pad_a[pad_idx] + 1/2 * minmod( theta * (pad_a[pad_idx+1] - pad_a[pad_idx]), theta*0.5 * (pad_a[pad_idx+1] - pad_a[pad_idx-1]), theta * (pad_a[pad_idx] - pad_a[pad_idx-1]) )
            depth_minus = pad_a[pad_idx] - 1/2 * minmod( theta * (pad_a[pad_idx] - pad_a[pad_idx-1]), theta*0.5 * (pad_a[pad_idx+1] - pad_a[pad_idx-1]), theta * (pad_a[pad_idx+1] - pad_a[pad_idx]) )
            
            F_plus_plus = F(U_plus_plus, depth_plus)
            F_plus_minus = F(U_plus_minus, depth_plus)
            F_minus_plus = F(U_minus_plus, depth_minus)
            F_minus_minus = F(U_minus_minus, depth_minus)

            Sbar = np.array([0, 
                             -self.g  * ( (depth_plus - depth_minus) / self.dx ) * ((U_plus_minus[0] - depth_plus) + (U_minus_plus[0] - depth_minus)) / 2       
                             ])


            try:
                if U_plus_minus[0] - depth_plus < 0:
                    U_plus_minus[0] = depth_plus
                    U_minus_plus[0] = 2*U_curr[0] - depth_plus
                
                if U_minus_plus[0] - depth_minus < 0:
                    U_plus_minus[0] = 2*U_curr[0] - depth_minus
                    U_minus_plus[0] = depth_minus
            except Exception as e:
                print(U_minus_plus)
                raise e


            h_plus_plus = U_plus_plus[0] - depth_plus
            h_plus_minus = U_plus_minus[0] - depth_plus
            h_minus_plus = U_minus_plus[0] - depth_minus
            h_minus_minus = U_minus_minus[0] - depth_minus

            u_plus_plus = np.sqrt(2) * (h_plus_plus)*U_plus_plus[1] / np.sqrt(h_plus_plus**4 + max(1e-12, h_plus_plus**4))
            u_plus_minus = np.sqrt(2) * (h_plus_minus)*U_plus_minus[1] / np.sqrt(h_plus_minus**4 + max(1e-12, h_plus_minus**4))
            u_minus_plus = np.sqrt(2) * (h_minus_plus)*U_minus_plus[1] / np.sqrt(h_minus_plus**4 + max(1e-12, h_minus_plus**4))
            u_minus_minus = np.sqrt(2) * (h_minus_minus)*U_minus_minus[1] / np.sqrt(h_minus_minus**4 + max(1e-12, h_minus_minus**4))

            a_plus_plus = max( 
                u_plus_plus + np.sqrt(self.g * h_plus_plus), 
                u_plus_minus + np.sqrt(self.g * h_plus_minus), 1e-12)
            a_plus_minus = max( 
                u_plus_plus - np.sqrt(self.g * h_plus_plus), 
                u_plus_minus - np.sqrt(self.g * h_plus_minus), 1e-12)

            a_minus_plus = max( 
                u_minus_plus + np.sqrt(self.g * h_minus_plus), 
                u_minus_minus + np.sqrt(self.g * h_minus_minus), 1e-12)
            a_minus_minus = max( 
                u_minus_plus - np.sqrt(self.g * h_minus_plus), 
                u_minus_minus - np.sqrt(self.g * h_minus_minus), 1e-12)

            H_plus = (a_plus_plus*F_plus_minus + a_plus_minus*F_plus_plus) / (a_plus_plus + a_plus_minus) + (a_plus_plus*a_plus_minus) / (a_plus_plus + a_plus_minus) * (U_plus_plus - U_plus_minus)
            H_minus = (a_minus_plus*F_minus_minus + a_minus_minus*F_minus_plus) / (a_minus_plus + a_minus_minus) + (a_minus_plus*a_minus_minus) / (a_minus_plus + a_minus_minus) * (U_minus_plus - U_minus_minus)
            
            dUdt = -(H_plus - H_minus)/2 / self.dx + Sbar
            h_next[i] = h_curr[i] + self.dt * dUdt[0]
            h_u_next[i] = h_u_curr[i] + self.dt * dUdt[1]

        self.enforce_bc(h_next, bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(h_u_next, bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

    def kurganov_petrova_v2(self, k: int) -> None:
        h_next = self.h[:,k]
        h_u_next = self.h_u[:,k]
        h_curr = self.h[:,k-1]
        h_u_curr = self.h_u[:,k-1]

        eps = 1e-12

        pad_h_curr = np.zeros(self.Nx + 4)
        pad_h_curr[2:-2] = h_curr
        pad_h_curr[0] = h_curr[0]
        pad_h_curr[1] = h_curr[0]
        pad_h_curr[-2] = h_curr[-1]
        pad_h_curr[-1] = h_curr[-1]

        pad_h_u_curr = np.zeros(self.Nx + 4)
        pad_h_u_curr[2:-2] = h_u_curr
        pad_h_u_curr[0] = 0 #h_u_curr[0]
        pad_h_u_curr[1] = 0 #h_u_curr[0]
        pad_h_u_curr[-2] = 0 #h_u_curr[-1]
        pad_h_u_curr[-1] = 0 #h_u_curr[-1]

        pad_a = np.zeros(self.Nx + 4)
        pad_a[2:-2] = self.a
        pad_a[0] = self.a[0]
        pad_a[1] = self.a[0]
        pad_a[-2] = self.a[-1]
        pad_a[-1] = self.a[-1]

        def F(U: NDArray, a: float) -> NDArray:
            # U = [w, hu], with h = w - a
            w = U[0]
            hu = U[1]
            h = max(w - a, 0.0)
            u = hu / max(h, eps) if h > eps else 0.0
            return np.array([hu, hu * u + 0.5 * self.g * h * h], dtype=float)

        for i in range(0, self.Nx):
            pad_idx = i + 2

            U_plus_two = np.array([pad_h_curr[pad_idx+2]+pad_a[pad_idx+2], pad_h_u_curr[pad_idx+2]])
            U_plus_one = np.array([pad_h_curr[pad_idx+1]+pad_a[pad_idx+1], pad_h_u_curr[pad_idx+1]])
            U_curr = np.array([pad_h_curr[pad_idx]+pad_a[pad_idx], pad_h_u_curr[pad_idx]]) # Here U = [w, hu]
            U_minus_one = np.array([pad_h_curr[pad_idx-1]+pad_a[pad_idx-1], pad_h_u_curr[pad_idx-1]])
            U_minus_two = np.array([pad_h_curr[pad_idx-2]+pad_a[pad_idx-2], pad_h_u_curr[pad_idx-2]])

            theta = 1
            def minmod(*args: NDArray) -> NDArray:
                """Minmod limiter function"""
                a = args[0]
                for b in args[1:]:
                    prod = a * b
                    a = np.where(prod <= 0, 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)))
                return a

            Udiff_curr = minmod(theta * (U_curr - U_minus_one), theta*0.5 * (U_plus_one - U_minus_one), theta * (U_plus_one - U_curr))
            Udiff_plus_one = minmod(theta * (U_plus_one - U_curr), theta*0.5 * (U_plus_two - U_curr), theta * (U_plus_two - U_plus_one))
            Udiff_minus_one = minmod(theta * (U_minus_one - U_minus_two), theta*0.5 * (U_curr - U_minus_two), theta * (U_curr - U_minus_one))

            U_plus_plus =  U_plus_one - 0.5 * Udiff_plus_one
            U_plus_minus = U_curr + 0.5 * Udiff_curr

            U_minus_plus = U_curr - 0.5 * Udiff_curr
            U_minus_minus = U_minus_one + 0.5 * Udiff_minus_one

            depth_plus = pad_a[pad_idx] + 0.5 * minmod( theta * (pad_a[pad_idx+1] - pad_a[pad_idx]), theta*0.5 * (pad_a[pad_idx+1] - pad_a[pad_idx-1]), theta * (pad_a[pad_idx] - pad_a[pad_idx-1]) )
            depth_minus = pad_a[pad_idx] - 0.5 * minmod( theta * (pad_a[pad_idx] - pad_a[pad_idx-1]), theta*0.5 * (pad_a[pad_idx+1] - pad_a[pad_idx-1]), theta * (pad_a[pad_idx+1] - pad_a[pad_idx]) )

            F_plus_plus = F(U_plus_plus, depth_plus)
            F_plus_minus = F(U_plus_minus, depth_plus)
            F_minus_plus = F(U_minus_plus, depth_minus)
            F_minus_minus = F(U_minus_minus, depth_minus)

            Sbar = np.array([0.0,
                             -self.g  * ( (depth_plus - depth_minus) / self.dx ) * ((U_plus_minus[0] - depth_plus) + (U_minus_plus[0] - depth_minus)) / 2.0
                             ], dtype=float)

            # Enforce non-negative reconstructed water heights
            try:
                if U_plus_minus[0] - depth_plus < 0:
                    U_plus_minus[0] = depth_plus
                    U_minus_plus[0] = 2.0*U_curr[0] - depth_plus

                if U_minus_plus[0] - depth_minus < 0:
                    U_plus_minus[0] = 2.0*U_curr[0] - depth_minus
                    U_minus_plus[0] = depth_minus
            except Exception as e:
                print("Reconstruction error:", e)
                raise e

            # compute h and u for interfaces (for KT speed estimates)
            h_pp = max(U_plus_plus[0] - depth_plus, 0.0)
            h_pm = max(U_plus_minus[0] - depth_plus, 0.0)
            h_mp = max(U_minus_plus[0] - depth_minus, 0.0)
            h_mm = max(U_minus_minus[0] - depth_minus, 0.0)

            u_pp = U_plus_plus[1] / max(h_pp, eps) if h_pp > eps else 0.0
            u_pm = U_plus_minus[1] / max(h_pm, eps) if h_pm > eps else 0.0
            u_mp = U_minus_plus[1] / max(h_mp, eps) if h_mp > eps else 0.0
            u_mm = U_minus_minus[1] / max(h_mm, eps) if h_mm > eps else 0.0

            # Interface i+1/2 (H_plus): left = U_plus_minus, right = U_plus_plus
            cL = np.sqrt(self.g * h_pm) if h_pm > eps else 0.0
            cR = np.sqrt(self.g * h_pp) if h_pp > eps else 0.0
            lambdaL_minus = u_pm - cL
            lambdaL_plus = u_pm + cL
            lambdaR_minus = u_pp - cR
            lambdaR_plus = u_pp + cR

            a_plus_iface = max(0.0, lambdaL_plus, lambdaR_plus)
            a_minus_iface = min(0.0, lambdaL_minus, lambdaR_minus)

            denom_plus = a_plus_iface - a_minus_iface
            if denom_plus < eps:
                H_plus = 0.5 * (F_plus_minus + F_plus_plus)
            else:
                H_plus = (a_plus_iface * F_plus_minus - a_minus_iface * F_plus_plus + a_plus_iface * a_minus_iface * (U_plus_plus - U_plus_minus)) / denom_plus

            # Interface i-1/2 (H_minus): left = U_minus_minus, right = U_minus_plus
            cL = np.sqrt(self.g * h_mm) if h_mm > eps else 0.0
            cR = np.sqrt(self.g * h_mp) if h_mp > eps else 0.0
            lambdaL_minus = u_mm - cL
            lambdaL_plus = u_mm + cL
            lambdaR_minus = u_mp - cR
            lambdaR_plus = u_mp + cR

            a_plus_iface = max(0.0, lambdaL_plus, lambdaR_plus)
            a_minus_iface = min(0.0, lambdaL_minus, lambdaR_minus)

            denom_minus = a_plus_iface - a_minus_iface
            if denom_minus < eps:
                H_minus = 0.5 * (F_minus_minus + F_minus_plus)
            else:
                H_minus = (a_plus_iface * F_minus_minus - a_minus_iface * F_minus_plus + a_plus_iface * a_minus_iface * (U_minus_plus - U_minus_minus)) / denom_minus

            # central-upwind update (no extra 1/2 scaling)
            dUdt = -(H_plus - H_minus) / self.dx + Sbar
            h_next[i] = h_curr[i] + self.dt * dUdt[0]
            h_u_next[i] = h_u_curr[i] + self.dt * dUdt[1]

        # enforce BCs after update
        self.enforce_bc(h_next, bc_type=self.bc_h, bc_vals=self.bc_h_vals)
        self.enforce_bc(h_u_next, bc_type=self.bc_hu, bc_vals=self.bc_hu_vals)

    def run(self) -> None:
         for k in tqdm(range(1, self.Nt)):
            self._method(k)
 
    def plot_initial_conditions(self, show=False) -> None:
            fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(6,5), tight_layout=True)
            line0, = ax0.plot(self.x, self.h[:,0]+self.a[:], label="Water Surface")
            ax0.plot(self.x, self.a[:], label="Bottom")
            ax0.set_title(f"Water height at t=0.00s")
            ax0.set_xlabel("x (m)")
            ax0.set_ylabel("Height (m)")
            ax0.legend()
            line1, = ax1.plot(self.x, self.h_u[:,0], label="Momentum h*u")
            ax1.set_title(f"Momentum at t=0.00s")
            ax1.set_xlabel("x (m)")
            ax1.set_ylabel("Momentum (m^2/s)")
            try:
                ax1.set_ylim(np.min(self.h_u)*1.1, np.max(self.h_u)*1.1)
            except ValueError:
                pass
            ax1.legend()
            plt.savefig(f"./data/{self.dirname}/initial_conditions.png")
            if show:
                plt.show()
            plt.close()

    def height3d(self, show=False) -> None:
        X, T = np.meshgrid(self.x, self.t, indexing='ij')
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, T, self.h + self.a[:, np.newaxis], cmap="viridis")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('h(x,t) + a(x)')
        ax.set_title('Shallow Water Equation Solution')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(f"./data/{self.dirname}/height_3d.png", dpi=300)
        if show:
            plt.show()
        plt.close()

    def velocity3d(self, show=False) -> None:
        X, T = np.meshgrid(self.x, self.t, indexing='ij')
        u = np.divide(self.h_u, self.h, out=np.zeros_like(self.h_u, dtype=float), where=self.h!=0)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, T, u, cmap="viridis")
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Velocity Field')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.savefig(f"./data/{self.dirname}/velocity_3d.png", dpi=300)
        if show:
            plt.show()
        plt.close()

    def heatmap_height(self, show=False) -> None:
        fig, ax = plt.subplots(figsize=(6,5), tight_layout=True, dpi=300)
        c = ax.pcolormesh(self.t, self.x, self.h + self.a[:, np.newaxis], shading='auto', cmap='viridis')
        ax.set_title("Water Height h(x,t) + a(x)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x (m)")
        fig.colorbar(c, ax=ax, label="Height (m)")
        fig.savefig(f"./data/{self.dirname}/height_heatmap.png", dpi=300)
        if show:
            plt.show()
        plt.close()
    
    def heatmap_velocity(self, show=False) -> None:
        u = np.divide(self.h_u, self.h, out=np.zeros_like(self.h_u, dtype=float), where=self.h!=0)
        fig, ax = plt.subplots(figsize=(6,5), tight_layout=True, dpi=300)
        c = ax.pcolormesh(self.t, self.x, u, shading='auto', cmap='viridis')
        ax.set_title("Velocity u(x,t)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x (m)")
        fig.colorbar(c, ax=ax, label="Velocity (m/s)")
        fig.savefig(f"./data/{self.dirname}/velocity_heatmap.png", dpi=300)
        if show:
            plt.show()
        plt.close()

    def animate(self, show=False) -> None:
        # Animation
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(6,5), tight_layout=True)
        line0, = ax0.plot(self.x, self.h[:,0]+self.a[:], label="Water Surface")
        ax0.plot(self.x, self.a[:], label="Bottom")
        ax0.set_title(f"Water height at t=0.00s")
        ax0.set_xlabel("x (m)")
        ax0.set_ylabel("Height (m)")
        ax0.legend()
        line1, = ax1.plot(self.x, self.h_u[:,0], label="Momentum h*u")
        ax1.set_title(f"Momentum at t=0.00s")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("Momentum (m^2/s)")
        try:
            ax0.set_ylim(np.min(self.a[:])*1.1, np.max(self.h)*1.1+np.max(self.a[:])*1.1)
            ax1.set_ylim(np.min(self.h_u)*1.1, np.max(self.h_u)*1.1)
        except ValueError:
            pass
        ax1.legend()
        def animate(i):
            line0.set_ydata(self.h[:,i]+self.a[:])
            ax0.set_title(f"Water height at t={self.t[i]:.2f}s")
            line1.set_ydata(self.h_u[:,i])
            ax1.set_title(f"Momentum at t={self.t[i]:.2f}s")
            return line0, line1
        anim = animation.FuncAnimation(fig, animate, frames=np.linspace(0, self.Nt-1, 200, dtype=int), interval=50)
        FFwriter = animation.FFMpegWriter(fps=20)
        # anim.save(f"./data/{self.dirname}/animation.mp4", writer=FFwriter)    
        anim.save(f"./data/{self.dirname}/animation.gif", writer='pillow')
        if show:
            plt.show()
        plt.close()

    def get_masses(self) -> None:
        masses = np.zeros_like(self.t)
        for k in range(self.Nt):
            masses[k] = np.sum(self.h[:,k]) * self.dx
        self.masses = masses

    def get_energies(self) -> None:
        u = np.divide(self.h_u, self.h, out=np.zeros_like(self.h_u, dtype=float), where=self.h!=0)

        energies = np.sum(u**2 * self.h /2 + self.g * self.h**2 /2 + self.g * self.a[:, np.newaxis] * self.h, axis=0) * self.dx
        self.energies = energies

    def get_courant_numbers(self) -> None:
        u = np.divide(self.h_u, self.h, out=np.zeros_like(self.h_u, dtype=float), where=self.h!=0)
        courant_numbers = (np.abs(u) + np.sqrt(self.g * np.abs(self.h)))* self.dt / self.dx
        self.courant_numbers = courant_numbers

    def save(self, last_only=True) -> None:
        """Outputs .txt file describing the Model. Save h and hu, and a to .csv with pandas"""
        if last_only==False:
            pd.DataFrame(self.h, index=self.x, columns=self.t).to_csv(f"./data/{self.dirname}/h.csv")
            pd.DataFrame(self.h_u, index=self.x, columns=self.t).to_csv(f"./data/{self.dirname}/hu.csv")
        else:
            pd.DataFrame(self.h[:, -1], index=self.x).to_csv(f"./data/{self.dirname}/h.csv")
            pd.DataFrame(self.h_u[:, -1], index=self.x).to_csv(f"./data/{self.dirname}/hu.csv")
        pd.DataFrame(self.a, index=self.x).to_csv(f"./data/{self.dirname}/a.csv")

        with open(f"./data/{self.dirname}/model_desc.txt", "w") as f:
            f.write(
f"""
Shallow Water Model Description
------------------------------
Spatial Domain: x in [{self.xmin}, {self.xmax}] m
Temporal Domain: t in [0, {self.T}] s
Number of Spatial Points: {self.Nx}
Number of Temporal Points: {self.Nt}
Method: {self.method}
Gravity: {self.g} m/s^2
Boundary Conditions for h: {self.bc_h} with values {self.bc_h_vals}
Boundary Conditions for hu: {self.bc_hu} with values {self.bc_hu_vals}
"""
                    )

    def plot_masses(self, show=False) -> None:
        fig, ax = plt.subplots(figsize=(6,5), tight_layout=True, dpi=300)
        ax.plot(self.t, self.masses)
        ax.set_title("Mass over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass (m^2)")
        ax.grid()
        ax.set_xlim(0, self.T)
        # ax.set_ylim(np.minimum(np.min(self.masses), 0), np.max(self.masses)*1.1)
        fig.savefig(f"./data/{self.dirname}/mass_over_time.png", )
        if show:
            fig.show()
        plt.close()

    def plot_energies(self, show=False) -> None:
        fig, ax = plt.subplots(figsize=(6,5), tight_layout=True, dpi=300)
        ax.plot(self.t, self.energies)
        ax.set_title("Energy over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (m^3 / s^2)")
        ax.grid()
        ax.set_xlim(0, self.T)
        # ax.set_ylim(np.minimum(np.min(self.energies), 0), np.max(self.energies)*1.1)
        fig.savefig(f"./data/{self.dirname}/energy_over_time.png", )
        if show:
            fig.show()
        plt.close()

    def plot_max_courant(self, show=False) -> None:
        fig, ax = plt.subplots(figsize=(6,5), tight_layout=True, dpi=300)
        max_courant = np.max(self.courant_numbers, axis=0)
        ax.plot(self.t, max_courant)
        ax.set_title("Maximum Courant Number over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Max Courant Number")
        ax.grid()
        ax.set_xlim(0, self.T)
        fig.savefig(f"./data/{self.dirname}/max_courant_over_time.png", )
        if show:
            fig.show()
        plt.close()

if __name__ == "__main__":
    model = ShallowWaterModel(dirname="kurganov_petrova", method="muscl_hydrostatic", Nt=10001, T=3, Nx=501)
    model.make_bottom_profile()
    model.create_solution_matrix()
    model.assign_initial_conditions()
    model.run()
    model.get_energies()
    model.get_masses()
    model.get_courant_numbers()
    model.save()
    model.plot_masses()
    model.plot_energies()
    model.plot_max_courant()
    model.height3d()
    model.velocity3d()
    model.heatmap_height()
    model.heatmap_velocity()
    model.animate(show=True)