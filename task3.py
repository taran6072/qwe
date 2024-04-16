import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve


def resting_state(u_init, v_init, eps, gamma, beta, I):
    """
    Function to find the resting state of the cell using Newton's method.
    
    Parameters:
    u_init -initial guess for u
    v_init - initial guess for v
    eps 
    gamma 
    beta 
    I - stimulus
    
    Returns:
    Resting state of the cell (u_star, v_star)
    """
    def equations(vars):
        """
        Function to represent the system of equations for the FitzHugh-Nagumo model.
        
        Parameters:
        vars - list of variables [u, v]
        
        Returns:
        List of equations [du/dt, dv/dt]
        """
        u, v = vars
        eq1 = (1/eps)*(u - u**3/3 - v + I)
        eq2 = eps*(u - gamma*v + beta)
        return [eq1, eq2]
    
    # Using fsolve to find the roots of the equations
    u_star, v_star =  fsolve(equations, (u_init, v_init))
    
    return u_star, v_star

def solve_ODE(u0, v0, nmax, dt, eps, gamma, beta, I):
    """
    Function to solve the ODE for the FitzHugh-Nagumo model using Euler's method.
    
    Parameters:
    u0 - initial value for u
    v0 - initial value for v
    nmax - number of time steps
    dt - time step size
    eps -- 
    gamma 
    beta
    I
    
    Returns:
    Solution of the ODE as a numpy array [u, v]
    """
    # Initialize arrays for u and v
    u = np.zeros(nmax+1)
    v = np.zeros(nmax+1)
    
    # Set initial conditions
    u[0] = u0
    v[0] = v0
    
    # Use Euler's method to solve the ODE
    for n in range(nmax):
        u[n+1] = u[n] + dt*(1/eps)*(u[n] - u[n]**3/3 - v[n] + I)
        v[n+1] = v[n] + dt*eps*(u[n] - gamma*v[n] + beta)
    
    return np.array([u, v])






def plot_solutions(uv, dt):
    """
    Function to plot the solutions of the ODE for the FitzHugh-Nagumo model.
    
    Parameters:
    uv - numpy array with 2 rows and nmax+1 columns containing (u_n, v_n) values
    dt - time step size
    
    Returns:
    fig 
    
    """
    # Create figure and axes objects
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    
    # Time array
    t = np.arange(0, (len(uv[0]) - 1) * dt + dt, dt)
    
    # Plot u and v vs time
    ax[0].plot(t, uv[0], label='$u_n$')
    ax[0].plot(t, uv[1], label='$v_n$')
 
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Value')
    ax[0].legend()
    ax[0].set_title('Time evolution')
    
    # Plot v vs u (phase space)
    ax[1].plot(uv[0], uv[1])
    ax[1].set_xlabel('$u_n$')
    ax[1].set_ylabel('$v_n$')
    ax[1].set_title('Phase space')
    
    return fig, ax