import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.optimize import root



def k_solve(b, d, eps):
    """
    Solving for the value of k that satisfies the given equation
    
    
    Parameters:
        b(float): constant value
        d(float): constant value
        eps(float)
    
    Returns:
        float or None: the value of k if a solution is found within the tolerance,
                       otherwise returns None
    """
    # Defining the function for which we are finding the root
    def func(k):
        return np.cosh(d/k) - np.sqrt(1 - k**2) * np.sinh(d/k) - b

    # Defining the derivative of the function
    def func_prime(k):
        return (d*np.sinh(d/k)/k**2) + np.sqrt(1 - k**2)*np.sinh(d/k) + k*np.cosh(d/k)*np.sinh(d/k)/np.sqrt(1 - k**2)

    # Initial guess
    k0 = 0.5

    # Using Newton's method to find the root
    sol = root(func, k0, jac=func_prime, method='hybr')

    # Checking if the solution is within the tolerance
    if abs(func(sol.x)) <= eps:
        return sol.x[0]
    else:
        return None


def radius(s, b, d):
    """
    Calculates the radius r(s) given parameters s, b, and d.
    
    Parameters:
        s (float)
        b (float)
        d (float)
    
    Returns:
        float or None: the value of r(s) if a solution for k is found,
                       otherwise returns None
    """
    # Tolerance
    eps = 1e-7

    # Solve for k
    k = k_solve(b, d, eps)

    # Check if a solution for k was found
    if k is None:
        return None

    # Calculating r(s)
    r_s = np.cosh(s/k) - np.sqrt(1 - k**2) * np.sinh(s/k)

    return r_s


def surface(b, d, elevation=30, figsize=5, alpha=0.05, n=1000):
    '''
    Plots a 3D surface of the soap film.
    '''
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(figsize, figsize))

    # MAKE YOUR DATA HERE:
    s = np.linspace(0, d, n)
    t = np.linspace(0, 2*np.pi, n)
    S, T = np.meshgrid(s, t)

    r_s = radius(S, b, d)
    X = r_s * np.cos(T)
    Y = r_s * np.sin(T)
    Z = S
    # END MAKING YOUR DATA HERE.

    # Ploting the surface
    ax.plot_surface(X, Y, Z,
                    cmap=cm.coolwarm,
                    antialiased=False,
                    alpha=alpha,
                    rstride=1,
                    cstride=n)
    
    # Editing the plot appearance
    ax.view_init(elev=elevation)
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()

    # Ploting the rings
    ax.plot(X[:, 0], Y[:, 0], Z[:, 0], color='black', linewidth=4)
    ax.plot(X[:, -1], Y[:, -1], Z[:, -1], color='black', linewidth=4)
     
    return fig, ax



import numpy as np
from scipy.optimize import root

def critical_d(b, eps):
    """
    Calculates the value for d where the stable soap film disintegrates.
    
    Parameters:
        b(float or np.ndarray)
        eps(float)
    
    Returns:
        float or np.ndarray: the value(s) of d
    """
    # Defining the function for which we are finding the root
    def func(z):
        return np.cosh(z) - b * np.sinh(z) - np.sqrt(np.cosh(z)**2 - b**2)

    # Initial guess
    z0 = 1.0

    # Using Newton's method to find the root
    sol = root(func, z0, method='hybr')

    # Checking if the solution is within the tolerance
    if np.all(np.abs(func(sol.x)) <= eps):
        return sol.x * np.sqrt(np.cosh(sol.x)**2 - b**2)
    else:
        return None
