import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def lorenz_system():
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def f(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, 40.0, 0.01)

    states = odeint(f, state0, t)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.draw()
    plt.show()


def vectorfield(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1, x2, y2 = w
    m1, m2, k1, k2, L1, L2, b1, b2 = p

    # Create f = (x1',y1',x2',y2'):
    f = [y1,
         (-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1,
         y2,
         (-b2 * y2 - k2 * (x2 - x1 - L2)) / m2]
    return f


def spring_mass_system(masses=(1.0, 1.5), constants=(8.0, 40.0),
                       lengths=(0.5, 1.0), to_plot=False):
    # Parameter values
    # Masses:
    m1 = masses[0]
    m2 = masses[1]
    # Spring constants
    k1 = constants[0]
    k2 = constants[1]
    # Natural lengths
    L1 = lengths[0]
    L2 = lengths[1]
    # Friction coefficients
    # b1 = 0.8
    # b2 = 0.5
    b1 = 0
    b2 = 0

    # Initial conditions
    # x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
    x1 = 0.5
    y1 = 0.0
    x2 = 2.25
    y2 = 0.0

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = 10.0
    numpoints = 250

    # Create the time samples for the output of the ODE solver.
    # I use a large number of points, only because I want to make
    # a plot of the solution that looks nice.
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

    # Pack up the parameters and initial conditions:
    p = [m1, m2, k1, k2, L1, L2, b1, b2]
    w0 = [x1, y1, x2, y2]

    # Call the ODE solver.
    # this results in 4D array: pos1, vel1, pos2, vel2
    wsol = odeint(vectorfield, w0, t, args=(p,),
                  atol=abserr, rtol=relerr)

    if to_plot:
        plt.figure(1, figsize=(6, 4.5))
        plt.xlabel('t')
        lw = 1

        plt.plot(t, wsol[:, 0], 'b', linewidth=lw, label='pos1')
        plt.plot(t, wsol[:, 1], 'g', linewidth=lw, label='vel1')
        plt.plot(t, wsol[:, 2], 'r', linewidth=lw, label='pos2')
        plt.plot(t, wsol[:, 3], 'y', linewidth=lw, label='vel2')
        plt.legend()
        plt.title('Time series for the\nCoupled Spring-Mass System')
        plt.show()

    return wsol


def coupled_pendula():
    """
    https://www.theorphys.science.ru.nl/people/fasolino/sub_java/pendula/doublependul-en.shtml
    """
    pass

# spring_mass_system()
