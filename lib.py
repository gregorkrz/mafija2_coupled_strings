import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib.animation import FuncAnimation


def rk45( f, x0, t, bc_correct_fn=None ):
    """Fourth-order Runge-Kutta method with error estimate.

    USAGE:
        x, err = rk45(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.
        bc_correct_fn - Function of x that corrects it to satisfy (arbitrarily complicated) boundary conditions.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        err   - NumPy array containing estimate of errors at each step.  If
                a system is being solved, err will be an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Mathematics and Computing" 6th Edition, by Cheney and Kincaid,
        Brooks-Cole, 2008.
    """

    # Coefficients used to compute the independent variable argument of f

    c20  =   2.500000000000000e-01  #  1/4
    c30  =   3.750000000000000e-01  #  3/8
    c40  =   9.230769230769231e-01  #  12/13
    c50  =   1.000000000000000e+00  #  1
    c60  =   5.000000000000000e-01  #  1/2

    # Coefficients used to compute the dependent variable argument of f

    c21 =   2.500000000000000e-01  #  1/4
    c31 =   9.375000000000000e-02  #  3/32
    c32 =   2.812500000000000e-01  #  9/32
    c41 =   8.793809740555303e-01  #  1932/2197
    c42 =  -3.277196176604461e+00  # -7200/2197
    c43 =   3.320892125625853e+00  #  7296/2197
    c51 =   2.032407407407407e+00  #  439/216
    c52 =  -8.000000000000000e+00  # -8
    c53 =   7.173489278752436e+00  #  3680/513
    c54 =  -2.058966861598441e-01  # -845/4104
    c61 =  -2.962962962962963e-01  # -8/27
    c62 =   2.000000000000000e+00  #  2
    c63 =  -1.381676413255361e+00  # -3544/2565
    c64 =   4.529727095516569e-01  #  1859/4104
    c65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute 4th order RK estimate

    a1  =   1.157407407407407e-01  #  25/216
    a2  =   0.000000000000000e-00  #  0
    a3  =   5.489278752436647e-01  #  1408/2565
    a4  =   5.353313840155945e-01  #  2197/4104
    a5  =  -2.000000000000000e-01  # -1/5

    b1  =   1.185185185185185e-01  #  16.0/135.0
    b2  =   0.000000000000000e-00  #  0
    b3  =   5.189863547758284e-01  #  6656.0/12825.0
    b4  =   5.061314903420167e-01  #  28561.0/56430.0
    b5  =  -1.800000000000000e-01  # -9.0/50.0
    b6  =   3.636363636363636e-02  #  2.0/55.0

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    e = numpy.array( [ 0 * x0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] )
        k2 = h * f( x[i] + c21 * k1, t[i] + c20 * h )
        k3 = h * f( x[i] + c31 * k1 + c32 * k2, t[i] + c30 * h )
        k4 = h * f( x[i] + c41 * k1 + c42 * k2 + c43 * k3, t[i] + c40 * h )
        k5 = h * f( x[i] + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4, \
                        t[i] + h )
        k6 = h * f( \
            x[i] + c61 * k1 + c62 * k2 + c63 * k3 + c64 * k4 + c65 * k5, \
            t[i] + c60 * h )

        x[i+1] = x[i] + a1 * k1 + a3 * k3 + a4 * k4 + a5 * k5
        if bc_correct_fn is not None:
            x[i+1] = bc_correct_fn( x[i+1], t[i])
        x5 = x[i] + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6

        e[i+1] = abs( x5 - x[i+1] )

    return x


def domain(n, d=100):
    # domain: [x, left_idx, middle_idx, right_idx]
    x = np.linspace(-d, d, 2*n+1)
    left_idx = np.arange(n)
    right_idx = np.arange(n+1, 2*n+1)
    middle_idx = n
    return [x, left_idx, middle_idx, right_idx]


def initial_gauss(x, idx, premik):
    # idx is either left_idx or right_idx
    y = np.zeros_like(x)
    d = x[-1] - x[0]
    y[idx] = np.exp(-(x[idx]-premik)**2)
    return y


def get_derivative_matrix(n, order=4):
    order_coeffs = [[-2, 1], [-5/2, 4/3, -1/12], [-49/18, 3/2, -3/20, 1/90],
                    [-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
                    [-5269 / 1800, 5 / 3, -5 / 21, 5 / 126, -5 / 1008, 1 / 3150],
                    [-5369 / 1800, 12 / 7, -15 / 56, 10 / 189, -1 / 112, 2 / 1925, -1 / 16632]]
    coeffs = order_coeffs[order-2]
    a = np.zeros((n, n), dtype=np.float64)
    for i, c in enumerate(coeffs):
        if i != 0:
            a += np.diag(np.ones(n-i), k=i) * c
        a += np.diag(np.ones(n-i), k=-i) * c
    return csr_matrix(a)


def dydt(y, domain, A, Qu=0., Qv=0., c_squared=1.):
    x, left_idx, middle_idx, right_idx = domain
    h = abs(x[1] - x[0])
    # A = sparse 2nd derivative matrix
    dydt = np.zeros_like(y)
    dydt[:, 0] = y[:, 1]
    dydt_all = A @ y[:, 0] / (h**2) * c_squared
    dydt_left = dydt_all[left_idx]
    dydt_right = dydt_all[right_idx]
    dydt[left_idx, 1] = dydt_left
    dydt[right_idx, 1] = dydt_right
    dydt[middle_idx, 0] = 0.
    return dydt

def bc_correct(y_upper, y_lower, domain, Qu, Qv):
    # function for correcting the boundary condition, assuming that we have a spring coupling
    x, left_idx, middle_idx, right_idx = domain
    n = len(x) // 2
    d = x[-1] - x[0]
    h = abs(x[1] - x[0])
    ur = y_upper[right_idx[0], 0]
    ul = y_upper[left_idx[-1], 0]
    vr = y_lower[right_idx[0], 0]
    vl = y_lower[left_idx[-1], 0]
    sol = np.linalg.solve(np.array([[1 / Qv + 2 / h, 1 / Qv], [1 / Qu, 1 / Qu + 2 / h]]), np.array([(ur+ul) / h, (vr+vl) / h]))
    return sol


def der(y_upper, y_lower, domain, A, Qu, Qv):
    dydt_upper = dydt(y_upper, domain, A, Qu, Qv)
    dydt_lower = dydt(y_lower, domain, A, Qu, Qv)
    return dydt_upper, dydt_lower

def integrate_rk(y_upper_0, y_lower_0, domain, A, Qu, Qv, T, n_steps, return_history=True, left_condition=None):
    # left_condition: function of t that returns x(t)
    y_upper = np.zeros_like(y_upper_0)
    y_lower = np.zeros_like(y_lower_0)
    y_upper[:, :] = y_upper_0
    y_lower[:, :] = y_lower_0
    middle_idx = domain[2]
    x0 = y_upper_0
    # flatten x0
    x0 = x0.flatten()
    x0 = np.concatenate([x0, y_lower_0.flatten()])
    def correct_bc(x, t):
        # returns corrected x
        y_upper = x[:len(x)//2].reshape(y_upper_0.shape)
        y_lower = x[len(x)//2:].reshape(y_lower_0.shape)
        y_upper[middle_idx, 0], y_lower[middle_idx, 0] = bc_correct(y_upper, y_lower, domain, Qu, Qv)
        if left_condition is not None:
            y_upper[0, 0] = left_condition(t)
        return np.concatenate([y_upper.flatten(), y_lower.flatten()])
    def f(x, t):
        y_upper = x[:len(x)//2]
        y_lower = x[len(x)//2:]
        y_upper = y_upper.reshape(y_upper_0.shape)
        y_lower = y_lower.reshape(y_lower_0.shape)
        dydt_upper, dydt_lower = der(y_upper, y_lower, domain, A, Qu, Qv)
        return np.concatenate([dydt_upper.flatten(), dydt_lower.flatten()])

    ts = np.linspace(0, T, n_steps+1)
    sol = rk45(f, x0, ts, bc_correct_fn=correct_bc)
    y_upper_sol, y_lower_sol = sol[:, :sol.shape[1] // 2], sol[:, sol.shape[1] // 2:]
    if return_history:
        return y_upper_sol.reshape([-1] + list(y_upper_0.shape)), y_lower_sol.reshape([-1] + list(y_lower_0.shape)), ts
    else:
        return y_upper_sol[-1].reshape(y_upper_0.shape), y_lower_sol[-1].reshape(y_upper_0.shape)


def save_animation(x, y_upper, y_lower, t, filename="test.gif", idx=0):
    # Which timeseries to animate (velocity or displacement)
    def animate(x, y, t, y_bottom):
        # Create the figure and axis
        fig, ax = plt.subplots(2, figsize=(9, 6))
        line, = ax[0].plot([], [], lw=2)
        ax[0].grid()
        ax[1].grid()
        ax[1].set_xlabel("x")
        ax[0].set_ylabel("u")
        ax[1].set_ylabel("-v")
        line_bottom = ax[1].plot([], [], lw=2)
        pt_top, = ax[0].plot([], [], 'o', color='green')
        pt_bottom, = ax[1].plot([], [], 'o', color='green')
        mid_pt = int(len(x) / 2)
        # Define the animation function
        def animate(i):
            line_bottom[0].set_data(x, y_bottom[i])
            line.set_data(x, y[i])
            pt_top.set_data([x[mid_pt]], [y[i][mid_pt]])
            pt_bottom.set_data([x[mid_pt]], [y_bottom[i][mid_pt]])
            ax[0].set_xlim(x[0], x[-1])
            ax[0].set_ylim(-1, 1)
            # ax[0].set_xlabel('x')
            # ax[0].set_ylabel('y')
            ax[0].set_title(f't={round(t[i], 3)}')
            ax[0].set_xlim(x[0], x[-1])
            ax[0].set_ylim(-1, 1)
            ax[1].set_xlim(x[0], x[-1])
            ax[1].set_ylim(-1, 1)
            return line, line_bottom

        # Create the animation
        anim = FuncAnimation(fig, animate, frames=len(y), interval=200)
        # Show the animation
        return anim, fig

    real_duration = 3  # secs   # this doesn't really work for some reason
    fps = len(t) / real_duration
    # select only 10% of frames
    selected = y_upper[::30, :, idx]
    selected_down = y_lower[::30, :, idx]
    ts = t[::30]
    # print(selected.shape)
    print("Saving animation...")
    anim, fig = animate(x, selected, ts, -1 * selected_down)
    anim.save(filename, fps=fps)
    print("Done - animation saved to", filename)

def integrate_rk_delta(y_upper_0, y_lower_0, domain, A, q_mu_u, q_mu_v, mu, T, n_steps, return_history=True, left_condition=None, force_fn=None,
                       c_squared_u=1., c_squared_v=1.):
    # This function integrates the system with a delta function at the middle of the domain, so there's no "right" and "left" zone
    # left_condition: function of t that returns x(t)
    middle_idx = domain[2]
    x0 = np.concatenate([y_upper_0.flatten(), y_lower_0.flatten()])
    x, left_idx, middle_idx, right_idx = domain
    h = (x[-1]-x[0]) / (len(x)-1)
    def force(upper_y, lower_y):
        Fu = -1 * q_mu_u * (upper_y + lower_y)
        Fv = -1 * q_mu_v * (upper_y + lower_y)
        return Fu, Fv
    if force_fn is None:
        force_fn = force

    def f(x, t):
        y_upper = x[:len(x) // 2]
        y_lower = x[len(x) // 2:]
        y_upper = y_upper.reshape(y_upper_0.shape)
        y_lower = y_lower.reshape(y_lower_0.shape)
        #print(y_upper.shape, y_lower.shape)
        #Fu = -1 * Qu * (y_upper[middle_idx, 0] + y_lower[middle_idx, 0])
        #Fv = -1 * Qv * (y_upper[middle_idx, 0] + y_lower[middle_idx, 0])
        Fu, Fv = force_fn(y_upper[middle_idx, 0], y_lower[middle_idx, 0])
        dydt_u = np.zeros_like(y_upper_0)
        dydt_u[:, 0] = y_upper[:, 1]
        force_u = np.zeros_like(y_upper[:, 0])
        force_u[middle_idx] = 1 / (h * mu) * Fu
        dydt_u[:, 1] = A @ y_upper[:, 0] / (h ** 2) * c_squared_u + force_u
        dydt_v = np.zeros_like(y_lower_0)
        dydt_v[:, 0] = y_lower[:, 1]
        force_v = np.zeros_like(y_lower[:, 0])
        force_v[middle_idx] = 1 / (h * mu) * Fv
        dydt_v[:, 1] = A @ y_lower[:, 0] / (h ** 2) * c_squared_v + force_v
        r = np.concatenate([dydt_u.flatten(), dydt_v.flatten()])
        return r
    ts = np.linspace(0, T, n_steps+1)
    def bc_corr_delta(x, t):
        y_upper = x[:len(x) // 2].reshape(y_upper_0.shape)
        y_lower = x[len(x) // 2:].reshape(y_lower_0.shape)
        y_upper[0, 0] = left_condition(t)
        return np.concatenate([y_upper.flatten(), y_lower.flatten()])
    if left_condition is not None:
        bc_corr_fn = bc_corr_delta
    else:
        bc_corr_fn = None
    sol = rk45(f, x0, ts, bc_correct_fn=bc_corr_fn)
    y_upper_sol, y_lower_sol = sol[:, :sol.shape[1] // 2], sol[:, sol.shape[1] // 2:]
    if return_history:
        return y_upper_sol.reshape([-1] + list(y_upper_0.shape)), y_lower_sol.reshape([-1] + list(y_lower_0.shape)), ts
    else:
        return y_upper_sol[-1].reshape(y_upper_0.shape), y_lower_sol[-1].reshape(y_upper_0.shape)



def save_animation_two(x, y_upper, y_lower, y_upper_1, y_lower_1, t, filename="test.gif", idx=0):
    # Which timeseries to animate (velocity or displacement)
    def animate(x, y, t, y_bottom, y_1, y_bottom_1):
        # Create the figure and axis
        fig, ax = plt.subplots(2, figsize=(9, 6))
        line, = ax[0].plot([], [], lw=2)
        line_1 = ax[0].plot([], [], "--", lw=2)
        ax[0].grid()
        ax[1].grid()
        line_bottom = ax[1].plot([], [], lw=2)
        line_bottom_1 = ax[1].plot([], [], "--", lw=2)
        pt_top, = ax[0].plot([], [], 'o', color='green')
        pt_bottom, = ax[1].plot([], [], 'o', color='green')
        pt_top_1, = ax[0].plot([], [], 'o', color='red')
        pt_bottom_1, = ax[1].plot([], [], 'o', color='red')
        ax[1].set_xlabel("x")
        ax[0].set_ylabel("u")
        ax[1].set_ylabel("-v")
        mid_pt = int(len(x) / 2)
        # Define the animation function
        def animate(i):
            line_bottom[0].set_data(x, y_bottom[i])
            line.set_data(x, y[i])
            pt_top.set_data([x[mid_pt]], [y[i][mid_pt]])
            pt_bottom.set_data([x[mid_pt]], [y_bottom[i][mid_pt]])
            pt_bottom_1.set_data([x[mid_pt]], [y_bottom_1[i][mid_pt]])
            pt_top_1.set_data([x[mid_pt]], [y_1[i][mid_pt]])
            line_1[0].set_data(x, y_1[i])
            line_bottom_1[0].set_data(x, y_bottom_1[i])
            ax[0].set_xlim(x[0], x[-1])
            ax[0].set_ylim(-1, 1)
            # ax[0].set_xlabel('x')
            # ax[0].set_ylabel('y')
            ax[0].set_title(f't={round(t[i], 3)}')
            ax[0].set_xlim(x[0], x[-1])
            ax[0].set_ylim(-1, 1)
            ax[1].set_xlim(x[0], x[-1])
            ax[1].set_ylim(-1, 1)
            return line, line_bottom

        # Create the animation
        anim = FuncAnimation(fig, animate, frames=len(y), interval=200)
        # Show the animation
        return anim, fig

    real_duration = 3  # secs   # this doesn't really work for some reason
    fps = len(t) / real_duration
    # select only 10% of frames
    selected = y_upper[::30, :, idx]
    selected_down = y_lower[::30, :, idx]
    selected_1 = y_upper_1[::30, :, idx]
    selected_down_1 = y_lower_1[::30, :, idx]
    ts = t[::30]
    # print(selected.shape)
    print("Saving animation...")
    anim, fig = animate(x, selected, ts, -1 * selected_down, selected_1, -1*selected_down_1)
    anim.save(filename, fps=fps)
    print("Done - animation saved to", filename)

