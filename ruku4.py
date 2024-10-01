def ruku4_step(f, t, x, u, h):
    """
    Perform one step of the fourth-order Runge-Kutta method.

    Parameters:
    - f: A function representing the system of ODEs. It should take two arguments: x (state) and t (time).
    - x: Current state.
    - t: Current time.
    - h: Step size.

    Returns:
    - x_new: New state after one step.
    """
    k1 = h * f(t, x, u)
    k2 = h * f(t + 0.5 * h, x + 0.5 * k1, u)
    k3 = h * f(t + 0.5 * h, x + 0.5 * k2, u)
    k4 = h * f(t + h, x + k3, u)
    x_new = x + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x_new