import numpy as np

# Implementing the Rosenbrock standard test function
def rosenbrock(x, y):
    """
    Computes the Rosenbrock function and its gradient at (x, y).
    Returns:
    loss: scalar f(x, y)
    gx: gradient of f with respect to x
    gy: gradient of f with respect to y
    """
    loss = (1 - x)**2 + 100 * (y - x**2)**2
    gx = -2 * (1 - x) - 400 * x * (y - x**2)
    gy = 200 * (y - x**2)
    return loss, gx, gy

# Sanity checks
if __name__ == "__main__":
    loss, gx, gy = rosenbrock(1.0, 1.0)
    print(f"At minimum (1, 1): loss={loss: .4f}, gx={gx: .4f}, gy={gy: .4f()}")

    loss, gx, gy = rosenbrock(-1.0, 1.0)
    print(f"At point (-1, 1): loss={loss: .4f}, gx={gx: .4f}, gy={gy: .4f()}")

