import numpy as np


def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

def steepest_descent(x0, alpha=0.1, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        x_new = x - alpha * grad
        if np.linalg.norm(grad) < tol:
            break
        x = x_new
    return x

x_min = steepest_descent(np.array([5.0, 5.0]))
print("Minimum nokta:", x_min)
print("Minimum f(x):", f(x_min))
