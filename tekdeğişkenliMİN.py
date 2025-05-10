from scipy.optimize import minimize_scalar

def f(x):
    return x**2

result = minimize_scalar(f)
print("Minimum Nokta:", result.x)
print("Minimum Değer:", result.fun)
