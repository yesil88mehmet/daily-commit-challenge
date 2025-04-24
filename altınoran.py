import numpy as np

def f(x):
    return (x - 2)**2 + 1

def golden_section_search(a, b, tol=1e-5):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2

xmin = golden_section_search(0, 5)
print("Minimum x değeri:", xmin)
print("Minimum f(x):", f(xmin))
