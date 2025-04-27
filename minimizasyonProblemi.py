import numpy as np
from scipy.optimize import minimize

# Fonksiyon tanımlanıyor: f(x) = x^2 - 4x + 4
def func(x):
    return x**8

# Başlangıç tahmini
initial_guess = [19]

# Fonksiyonun minimizasyonu
result = minimize(func, initial_guess, method='BFGS')

# Sonuç yazdırılıyor
print(f"Minimizasyon Sonucu: {result.x[0]}")
