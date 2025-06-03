

import numpy as np
import matplotlib.pyplot as plt
# Model: y = a * exp(b * x)

def model(x, params):
    a, b = params
    return a * np.exp(b * x)

# Hata vektörü: e(x) = y_data - model(x, params)
def error_vector(x_data, y_data, params):
    return y_data - model(x_data, params)

# Jacobian matrisi hesapla
def jacobian(x_data, params):
    a, b = params
    J = np.zeros((len(x_data), len(params)))
    exp_bx = np.exp(b * x_data)
    J[:, 0] = -exp_bx               # d/da [a * exp(bx)] = exp(bx)
    J[:, 1] = -a * x_data * exp_bx  # d/db [a * exp(bx)] = a * x * exp(bx)
    return J
def levenberg_marquardt(x_data, y_data, x0, mu0=1e-3, mu_scal=10, mu_min=1e-12, mu_max=1e12,
                         epsilon1=1e-8, epsilon2=1e-8, epsilon3=1e-8, Nmax=100):
    xk = np.array(x0, dtype=float)
    mu_k = mu0
    k = 0

    while True:
        e = error_vector(x_data, y_data, xk)
        J = jacobian(x_data, xk)
        JTJ = J.T @ J
        JTe = J.T @ e

        # Step 3: Solve for zk
        H_lm = JTJ + mu_k * np.identity(len(xk))
        zk = -np.linalg.solve(H_lm, JTe)

        # Evaluate f(x)
        f_xk = np.sum(e ** 2)
        f_xk_zk = np.sum(error_vector(x_data, y_data, xk + zk) ** 2)

        if f_xk_zk < f_xk:
            pk = zk
            sk = 1.0  # line search sabit
            xk_next = xk + sk * pk
            mu_k = mu_k / mu_scal
        else:
            mu_k = mu_k * mu_scal
            if mu_k < mu_max and mu_k > mu_min:
                continue
            else:
                xk_next = xk

        # Adım 5
        k += 1

        # Adım 6: Sonlandırma kriterleri
        delta_f = abs(f_xk_zk - f_xk)
        delta_x = np.linalg.norm(xk_next - xk)
        grad_norm = np.linalg.norm(JTe)

        if (k >= Nmax or
            delta_f < epsilon1 or
            delta_x < epsilon2 or
            grad_norm < epsilon3 or
            not (mu_min < mu_k < mu_max)):
            break

        xk = xk_next

    return xk, k
# Örnek veri (noisy)
x_vals = np.array([-4.0, -3.2, -2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4, 3.2])
y_vals = np.array([-0.3, -0.49, -0.8, -1.31, -2.13, -3.47, -5.66, -9.23, -15.04, -24.53])



# Başlangıç tahmini
initial_guess = [1.0, -1.0]

# Algoritmayı çalıştır
params_estimated, iterations = levenberg_marquardt(x_vals, y_vals, initial_guess)

print("Tahmin edilen parametreler:", params_estimated)
print("İterasyon sayısı:", iterations)

# Sonuç grafiği
plt.scatter(x_vals, y_vals, label='Veri')
plt.plot(x_vals, model(x_vals, params_estimated), color='red', label='Model')
plt.legend()
plt.title("Levenberg-Marquardt Sonucu")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
