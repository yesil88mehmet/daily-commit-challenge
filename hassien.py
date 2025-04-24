import sympy as sp

# Semboller
x, y = sp.symbols('x y')

# Fonksiyon tanımı
f = x**3 + y**2 - 4*x + 6*y  # örnek fonksiyon

# Gradient (∇f)
grad_f = [sp.diff(f, var) for var in (x, y)]

# Durağan nokta çözümü
stationary_points = sp.solve(grad_f, (x, y))
print("Durağan Noktalar:", stationary_points)

# Hessian Matrisi
H = sp.hessian(f, (x, y))
print("\nHessian Matrisi:")
sp.pprint(H)

# Durağan noktada Hessian'ı değerlendirme
if stationary_points:
    x0, y0 = stationary_points[0]
    H_at_point = H.subs({x: x0, y: y0})
    print("\nDurağan Noktadaki Hessian:")
    sp.pprint(H_at_point)

    # Özdeğerleri bul
    eigenvals = H_at_point.eigenvals()
    print("\nÖzdeğerler:", eigenvals)

    # Noktanın tipi
    values = list(eigenvals.keys())
    if all(ev > 0 for ev in values):
        print("→ Bu bir yerel minimum noktasıdır.")
    elif all(ev < 0 for ev in values):
        print("→ Bu bir yerel maksimum noktasıdır.")
    else:
        print("→ Bu bir saddle noktasıdır.")
else:
    print("Durağan nokta bulunamadı.")
