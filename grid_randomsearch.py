# Fonksiyon tanımı
def f(x1, x2):
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2

# Izgara arama (Grid Search)
def grid_search():
    # Izgara aralıkları
    x1_min, x1_max = -3, 3
    x2_min, x2_max = -3, 3
    step = 0.01  # Izgara aralığı

    # Minimum değer ve nokta
    min_val = float('inf')
    min_point = (None, None)

    # Elle grid taraması
    x1 = x1_min
    while x1 <= x1_max:
        x2 = x2_min
        while x2 <= x2_max:
            val = f(x1, x2)  # Fonksiyon değeri
            if val < min_val:
                min_val = val
                min_point = (x1, x2)
            x2 += step
        x1 += step

    return min_point, min_val

# Rastgele arama (Random Search)
def random_search():
    # Rastgele sayı üretimi için basit bir algoritma yazıyoruz
    def random_uniform(a, b, seed):
        seed = (1103515245 * seed + 12345) % (2**31)
        rand = seed / (2**31)  # [0, 1) aralığında rastgele sayı
        return a + (b - a) * rand, seed

    # Parametreler
    num_samples = 1000
    x1_min, x1_max = -3, 3
    x2_min, x2_max = -3, 3
    seed = 42  # Sabit seed

    # Minimum değer ve nokta
    min_val = float('inf')
    min_point = (None, None)

    # Rastgele arama
    for _ in range(num_samples):
        x1, seed = random_uniform(x1_min, x1_max, seed)
        x2, seed = random_uniform(x2_min, x2_max, seed)
        val = f(x1, x2)  # Fonksiyon değeri
        if val < min_val:
            min_val = val
            min_point = (x1, x2)

    return min_point, min_val

# Ana program
if __name__ == "__main__":
    # Grid Search Sonuçları
    grid_min_point, grid_min_val = grid_search()
    print("Grid Search:")
    print(f"Minimum Nokta: {grid_min_point}, Minimum Değer: {grid_min_val}")

    # Random Search Sonuçları
    random_min_point, random_min_val = random_search()
    print("\nRandom Search:")
    print(f"Minimum Nokta: {random_min_point}, Minimum Değer: {random_min_val}")