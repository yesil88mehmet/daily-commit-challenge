# Soru 1 (a): Fonksiyon ve matris çıktısı
def question_1a():
    # Fonksiyon tanımı
    def f(x1, x2):
        return 1.5 * x1**2 + x2**2 - 2 * x1 * x2 + 2 * x1**3 + 0.5 * x1**4

    # Aralık ve adım
    x1_min, x1_max = -4, 2
    x2_min, x2_max = -4, 2
    step = 0.5

    # Elle grid taraması
    x1 = x1_min
    while x1 <= x1_max:
        x2 = x2_min
        while x2 <= x2_max:
            val = f(x1, x2)  # Fonksiyon değeri
            print(f"x1: {x1:.2f}, x2: {x2:.2f}, f(x1, x2): {val:.2f}")
            x2 += step
        x1 += step

# Soru 1 (b) ve Soru 2: Ekstremum noktaları
def question_1b_and_2():
    # Fonksiyon türevlerini elle hesaplıyoruz:
    # df/dx1 = 3x1 - 2x2 + 6x1^2 + 2x1^3
    # df/dx2 = 2x2 - 2x1

    # Kritik noktaları elle çözümleyerek buluyoruz:
    critical_points = [
        (0, 0),  # Örnek kritik nokta
        # Diğer kritik noktalar elle çözülüp buraya eklenebilir
    ]

    for point in critical_points:
        x1, x2 = point
        print(f"Kritik Nokta: x1 = {x1}, x2 = {x2}")

    # İkinci türev testi ile yerel minimum/maksimum belirlenebilir.

# Ana program
if __name__ == "__main__":
    print("Soru 1 (a): Fonksiyon Değerleri (Matris)")
    question_1a()

    print("\nSoru 1 (b) ve Soru 2: Ekstremum Noktalar")
    question_1b_and_2()