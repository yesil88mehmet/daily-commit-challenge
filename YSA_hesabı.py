import numpy as np

# Aktivasyon fonksiyonları
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Kullanıcıdan parametre alalım
input_vector = np.array([0.5, 0.8])  # Girdi verisi (örnek)

# Gizli katman ağırlıkları (2 giriş -> 3 nöron)
weights_input_hidden = np.array([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6]
])

# Gizli katman bias
bias_hidden = np.array([0.1, 0.1, 0.1])

# Çıkış katmanı ağırlıkları (3 gizli -> 1 çıkış)
weights_hidden_output = np.array([[0.7, 0.8, 0.9]])

# Çıkış bias
bias_output = np.array([0.2])

# Feedforward işlemi
hidden_input = np.dot(weights_input_hidden, input_vector) + bias_hidden
hidden_output = sigmoid(hidden_input)  # veya relu(hidden_input)

final_input = np.dot(weights_hidden_output, hidden_output) + bias_output
final_output = sigmoid(final_input)  # veya başka bir aktivasyon

print(f"Çıktı: {final_output[0]:.4f}")
