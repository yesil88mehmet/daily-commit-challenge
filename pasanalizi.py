# ai.py
import networkx as nx
import matplotlib.pyplot as plt

# Oyuncu listesi (örnek)
oyuncular = ["Ali", "Mehmet", "Ayşe", "Fatma", "Ahmet"]

# Pas verileri (kim kime kaç pas attı)
paslar = [
    ("Ali", "Mehmet"),
    ("Mehmet", "Ayşe"),
    ("Ayşe", "Ali"),
    ("Ali", "Ahmet"),
    ("Ahmet", "Mehmet"),
    ("Fatma", "Ayşe"),
    ("Mehmet", "Fatma"),
    ("Fatma", "Ali"),
    ("Ali", "Mehmet"),
    ("Mehmet", "Ahmet"),
]

# Yönlü grafik oluştur
G = nx.DiGraph()

# Oyuncuları düğüm olarak ekle
G.add_nodes_from(oyuncular)

# Pasları kenar olarak ekle
G.add_edges_from(paslar)

# Düğümlerin pozisyonunu belirle (dairesel yerleşim)
pos = nx.circular_layout(G)

# Grafiği çiz
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Kenarlardaki pas sayısını kalınlaştırmak için ağırlık ekleyelim
edge_labels = {}
for (u, v) in G.edges():
    if (u, v) in edge_labels:
        edge_labels[(u, v)] += 1
    else:
        edge_labels[(u, v)] = 1

# Kenar kalınlığı (pas sayısına göre)
edge_widths = [edge_labels[edge] for edge in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True, arrowstyle='->', arrowsize=15)

# Kenarların üstüne pas sayısını yaz
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Basketbol Pas Trafiği Analizi 🏀", fontsize=15)
plt.axis('off')
plt.tight_layout()
plt.show()
