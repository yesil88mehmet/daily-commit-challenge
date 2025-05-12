# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:04:39 2024

@author: muozi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüleri oku
img1 = cv2.imread("Crop3.jpg",0)
img2 = cv2.imread("j2.png")
img3 = cv2.imread("j3.png")

# Gradient işlemi için kernel
kernel = np.ones((4, 4), np.uint8)

kernel = np.array([[1, 0, 1],
                   [1,0,1],
                   [1, 0, 1]], dtype=np.uint8)

# Iterasyon sayısı
iterations = 1

# Gradient işlemi (erosion + dilation)
gradient1 = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
gradient2 = cv2.morphologyEx(img2, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
gradient3 = cv2.morphologyEx(img3, cv2.MORPH_GRADIENT, kernel, iterations=iterations)

# Görüntüleri ve gradient işlemi sonucunu yan yana göster
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# Orijinal görüntüleri göster
axs[0, 0].imshow(img1)
axs[0, 0].set_title('Orijinal Image 1')
axs[0, 0].axis('off')

axs[1, 0].imshow(img2)
axs[1, 0].set_title('Orijinal Image 2')
axs[1, 0].axis('off')

axs[2, 0].imshow(img3)
axs[2, 0].set_title('Orijinal Image 3')
axs[2, 0].axis('off')

# Gradient işlemi sonuçlarını göster
axs[0, 1].imshow(cv2.cvtColor(gradient1, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Gradient Image 1')
axs[0, 1].axis('off')

axs[1, 1].imshow(cv2.cvtColor(gradient2, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Gradient Image 2')
axs[1, 1].axis('off')

axs[2, 1].imshow(cv2.cvtColor(gradient3, cv2.COLOR_BGR2RGB))
axs[2, 1].set_title('Gradient Image 3')
axs[2, 1].axis('off')

plt.show()

cv2.imshow('ddd',gradient1)
cv2.waitKey(0)
cv2.destroyAllWindows()