import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def convert_to_opponent_color_space(image):
    """ Convertit une image RGB en espace de couleur opposé (rg, by, wb) """
    r, g, b = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    rg = r - g
    by = 2 * b - r - g
    wb = r + g + b
    return np.dstack((rg, by, wb))

# Chargement des images
image1 = cv2.imread("image5_resized.jpg")
image2 = cv2.imread("image6_resized.jpg")
image3 = cv2.imread("image4_resized.jpg")

# Conversion en gris
image1g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2g = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image3g = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image grise 1", image1g)
cv2.waitKey(0)
cv2.imshow("Image grise 2", image2g)
cv2.waitKey(0)
cv2.imshow("Image grise 3", image3g)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Histogrammes gris
histgris1 = cv2.calcHist([image1g], [0], None, [256], [0, 256])
histgris2 = cv2.calcHist([image2g], [0], None, [256], [0, 256])
histgris3 = cv2.calcHist([image3g], [0], None, [256], [0, 256])

# conv en couleyr opposé
img1_opponent = convert_to_opponent_color_space(image1)
img2_opponent = convert_to_opponent_color_space(image2)
img3_opponent = convert_to_opponent_color_space(image3)

# Récupération histo
hist_bins = 256
hist_range = [0, 256]

def compute_histograms(image):
    histRG = cv2.calcHist([image], [0], None, [hist_bins], hist_range)
    histBY = cv2.calcHist([image], [1], None, [hist_bins], hist_range)
    histWB = cv2.calcHist([image], [2], None, [hist_bins], hist_range)
    return histRG, histBY, histWB

histRG1, histBY1, histWB1 = compute_histograms(img1_opponent)
histRG2, histBY2, histWB2 = compute_histograms(img2_opponent)
histRG3, histBY3, histWB3 = compute_histograms(img3_opponent)

plt.figure(figsize=(12, 6))

# Histogrammes en gris
plt.subplot(2, 3, 1)
plt.plot(histgris1, color='black')
plt.title("Histogramme gris - Image 1")
plt.subplot(2, 3, 2)
plt.plot(histgris2, color='black')
plt.title("Histogramme gris - Image 2")
plt.subplot(2, 3, 3)
plt.plot(histgris3, color='black')
plt.title("Histogramme gris - Image 3")

# Histogrammes en rg-by-wb
plt.subplot(2, 3, 4)
plt.plot(histRG1, label='RG', color='r')
plt.plot(histBY1, label='BY', color='b')
plt.plot(histWB1, label='WB', color='g')
plt.legend()
plt.title("Histogramme couleur - Image 1")

plt.subplot(2, 3, 5)
plt.plot(histRG2, label='RG', color='r')
plt.plot(histBY2, label='BY', color='b')
plt.plot(histWB2, label='WB', color='g')
plt.legend()
plt.title("Histogramme couleur - Image 2")

plt.subplot(2, 3, 6)
plt.plot(histRG3, label='RG', color='r')
plt.plot(histBY3, label='BY', color='b')
plt.plot(histWB3, label='WB', color='g')
plt.legend()
plt.title("Histogramme couleur - Image 3")

plt.show()

def create_compact_representation(hist_list):
    combined_hist = np.concatenate([h.flatten() for h in hist_list])
    combined_hist = combined_hist / np.sum(combined_hist)
    return combined_hist

hist1_compact = create_compact_representation([histRG1, histBY1, histWB1])
hist2_compact = create_compact_representation([histRG2, histBY2, histWB2])
hist3_compact = create_compact_representation([histRG3, histBY3, histWB3])

def calculate_histogram_distance(hist1, hist2):
    eps = 1e-10
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps))


print("\nDistances entre les histogrammes (rg-by-wb):")
dist_1_2 = calculate_histogram_distance(hist1_compact, hist2_compact)
dist_1_3 = calculate_histogram_distance(hist1_compact, hist3_compact)
dist_2_3 = calculate_histogram_distance(hist2_compact, hist3_compact)

print(f"Distance entre Image 1 et Image 2: {dist_1_2:.4f}")
print(f"Distance entre Image 1 et Image 3: {dist_1_3:.4f}")
print(f"Distance entre Image 2 et Image 3: {dist_2_3:.4f}")

feature_vectors = np.vstack([hist1_compact, hist2_compact, hist3_compact])
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(feature_vectors)

print("\nRésultats du clustering:")
print(f"Image 1 appartient au cluster: {clusters[0]}")
print(f"Image 2 appartient au cluster: {clusters[1]}")
print(f"Image 3 appartient au cluster: {clusters[2]}")