import cv2 #opencv itself
import numpy as np # matrix manipulations
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#on lit les images
image1 = cv2.imread("image5_resized.jpg")
image2 = cv2.imread("image6_resized.jpg")
image3 = cv2.imread("image4_resized.jpg")


#on les mets en niveau de gris
image1g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2g = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image3g = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

cv2.imshow("image grise", image1g)
cv2.waitKey(0)
cv2.imshow("image grise", image2g)
cv2.waitKey(0)
cv2.imshow("image grise", image3g)
cv2.waitKey(0)

# récuperer les histogrammes des images en niveau de gris

histgris1 = cv2.calcHist(image1g, [0], None, [256], [0,256])
histgris2 = cv2.calcHist(image2g, [0], None, [256], [0,256])
histgris3 = cv2.calcHist(image3g, [0], None, [256], [0,256])

#on affiche les histogrammes en niveau de gris (bleu = image 1, orange = image 2, vert = image 3)
plt.plot(histgris1)
plt.plot(histgris2)
plt.plot(histgris3)
plt.show()


#images dans l'espace YUV
img1_to_yuv = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
img2_to_yuv = cv2.cvtColor(image2, cv2.COLOR_BGR2YUV)
img3_to_yuv = cv2.cvtColor(image3, cv2.COLOR_BGR2YUV)

# récuperer les histogrammes couleurs des images

histColorB1 = cv2.calcHist(img1_to_yuv, [0], None, [256], [0,256])
histColorB2 = cv2.calcHist(img2_to_yuv, [0], None, [256], [0,256])
histColorB3 = cv2.calcHist(img3_to_yuv, [0], None, [256], [0,256])

histColorG1 = cv2.calcHist(img1_to_yuv, [1], None, [256], [0,256])
histColorG2 = cv2.calcHist(img2_to_yuv, [1], None, [256], [0,256])
histColorG3 = cv2.calcHist(img3_to_yuv, [1], None, [256], [0,256])

histColorR1 = cv2.calcHist(img1_to_yuv, [2], None, [256], [0,256])
histColorR2 = cv2.calcHist(img2_to_yuv, [2], None, [256], [0,256])
histColorR3 = cv2.calcHist(img3_to_yuv, [2], None, [256], [0,256])


#on affiche les histogrammes couleur (bleu = image 1, orange = image 2, vert = image 3)

plt.plot(histColorB1)
plt.plot(histColorB2)
plt.plot(histColorB3)
plt.show()

plt.plot(histColorG1)
plt.plot(histColorG2)
plt.plot(histColorG3)
plt.show()

plt.plot(histColorR1)
plt.plot(histColorR2)
plt.plot(histColorR3)
plt.show()

# crée une représentation compact des histogrammes couleurs 
def create_compact_representation(yuv_hist):
    # Assemble YUV pour faire un vecteur
    # Potentiellement utiliser PCA ici pour réduire dimensionnalité 
    combined_hist = np.concatenate([yuv_hist[0].flatten(), 
                                  yuv_hist[1].flatten(), 
                                  yuv_hist[2].flatten()])
    
    # Noramlsiation 
    combined_hist = combined_hist / np.sum(combined_hist)
    
    return combined_hist

# Calcul distance entre histo
def calculate_histogram_distance(hist1, hist2, method='chi-squared'):
    if method == 'chi-squared':
        eps = 1e-10
        return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps))
    elif method == 'intersection':
        # intersection
        return np.sum(np.minimum(hist1, hist2))
    else:
        # dist eucli
        return np.sqrt(np.sum((hist1 - hist2) ** 2))

# regroupement Kmeans
def cluster_histograms(feature_vectors, k=2):
    from sklearn.cluster import KMeans
    
    # crée le model kmeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(feature_vectors)
    
    return clusters

# representation chaque img
hist1_compact = create_compact_representation([histColorB1, histColorG1, histColorR1])
hist2_compact = create_compact_representation([histColorB2, histColorG2, histColorR2])
hist3_compact = create_compact_representation([histColorB3, histColorG3, histColorR3])


# calcul distance entre histo
print("\nDistances between histograms (Chi-squared distance):")
dist_1_2 = calculate_histogram_distance(hist1_compact, hist2_compact)
dist_1_3 = calculate_histogram_distance(hist1_compact, hist3_compact)
dist_2_3 = calculate_histogram_distance(hist2_compact, hist3_compact)

print(f"Distance between Image 1 and Image 2: {dist_1_2:.4f}")
print(f"Distance between Image 1 and Image 3: {dist_1_3:.4f}")
print(f"Distance between Image 2 and Image 3: {dist_2_3:.4f}")

# regroupement
feature_vectors = np.vstack([hist1_compact, hist2_compact, hist3_compact])
clusters = cluster_histograms(feature_vectors, k=2)

print("\nClustering results:")
print(f"Image 1 belongs to cluster: {clusters[0]}")
print(f"Image 2 belongs to cluster: {clusters[1]}")
print(f"Image 3 belongs to cluster: {clusters[2]}")

