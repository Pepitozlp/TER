import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

def convert_to_opponent_color_space(image):
    r, g, b = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    rg = r - g
    by = 2 * b - r - g
    wb = r + g + b
    return np.dstack((rg, by, wb))

def compute_histograms(image):
    histRG = cv2.calcHist([image], [0], None, [256], [0, 256])
    histBY = cv2.calcHist([image], [1], None, [256], [0, 256])
    histWB = cv2.calcHist([image], [2], None, [256], [0, 256])
    return histRG, histBY, histWB

def create_compact_representation(hist_list):
    combined_hist = np.concatenate([h.flatten() for h in hist_list])
    combined_hist = combined_hist / np.sum(combined_hist)
    return combined_hist

def calculate_histogram_distance(hist1, hist2):
    eps = 1e-10
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps))

# Chemins
chemin_image = "coil-100/obj30__0.png"
chemin_dossier = "coil-100"

image1 = cv2.imread(chemin_image)
image1g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
histgris1 = cv2.calcHist([image1g], [0], None, [256], [0, 256])
histgris1 = histgris1 / np.sum(histgris1)
img1_opponent = convert_to_opponent_color_space(image1)
histRG1, histBY1, histWB1 = compute_histograms(img1_opponent)
hist1_compact = create_compact_representation([histRG1, histBY1, histWB1])

# Liste des distances
distances_multiples = []

for x in range(1, 101):
    for y in range(0, 356, 5):
        nom_fichier = f"obj{x}__{y}.png"
        chemin_fichier = os.path.join(chemin_dossier, nom_fichier)
        if os.path.exists(chemin_fichier):
            image2 = cv2.imread(chemin_fichier)
            if image2 is None:
                continue
            image2g = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            histgris2 = cv2.calcHist([image2g], [0], None, [256], [0, 256])
            histgris2 = histgris2 / np.sum(histgris2)
            img2_opponent = convert_to_opponent_color_space(image2)
            histRG2, histBY2, histWB2 = compute_histograms(img2_opponent)

            hist2_compact = create_compact_representation([histRG2, histBY2, histWB2])
            dist_opponent = calculate_histogram_distance(hist1_compact, hist2_compact)
            dist_gray = calculate_histogram_distance(histgris1.flatten(), histgris2.flatten())

            distances_multiples.append((chemin_fichier, dist_gray, dist_opponent))

distances_multiples.sort(key=lambda x: x[2])
for i in range(10):
    print(distances_multiples[i])

#on choisi une image de référence (pas celle d'indice 0 car c'est l'image qu'on a choisi arbitrairement
img_ref_name, dist_gray_ref, dist_opp_ref = distances_multiples[3]
# Mélange et sélection d’un échantillon

random.shuffle(distances_multiples)
echantillon = distances_multiples[:15]
if img_ref_name not in [item[0] for item in echantillon]:
    echantillon.append((img_ref_name, dist_gray_ref, dist_opp_ref))

# Extraire les données pour le tracé
noms = [os.path.basename(item[0]) for item in echantillon]
dist_gray = [item[1] for item in echantillon]
dist_opponent = [item[2] for item in echantillon]

# Tracé
plt.figure(figsize=(12, 6))
plt.plot(noms, dist_gray, marker='o', linestyle='-', color='gray', label='Distance (histogramme gris)')
plt.plot(noms, dist_opponent, marker='s', linestyle='--', color='orange', label='Distance (histogramme opponent)')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Nom de l'image")
plt.ylabel("Distance")
plt.title(f"Comparaison des distances (gris vs opponent) à {os.path.basename(chemin_image)}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
