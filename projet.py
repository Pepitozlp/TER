import os
import random
import cv2
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import os

def convert_to_opponent_color_space(image):
    """Convertit une image RGB vers l'espace couleur Opponent"""
    if image is None or len(image.shape) != 3:
        raise ValueError("Image invalide pour la conversion Opponent")

    r = image[:, :, 2].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 0].astype(np.float32)

    rg = r - g
    by = 2 * b - r - g
    wb = r + g + b

    return np.dstack((rg, by, wb))


def compute_opponent_histograms(image):
    """Calcule les histogrammes pour chaque canal de l'espace Opponent"""
    histRG = cv2.calcHist([image], [0], None, [256], [-255, 255])
    histBY = cv2.calcHist([image], [1], None, [256], [-510, 510])
    histWB = cv2.calcHist([image], [2], None, [256], [0, 765])
    return histRG, histBY, histWB


def compute_rgb_histograms(image):
    """Calcule les histogrammes pour chaque canal RGB"""
    histR = cv2.calcHist([image], [2], None, [256], [0, 256])
    histG = cv2.calcHist([image], [1], None, [256], [0, 256])
    histB = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histR, histG, histB


def compute_yuv_histograms(image):
    """Calcule les histogrammes pour chaque canal YUV"""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    histY = cv2.calcHist([img_yuv], [0], None, [256], [0, 256])
    histU = cv2.calcHist([img_yuv], [1], None, [256], [0, 256])
    histV = cv2.calcHist([img_yuv], [2], None, [256], [0, 256])
    return histY, histU, histV


def create_compact_representation(hist_list):
    """Crée une représentation compacte normalisée"""
    combined_hist = np.concatenate([h.flatten() for h in hist_list])
    combined_hist = combined_hist / (np.sum(combined_hist) + 1e-10)
    return combined_hist


def chi_squared_distance(hist1, hist2):
    """Distance du Chi-carré"""
    eps = 1e-10
    return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps))


def euclidean_distance(hist1, hist2):
    """Distance euclidienne"""
    return np.sqrt(np.sum((hist1 - hist2) ** 2))


def manhattan_distance(hist1, hist2):
    """Distance de Manhattan"""
    return np.sum(np.abs(hist1 - hist2))


def correlation_distance(hist1, hist2):
    """Distance de corrélation"""
    return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)


def get_image_class(filename):
    """Extrait la classe de l'image (ex: 'obj25' pour 'obj25__0.png')"""
    return filename.split('__')[0]


def extract_angle(filename):
    """Extrait l'angle du nom de fichier (ex: 0 pour 'obj25__0.png')"""
    try:
        return int(filename.split('__')[-1].split('.')[0])
    except:
        return None


def calculate_precision_at_k(query_class, retrieved_files, k):
    """Calcule la précision@k"""
    relevant = sum(1 for i in range(min(k, len(retrieved_files)))
                   if get_image_class(os.path.basename(retrieved_files[i][0])) == query_class)
    return relevant / k


def calculate_recall(query_class, retrieved_files, total_relevant, k):
    """Calcule le rappel@k"""
    relevant_retrieved = sum(1 for i in range(min(k, len(retrieved_files)))
                             if get_image_class(os.path.basename(retrieved_files[i][0])) == query_class)
    return relevant_retrieved / total_relevant if total_relevant > 0 else 0


def calculate_f1_score(precision, recall):
    """Calcule le F1-score"""
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def evaluate_system(dataset_path, color_space='opponent', distance_metric='chi_squared',
                    num_queries=10, k=5, angles=None, debug=False):
    """
    Évalue le système avec gestion robuste des erreurs

    Args:
        dataset_path: Chemin vers le dossier d'images
        color_space: 'opponent', 'rgb' ou 'yuv'
        distance_metric: 'chi_squared', 'euclidean', 'manhattan' ou 'correlation'
        num_queries: Nombre de requêtes à tester
        k: Nombre de résultats à considérer
        angles: Liste d'angles à utiliser (None pour tous)
        debug: Afficher des informations de débogage
    """
    if not os.path.exists(dataset_path):
        print(f"Erreur: Le dossier {dataset_path} n'existe pas")
        return 0, 0, 0

    try:
        all_files = [f for f in os.listdir(dataset_path)
                     if f.lower().endswith('.png') and os.path.isfile(os.path.join(dataset_path, f))]
        if not all_files:
            print("Aucune image PNG trouvée dans le dossier")
            return 0, 0, 0
    except Exception as e:
        print(f"Erreur lors de la lecture du dossier: {e}")
        return 0, 0, 0

    if angles is not None:
        if isinstance(angles, int):
            angles = [angles]
        valid_files = [f for f in all_files if extract_angle(f) in angles]
    else:
        valid_files = all_files.copy()

    try:
        queries = random.sample(valid_files, min(num_queries, len(valid_files)))
    except ValueError:
        print(f"Pas assez d'images disponibles ({len(valid_files)} trouvées, {num_queries} demandées)")
        return 0, 0, 0

    color_space_funcs = {
        'opponent': (convert_to_opponent_color_space, compute_opponent_histograms),
        'rgb': (lambda x: x, compute_rgb_histograms),
        'yuv': (lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2YUV), compute_yuv_histograms)
    }

    distance_funcs = {
        'chi_squared': chi_squared_distance,
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'correlation': correlation_distance
    }

    if color_space not in color_space_funcs:
        print(f"Espace colorimétrique inconnu: {color_space}")
        return 0, 0, 0

    if distance_metric not in distance_funcs:
        print(f"Métrique de distance inconnue: {distance_metric}")
        return 0, 0, 0

    convert_func, hist_func = color_space_funcs[color_space]
    distance_func = distance_funcs[distance_metric]

    class_counts = defaultdict(int)
    for f in all_files:
        class_counts[get_image_class(f)] += 1

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    processed_queries = 0

    for query_file in queries:
        #pour chaque requête fichier on regarde son chemin et son image
        try:
            query_path = os.path.join(dataset_path, query_file)
            query_img = cv2.imread(query_path)
            if query_img is None:
                if debug:
                    print(f"Impossible de lire l'image: {query_file}")
                continue

            #on récupère sa classe et le nombre d'image de même classe -1 (on ne compte pas l'image que analyse)
            query_class = get_image_class(query_file)
            total_relevant = class_counts.get(query_class, 1) - 1

            #on calcule les histogrammes des 3 représentations
            query_converted = convert_func(query_img)
            query_hists = hist_func(query_converted)
            query_hist = create_compact_representation(query_hists)

            #on calcule les 4 distances par rapport aux images cibles
            distances = []
            for target_file in all_files:
                if target_file == query_file:
                    continue

                target_path = os.path.join(dataset_path, target_file)
                target_img = cv2.imread(target_path)
                if target_img is None:
                    continue

                target_converted = convert_func(target_img)
                target_hists = hist_func(target_converted)
                target_hist = create_compact_representation(target_hists)

                dist = distance_func(query_hist, target_hist)
                distances.append((target_path, dist))

            #on les tries puis on calcule précision, rappel et F1-score
            reverse_sort = (distance_metric == 'correlation')
            distances.sort(key=lambda x: x[1], reverse=reverse_sort)

            precision = calculate_precision_at_k(query_class, distances, k)
            recall = calculate_recall(query_class, distances, total_relevant, k)
            f1 = calculate_f1_score(precision, recall)

            #on calcule la précision, le rappel et le F1-score total des requêtes
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            processed_queries += 1

            if debug:
                print(f"Requête: {query_file} | Classe: {query_class} | Precision@{k}: {precision:.2f} | F1: {f1:.2f}")

        except Exception as e:
            if debug:
                print(f"Erreur lors du traitement de {query_file}: {str(e)}")
            continue

    if processed_queries == 0:
        print("Aucune requête n'a pu être traitée")
        return 0, 0, 0

    #on calcule la precision, rappel et F1-score moyen
    avg_precision = total_precision / processed_queries
    avg_recall = total_recall / processed_queries
    avg_f1 = total_f1 / processed_queries

    """pour tester les résultats un par un"""
    # print("\n=== Résultats finaux ===")
    # print(f"Espace colorimétrique: {color_space}")
    # print(f"Métrique de distance: {distance_metric}")
    # print(f"Requêtes traitées: {processed_queries}/{len(queries)}")
    # print(f"Precision@{k}: {avg_precision:.4f}")
    # print(f"Recall@{k}: {avg_recall:.4f}")
    # print(f"F1-score@{k}: {avg_f1:.4f}")

    return avg_precision, avg_recall, avg_f1, query_path, distances

def plot_top_results(query_path, retrieved_files, run_id, color_space, metric, output_dir="results"):
    """Affiche et sauvegarde les 10 premières images similaires à la requête"""
    os.makedirs(output_dir, exist_ok=True)

    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"Erreur de lecture image requête : {query_path}")
        return

    fig, axes = plt.subplots(1, 11, figsize=(22, 4))
    axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Query")
    axes[0].axis("off")

    for i, (path, _) in enumerate(retrieved_files[:10]):
        img = cv2.imread(path)
        if img is not None:
            axes[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f"{i + 1}")
            axes[i + 1].axis("off")

    plt.tight_layout()
    filename = f"{output_dir}/run{run_id}_{color_space}_{metric}.png"
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    DATASET_PATH = "C:/Users/Pepit/Desktop/FAC/M1/TER/coil-100"
    TEST_ANGLES = [0, 90, 180, 270]

    if not os.path.exists(DATASET_PATH):
        print(f"\nERREUR: Dossier '{DATASET_PATH}' introuvable")
    else:
        color_spaces = ['opponent', 'rgb', 'yuv']
        metrics = ['chi_squared', 'euclidean', 'manhattan', 'correlation']

        for run in range(1):
            print(f"\n=========== EXÉCUTION {run} ===========\n")
            results = {}
            top_images = {}  # pour stocker les fichiers récupérés

            for space in color_spaces:
                for metric in metrics:
                    print(f"\nConfiguration: {space.upper()} + {metric.upper()}")

                    # On active le mode debug partiellement pour avoir les fichiers récupérés
                    precision, recall, f1, query_path, distances= evaluate_system(
                        dataset_path=DATASET_PATH,
                        color_space=space,
                        distance_metric=metric,
                        num_queries=1,  # une seule requête pour le graphe
                        k=10,
                        angles=0,
                        debug=True
                    )
                    results[f"{space}_{metric}"] = (precision, recall, f1)

                    # Recharger les fichiers pour graphe
                    # On récupère manuellement les distances ici
                    all_files = [f for f in os.listdir(DATASET_PATH)
                                 if f.lower().endswith('.png') and os.path.isfile(os.path.join(DATASET_PATH, f))]
                    query_img = cv2.imread(query_path)

                    convert_func, hist_func = {
                        'opponent': (convert_to_opponent_color_space, compute_opponent_histograms),
                        'rgb': (lambda x: x, compute_rgb_histograms),
                        'yuv': (lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2YUV), compute_yuv_histograms)
                    }[space]

                    distance_func = {
                        'chi_squared': chi_squared_distance,
                        'euclidean': euclidean_distance,
                        'manhattan': manhattan_distance,
                        'correlation': correlation_distance
                    }[metric]

                    if query_img is None:
                        continue
                    query_converted = convert_func(query_img)
                    query_hists = hist_func(query_converted)
                    query_hist = create_compact_representation(query_hists)

                    top_images[f"{space}_{metric}"] = (query_path, distances)

            #on affiche la meilleure configuration lors du test
            if results:
                best_config = max(results.items(), key=lambda x: x[1][2])
                best_key = best_config[0]
                best_precision, best_recall, best_f1 = best_config[1]

                print("\n=== Meilleure configuration ===")
                print(f"Paramètres: {best_key}")
                print(f"Precision@10: {best_precision:.4f}")
                print(f"Recall@10: {best_recall:.4f}")
                print(f"F1-score@10: {best_f1:.4f}")

                query_path, retrieved = top_images[best_key]
                parts = best_key.split("_")
                best_space = parts[0]
                best_metric = "_".join(parts[1:])

                plot_top_results(query_path, retrieved, run, best_space, best_metric)

                NUM_QUERIES = 10
                K = 5

                color_spaces = ['opponent', 'rgb', 'yuv']
                distance_metrics = ['chi_squared', 'euclidean', 'manhattan', 'correlation']
                results = {}

                if not os.path.exists(DATASET_PATH):
                    print(f"\nERREUR: Dossier '{DATASET_PATH}' introuvable")
                else:
                    os.makedirs("robustesse_graphs", exist_ok=True)

                    print("\n== ÉVALUATION ROBUSTESSE PAR ANGLE, MÉTRIQUE ET REPRÉSENTATION ==")
                    for cs in color_spaces:
                        results[cs] = {}
                        for metric in metrics:
                            print(f"\n=== {cs.upper()} + {metric.upper()} ===")
                            f1_per_angle = []
                            for angle in TEST_ANGLES:
                                print(f"  > Angle {angle}°")
                                precision, recall, f1, query_path, distances = evaluate_system(
                                    dataset_path=DATASET_PATH,
                                    color_space=cs,
                                    distance_metric=metric,
                                    num_queries=NUM_QUERIES,
                                    k=K,
                                    angles=[angle],
                                    debug=False
                                )
                                f1_per_angle.append((angle, f1))
                            results[cs][metric] = f1_per_angle

                    # === Graphe par représentation ===
                    for cs in color_spaces:
                        plt.figure(figsize=(10, 6))
                        for metric in distance_metrics:
                            data = results[cs][metric]
                            angles = [a for a, _ in data]
                            f1_scores = [f1 for _, f1 in data]
                            plt.plot(angles, f1_scores, marker='o', label=metric.upper())

                        #on affiche dans un même plot les calculs de la  F-mesure en fonction des angles
                        plt.title(f"Robustesse de {cs.upper()} selon l'angle")
                        plt.xlabel("Angle (°)")
                        plt.ylabel(f"F1-score@{K}")
                        plt.xticks(TEST_ANGLES)
                        plt.grid(True)
                        plt.legend()
                        plt.tight_layout()
                        #on sauvegarde le graphe
                        plt.savefig(f"robustesse_graphs/{cs}_robustesse.png")
                        plt.close()