import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
import utils as util
import time


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_k_nearest_neighbors(attributes_train, species_train, test_sample, k):
    distances = []

    for i in range(len(attributes_train)):
        distance = euclidean_distance(attributes_train[i], test_sample)
        distances.append((distance, species_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = [distances[i][1] for i in range(k)]

    return neighbors

def predict(attributes_train, species_train, attributes_test, k):

    predictions = []

    for test_sample in attributes_test:

        neighbors = get_k_nearest_neighbors(attributes_train, species_train, test_sample, k)
        most_common = Counter(neighbors).most_common(1)[0][0]
        predictions.append(most_common)

    return predictions

def print_default_message():
    print("=========================================================="
          + f"\n           Classificador KNN - Manual (Hardcore)"
          + "\n==========================================================\n\n")

def run():

    print_default_message()

    iris = load_iris()
    attributes, species = load_iris(return_X_y=True)
    attributes_train, attributes_test, species_train, species_test = train_test_split(attributes, species, test_size=0.2,random_state=13)

    total_execution_time = 0
    for k in [1, 3, 5, 7]:
        start_time = time.time()  # Começa a contagem

        species_predict = predict(attributes_train, species_train, attributes_test, k)

        end_time = time.time()  # Finaliza a contagem
        execution_time = end_time - start_time  # Tempo de execução
        total_execution_time += execution_time

        report = classification_report(species_test, species_predict, target_names=iris.target_names)
        report_dictionary = classification_report(species_test, species_predict, target_names=iris.target_names, output_dict=True)
        accuracy = accuracy_score(species_test, species_predict)

        util.print_report(report, k)
        util.plot_metrics_table(report_dictionary, accuracy, iris.target_names, k, "manual")
        util.plot_confusion_matrix(species_test, species_predict, iris, k)

        print(f"Tempo de execução para k={k} (manual): {execution_time:.6f} segundos\n")

    print(f"Tempo total de execução (manual): {total_execution_time:.6f} segundos\n")  # Exibe o tempo total