from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import utils as util
import time

def print_default_message():
    print("=========================================================="
          + f"\n           Classificador KNN - Biblioteca (Sklearn)"
          + "\n==========================================================\n"
          + "KNN classifier using ready-made libraries like scikit-learn,\n"
            "simplifying the implementation and optimizing time with\n"
            "built-in functions for distance calculation and prediction.\n")

def run():

    print_default_message()

    iris = load_iris()
    attributes, species = load_iris(return_X_y=True)
    attributes_train, attributes_test, species_train, species_test = train_test_split(attributes, species,test_size=0.2, random_state=13)

    total_execution_time = 0
    for k in [1, 3, 5, 7]:
        start_time = time.time()

        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(attributes_train, species_train)
        species_predict = clf.predict(attributes_test)

        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time

        report = classification_report(species_test, species_predict, target_names=iris.target_names)
        report_dictionary = classification_report(species_test, species_predict, target_names=iris.target_names, output_dict=True)
        accuracy = accuracy_score(species_test, species_predict)

        util.print_report(report, k, execution_time, "library")
        util.plot_metrics_table(report_dictionary, accuracy, iris.target_names, k, "library")
        util.plot_confusion_matrix(species_test, species_predict, iris, k)

    print(f"Total execution time (library): {total_execution_time:.6f} seconds")