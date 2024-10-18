from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import utils as util

def print_default_message():
    print("=========================================================="
          + f"\n           Classificador KNN - Biblioteca (Sklearn)"
          + "\n==========================================================\n\n")

def run():

    print_default_message()

    iris = load_iris()
    attributes, species = load_iris(return_X_y=True)
    attributes_train, attributes_test, species_train, species_test = train_test_split(attributes, species,test_size=0.2, random_state=13)

    for k in [1, 3, 5, 7]:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(attributes_train, species_train)
        species_predict = clf.predict(attributes_test)

        report = classification_report(species_test, species_predict, target_names=iris.target_names)
        report_dictionary = classification_report(species_test, species_predict, target_names=iris.target_names, output_dict=True)
        accuracy = accuracy_score(species_test, species_predict)

        util.print_report(report, k)
        util.plot_metrics_table(report_dictionary, accuracy, iris.target_names, k)
        util.plot_confusion_matrix(species_test, species_predict, iris, k)




