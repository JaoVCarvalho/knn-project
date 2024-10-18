from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

def plot_confusion_matrix(species_test, species_predict, iris, k):
    cm = confusion_matrix(species_test, species_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão para k = {k}")
    plt.show()

def plot_metrics_table(report, accuracy, target_names, k):
    metrics = ["precision", "recall", "f1-score"]

    # Criação da tabela
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Linhas da tabela com as métricas extraídas do report
    table_data = []
    for target in target_names:
        row = [report[target][metric] for metric in metrics]  # Pegamos precisão, recall e f1-score
        table_data.append(row)

    # Adicionar acurácia na última linha
    table_data.append([accuracy, accuracy, accuracy])

    # Cabeçalhos da tabela
    header = ["Precisão", "Revocação", "F1-score"]

    # Nome das classes
    classes = list(target_names) + ["Acurácia"]

    # Plotar a tabela
    ax.table(cellText=table_data, colLabels=header, rowLabels=classes, loc="center")
    plt.title(f"Métricas para KNN manual (k = {k})")
    plt.show()

def print_report(report, k):
    print("=========================================================="
          + f"\n           Métricas de Avaliação para K = {k}"
          + "\n==========================================================")
    print("\n" + report + "\n==========================================================\n")
