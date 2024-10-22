from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(species_test, species_predict, iris, k):
    cm = confusion_matrix(species_test, species_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

    fig, ax = plt.subplots(figsize=(8, 6))

    disp.plot(cmap=plt.cm.Oranges, ax=ax)

    ax.set_title(f"Confusion Matrix for k = {k}", fontsize=16, pad=20)

    ax.set_xlabel('Predicted label', fontsize=12, labelpad=15)
    ax.set_ylabel('True label', fontsize=12, labelpad=15)

    for labels in ax.texts:
        labels.set_fontsize(14)

    plt.tight_layout()
    plt.show()

def plot_metrics_table(report, accuracy, target_names, k, title):
    metrics = ["precision", "recall", "f1-score", "support"]  # Incluí "f1-score" novamente

    # Criação da tabela
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Função para formatar os valores para duas casas decimais e arredondar para cima
    def format_value(val, is_support=False):
        if is_support:
            return int(val)  # Remove as casas decimais para o support e para o total na última linha
        else:
            return np.ceil(val * 100) / 100  # Arredonda para cima com 2 casas decimais

    # Linhas da tabela com as métricas extraídas do report
    table_data = []
    total_support = 0
    for target in target_names:
        row = [format_value(report[target][metric], metric == "support") for metric in metrics]
        total_support += report[target]["support"]  # Somar os valores de support
        table_data.append(row)

    # Adicionar acurácia na última linha (support como total)
    table_data.append([format_value(accuracy), format_value(accuracy), format_value(accuracy), int(total_support)])

    # Cabeçalhos da tabela (em inglês)
    header = ["Precision", "Recall", "F1-score", "Support"]  # Incluí "F1-score" novamente

    # Nome das classes
    classes = list(target_names) + ["Accuracy/Total"]

    # Plotar a tabela com melhorias visuais
    table = ax.table(cellText=table_data, colLabels=header, rowLabels=classes, loc="center", cellLoc='center', rowLoc='center')

    # Ajustar estilo da tabela
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  # Aumenta a altura das células

    # Aumentar a largura da coluna de rótulos (onde estão os nomes das espécies)
    table.auto_set_column_width(col=list(range(len(header))))
    for (i, j), cell in table.get_celld().items():
        if j == -1:  # Coluna de rótulos (títulos das linhas)
            cell.set_width(0.2)  # Aumentar a largura dessa coluna

    plt.title(f"Metrics for {title} KNN (k = {k})", fontsize=14)
    plt.show()

def print_report(report, k):
    print("=========================================================="
          + f"\n           Métricas de Avaliação para K = {k}"
          + "\n==========================================================")
    print("\n" + report)