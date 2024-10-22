from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import psutil
import os
import time

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
    metrics = ["precision", "recall", "f1-score", "support"]

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    def format_value(val, is_support=False):
        if is_support:
            return int(val)
        else:
            return np.ceil(val * 100) / 100

    table_data = []
    total_support = 0
    for target in target_names:
        row = [format_value(report[target][metric], metric == "support") for metric in metrics]
        total_support += report[target]["support"]  # Somar os valores de support
        table_data.append(row)

    table_data.append([format_value(accuracy), format_value(accuracy), format_value(accuracy), int(total_support)])

    header = ["Precision", "Recall", "F1-score", "Support"]

    classes = list(target_names) + ["Accuracy/Total"]

    table = ax.table(cellText=table_data, colLabels=header, rowLabels=classes, loc="center", cellLoc='center', rowLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)


    table.auto_set_column_width(col=list(range(len(header))))
    for (i, j), cell in table.get_celld().items():
        if j == -1:
            cell.set_width(0.2)

    plt.title(f"Metrics for {title} KNN for k = {k}", fontsize=14)
    plt.show()

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2

def run_with_memory_profiling(func, *args, **kwargs):
    start_memory = memory_usage()
    func(*args, **kwargs)
    end_memory = memory_usage()
    print(f"Memory used: {end_memory - start_memory:.2f} MB"
          + "\n\n==========================================================\n")

def print_report(report, k, execution_time, title):
    print("=========================================================="
          + f"\n           Assessment Metrics for k = {k}"
          + "\n==========================================================")
    print("\n" + report)
    print(
        f"Total execution time k = {k} ({title}): {execution_time:.6f} seconds\n"
        + "\n==========================================================\n")