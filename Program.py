import pandas as pd
import math
from collections import Counter


# Função para calcular a distância Euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


# Função para encontrar os k vizinhos mais próximos
def get_k_neighbors(training_data, test_point, k):
    distances = []
    # Calcula a distância do ponto de teste para todos os pontos de treinamento
    for train_point in training_data:
        distance = euclidean_distance(test_point[:-1], train_point[:-1])  # Ignora a classe
        distances.append((train_point, distance))

    # Ordena os pontos de treinamento pela distância (menor para maior)
    distances.sort(key=lambda x: x[1])

    # Retorna os k vizinhos mais próximos
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors


# Função para prever a classe com base nos vizinhos
def predict_classification(neighbors):
    # Coleta as classes dos vizinhos
    classes = [neighbor[-1] for neighbor in neighbors]

    # Conta a frequência de cada classe
    most_common = Counter(classes).most_common(1)

    # Retorna a classe mais comum
    return most_common[0][0]


# Função principal do algoritmo KNN
def knn_algorithm(training_data, test_data, k):
    predictions = []

    for test_point in test_data:
        # Encontra os k vizinhos mais próximos
        neighbors = get_k_neighbors(training_data, test_point, k)

        # Prediz a classe com base nos vizinhos
        predicted_class = predict_classification(neighbors)

        # Adiciona a previsão à lista
        predictions.append(predicted_class)

    return predictions


# Função para calcular a acurácia do modelo
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual) * 100.0


# Função para embaralhar e dividir os dados em treinamento e teste
def split_data(df, test_size=0.2):
    # Embaralhar os dados aleatoriamente
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Determinar o tamanho do conjunto de teste (20% dos dados)
    test_len = int(len(df) * test_size)

    # Dividir os dados em treinamento e teste
    test_data = df_shuffled[:test_len]
    training_data = df_shuffled[test_len:]

    return training_data, test_data


# Função para normalizar os dados usando Min-Max Scaling
def normalize(X):
    return (X - X.min()) / (X.max() - X.min())


# Função principal
def main():
    # 1. Carregar o conjunto de dados Iris
    iris_df = pd.read_csv('iris.data')

    # 2. Imprimir os 5 primeiros elementos dos dados
    print("Dados originais:\n", iris_df.head())

    # 3. Excluir a coluna 'Id' e manter apenas as características e a espécie
    iris_df = iris_df.drop(columns=['Id'])

    # 4. Separar as características (features) e as classes (species)
    X = iris_df.iloc[:, :-1]  # Todas as colunas exceto a última (Species)
    y = iris_df.iloc[:, -1]  # A última coluna (Species)

    # 5. Normalizar as características (X) para o intervalo [0, 1]
    X_normalized = normalize(X)

    # 6. Visualizar os dados normalizados
    print("\nDados normalizados (primeiras 5 linhas):")
    print(X_normalized.head())

    # 7. Combinar as características normalizadas e as classes
    iris_normalized = pd.concat([X_normalized, y], axis=1)
    print("\nDados normalizados e concatenados com as classes (primeiras 5 linhas):")
    print(iris_normalized.head())

    # 8. Dividir os dados em treinamento (80%) e teste (20%)
    training_data, test_data = split_data(iris_normalized)

    # 9. Exibir os tamanhos dos conjuntos de treinamento e teste
    print("\nTamanho do conjunto de treinamento:", len(training_data))
    print("Tamanho do conjunto de teste:", len(test_data))

    # 10. Rodar o algoritmo KNN para k = 3
    k = 1
    test_points = test_data.iloc[:, :-1].values  # Dados de teste sem as classes
    actual_classes = test_data.iloc[:, -1].values  # Classes reais dos dados de teste

    print(f"\nRodando o algoritmo KNN com k = {k}...")
    predicted_classes = knn_algorithm(training_data.values, test_points, k)

    # 11. Exibir as classes reais e as classes previstas
    print("\nClasses Reais vs. Classes Previstas:")
    for actual, predicted in zip(actual_classes, predicted_classes):
        print(f"Real: {actual}, Previsto: {predicted}")

    # 12. Calcular a acurácia
    accuracy = calculate_accuracy(actual_classes, predicted_classes)
    print(f"\nAcurácia do modelo (k = {k}): {accuracy:.2f}%")

if __name__ == "__main__":
    main()
