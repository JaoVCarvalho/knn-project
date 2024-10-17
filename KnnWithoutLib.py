import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score

def euclidean_distance(point1, point2): # Função para calcular a distância euclidiana
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_k_nearest_neighbors(X_train, y_train, test_sample, k): # Função para encontrar os k vizinhos mais próximos
    distances = []

    for i in range(len(X_train)): # Calcula a distância entre o ponto de teste e todos os pontos de treino
        distance = euclidean_distance(X_train[i], test_sample)
        distances.append((distance, y_train[i])) # Salva a distância e o rótulo correspondente

    distances.sort(key=lambda x: x[0]) # Ordena os vizinhos pela distância (do menor para o maior)
    neighbors = [distances[i][1] for i in range(k)] # Retorna os rótulos dos k vizinhos mais próximos

    return neighbors

def predict(X_train, y_train, X_test, k): # Função para prever a classe com base na maioria dos vizinhos
    predictions = []

    for test_sample in X_test:
        neighbors = get_k_nearest_neighbors(X_train, y_train, test_sample, k)
        most_common = Counter(neighbors).most_common(1)[0][0] # Determina a classe mais comum entre os vizinhos
        predictions.append(most_common)

    return predictions

iris = load_iris()
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

for k in [1, 3, 5, 7]:
    print(f"Resultados para k = {k}:")
    y_predict = predict(X_train, y_train, X_test, k)
    print(classification_report(y_test, y_predict, target_names=iris.target_names))
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Acurácia: {accuracy:.2f}\n")