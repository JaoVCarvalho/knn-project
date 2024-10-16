# Importando as bibliotecas necessárias
from sklearn.datasets import load_iris  # Biblioteca Sklearn tem um conjunto de dados Iris já pronto
import pandas as pd  # Pandas será usado para organizar os dados de uma forma mais legível

# Carregar o banco de dados Iris
iris = load_iris()  # Aqui estamos carregando o conjunto de dados Iris com a função load_iris

# Criar um DataFrame do Pandas para organizar os dados de forma tabular
# A base Iris tem os dados das flores em 'data' e as classes (espécies) em 'target'
iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target'] # Adiciona a coluna 'species' com as espécies (ou classe) de cada flor

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Imprimir o DataFrame no console
print(iris_df)  # Mostra os dados da base Iris organizados
