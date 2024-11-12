# Algoritmo KNN


## Índice

- [Descrição](#descricao)
- [Equipe](#equipe)
- [Relatório](#relatorio)
  
## Descrição

Este repositório contém o trabalho 01 da disciplina de Inteligência Artificial do curso de Sistemas de Informação da Universidade Federal de Lavras. O trabalho consiste na implementação do algoritmo KNN(vizinho-mais-próximo), utilizando a base de dados Iris, utilizada para a classificação de espécies de flores.

O desafio do trabalho envolve desenvolver um classificador em Python, sem recorrer a bibliotecas que já implementam o KNN. O algoritmo será avaliado para os valores de k = {1, 3, 5, 7}, e será mostrada a taxa de reconhecimento correspondente. Além disso, o trabalho inclui a geração de uma matriz de confusão e a apresentação das métricas de avaliação, como precisão, revocação e acurácia. 

Após a implementação manual, será desenvolvido um novo classificador utilizando a biblioteca Sklearn, ou outra similar que já contenha o KNN implementado. As mesmas métricas de avaliação e a matriz de confusão serão geradas para comparar os resultados obtidos com a implementação manual.

## Equipe

- [Diogo Carrer de Macedo](https://github.com/diogocarrer)
- [João Victor Carvalho dos Santos](https://github.com/JaoVCarvalho) 

## Relatório

A análise das métricas de avaliação dos classificadores KNN, tanto na implementação manual quanto na biblioteca Sklearn, revelou resultados idênticos em diferentes configurações de K, conforme a randomização realizada com: 
```python
train_test_split(attributes, species, test_size=0.20, random_state=13)
```

Ambos os classificadores apresentaram desempenhos coincidentes, validando a eficácia da implementação manual e confirmando a precisão da biblioteca Sklearn. A precisão e revocação foram iguais entre os dois modelos para todas as configurações de K.

Em relação ao Tempo de Execução, o classificador manual apresentou um tempo total de 0.12 segundos, enquanto o classificador Sklearn foi significativamente mais rápido, com 0.01 segundos. Essa diferença indica que a implementação da biblioteca é mais eficiente, devido à otimização interna e ao uso de estruturas de dados mais adequadas. 

Em relação ao Uso de Memória, em termos de consumo de memória, o classificador manual utilizou 37.27 MB, enquanto o Sklearn consumiu 25.35 MB. Embora ambos os métodos consumam uma quantidade razoável de memória, a implementação da biblioteca se mostrou mais econômica, refletindo uma melhor gestão de recursos. Foi utilizada a biblioteca psutil para fazer a medição.

É relevante destacar que, ao executar o código novamente, os valores de tempo e memória irão variar, mas é esperado que permaneçam em uma faixa semelhante. Isso se deve a fatores como a carga do sistema e a aleatoriedade na divisão dos dados, que podem influenciar o desempenho dos classificadores.
