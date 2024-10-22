import knn_manual as knn_manual
import knn_sklearn as knn_sklearn

def print_default_message():
    print("=========================================================="
            + "\n           Trabalho Prático - Classificação KNN"
            + "\n=========================================================="
            + "\n\n Integrantes: "
            + "\n  - João Victor Carvalho dos Santos, 14A, 202210122"
            + "\n  - Diogo Carrer de Macedo, 14A, 202210114 \n\n"
          )

def main():
    print_default_message()
    knn_manual.run()
    knn_sklearn.run()

if __name__ == "__main__":
    main()

