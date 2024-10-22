import knn_manual as knn_manual
import knn_sklearn as knn_sklearn
import utils as util

def print_default_message():
    print("=========================================================="
            + "\n           Practical Work - KNN Classification"
            + "\n=========================================================="
            + "\n\n Members: "
            + "\n  - Diogo Carrer de Macedo, 14A, 202210114"
            + "\n  - João Victor Carvalho dos Santos, 14A, 202210122 \n"
          )

def main():
    print_default_message()
    util.run_with_memory_profiling(knn_manual.run)
    util.run_with_memory_profiling(knn_sklearn.run)

if __name__ == "__main__":
    main()