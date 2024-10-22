import knn_manual as knn_manual
import knn_sklearn as knn_sklearn
import psutil
import os

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2

def run_with_memory_profiling(func, *args, **kwargs):
    start_memory = memory_usage()
    func(*args, **kwargs)
    end_memory = memory_usage()

    print(f"Memória usada: {end_memory - start_memory:.2f} MB" + "\n\n==========================================================\n")

def print_default_message():
    print("=========================================================="
            + "\n           Trabalho Prático - Classificação KNN"
            + "\n=========================================================="
            + "\n\n Integrantes: "
            + "\n  - João Victor Carvalho dos Santos, 14A, 202210122"
            + "\n  - Diogo Carrer de Macedo, 14A, 202210114 \n"
          )

def main():
    print_default_message()
    run_with_memory_profiling(knn_manual.run)
    run_with_memory_profiling(knn_sklearn.run)

if __name__ == "__main__":
    main()