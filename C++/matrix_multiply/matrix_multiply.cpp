#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void multiply_matrices_single(const std::vector<std::vector<double>>& A,
                             const std::vector<std::vector<double>>& B,
                             std::vector<std::vector<double>>& C) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void multiply_matrices_parallel(const std::vector<std::vector<double>>& A,
                               const std::vector<std::vector<double>>& B,
                               std::vector<std::vector<double>>& C) {
    int n = A.size();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    const int n = 500;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    std::vector<std::vector<double>> B(n, std::vector<double>(n, 2.0));
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));
    
    auto start = std::chrono::high_resolution_clock::now();
    multiply_matrices_single(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    auto single_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    multiply_matrices_parallel(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "1-stream: " << single_time.count() << " мс\n";
    std::cout << "mullti-stream: " << parallel_time.count() << " мс\n";
    std::cout << "speed-up: " << (double)single_time.count() / parallel_time.count() << "x\n";
    
    return 0;
}