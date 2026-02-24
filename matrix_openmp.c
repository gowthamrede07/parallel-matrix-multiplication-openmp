#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 500   // Change matrix size if needed

int main() {/c/Users/reddy

    int i, j, k;
    double start, end;

    // Static allocation to avoid stack overflow
    static double A[N][N], B[N][N];
    static double C_seq[N][N], C_par[N][N];

    // Initialize matrices with random values
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C_seq[i][j] = 0.0;
            C_par[i][j] = 0.0;
        }
    }

    // Sequential matrix multiplication
    start = omp_get_wtime();
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C_seq[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = omp_get_wtime();
    double seq_time = end - start;

    // Parallel matrix multiplication using OpenMP
    start = omp_get_wtime();
    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C_par[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    end = omp_get_wtime();
    double par_time = end - start;

    // Verify correctness
    int correct = 1;
    for (i = 0; i < N && correct; i++) {
        for (j = 0; j < N; j++) {
            if (C_seq[i][j] != C_par[i][j]) {
                correct = 0;
                break;
            }
        }
    }

    // Print results
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Sequential Time: %f seconds\n", seq_time);
    printf("Parallel Time:   %f seconds\n", par_time);
    printf("Speedup:         %f\n", seq_time / par_time);

    if (correct)
        printf("Result Verification: SUCCESS\n");
    else
        printf("Result Verification: FAILED\n");

    return 0;
}
