#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <iostream>

using namespace std;

#define N 1000 // Matrix size

MPI_Status status;
double matrix_b[N][N], matrix_c[N][N];  // matrix_b and matrix_c shared globally

// Function to test correctness
void testCorrectness(const vector<vector<int>>& A, const vector<vector<int>>& B, const vector<vector<int>>& C_computed) {
    int M = A.size();
    int K = A[0].size();
    int N_cols = B[0].size();
    vector<vector<int>> C_expected(M, vector<int>(N_cols, 0));

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N_cols; ++j)
            for (int k = 0; k < K; ++k)
                C_expected[i][j] += A[i][k] * B[k][j];

    bool correct = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N_cols; ++j) {
            if (C_computed[i][j] != C_expected[i][j]) {
                cout << "Mismatch at (" << i << ", " << j << "): expected " << C_expected[i][j]
                     << ", got " << C_computed[i][j] << endl;
                correct = false;
            }
        }
    }
    if (correct)
        cout << "Test passed: Computed matrix matches expected result." << endl;
    else
        cout << "Test failed: Mismatches found." << endl;
}

int main(int argc, char **argv) {
    int processCount, processId;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    // Calculate rows distribution
    vector<int> sendcounts(processCount, 0);
    vector<int> displs(processCount, 0);
    int baseRows = N / processCount;
    int extraRows = N % processCount;

    for (int i = 0; i < processCount; i++) {
        sendcounts[i] = (baseRows + (i < extraRows ? 1 : 0)) * N; // Number of elements per process
    }
    displs[0] = 0;
    for (int i = 1; i < processCount; i++) {
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    int local_rows = sendcounts[processId] / N;

    // Local buffers
    vector<double> local_a(sendcounts[processId]); // local rows of A
    vector<double> local_c(sendcounts[processId], 0); // local result C

    // Root process initializes matrix_a and matrix_b
    vector<double> matrix_a;
    if (processId == 0) {
        srand(time(NULL));
        matrix_a.resize(N * N);
        for (int i = 0; i < N * N; i++) {
            matrix_a[i] = rand() % 10;
            matrix_b[0][0] = 0; // just to silence unused warning for matrix_b for now, will be overwritten later
        }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                matrix_b[i][j] = rand() % 10;

        // Print Matrix A
        printf("\nMatrix A\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f\t", matrix_a[i * N + j]);
            }
            printf("\n");
        }

        // Print Matrix B
        printf("\nMatrix B\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f\t", matrix_b[i][j]);
            }
            printf("\n");
        }
    }

    // Broadcast matrix_b to all processes
    MPI_Bcast(&matrix_b[0][0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of matrix_a to all processes
    MPI_Scatterv(processId == 0 ? &matrix_a[0] : NULL, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 &local_a[0], sendcounts[processId], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start timing on root
    double startTime = 0;
    if (processId == 0) {
        startTime = MPI_Wtime();
    }

    // Compute local matrix multiplication: local_c = local_a * matrix_b
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += local_a[i * N + k] * matrix_b[k][j];
            }
            local_c[i * N + j] = sum;
        }
    }

    // Gather local results back to matrix_c on root
    MPI_Gatherv(&local_c[0], sendcounts[processId], MPI_DOUBLE,
                processId == 0 ? &matrix_c[0][0] : NULL, sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Root prints results and tests correctness
    if (processId == 0) {
        double endTime = MPI_Wtime();
        double computationTime = endTime - startTime;

        printf("\nResult Matrix C = Matrix A * Matrix B:\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f\t", matrix_c[i][j]);
            }
            printf("\n");
        }

        // Convert to int vectors for testing correctness
        vector<vector<int>> A(N, vector<int>(N));
        vector<vector<int>> B(N, vector<int>(N));
        vector<vector<int>> C_computed(N, vector<int>(N));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = (int)matrix_a[i * N + j];
                B[i][j] = (int)matrix_b[i][j];
                C_computed[i][j] = (int)matrix_c[i][j];
            }
        }

        testCorrectness(A, B, C_computed);

        printf("Matrix multiplication took %.6f seconds\n", computationTime);
    }

    MPI_Finalize();
    return 0;
}
