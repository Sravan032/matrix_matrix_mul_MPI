#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <sys/time.h>

// Number of rows and columns in a matrix
#define N 1000

MPI_Status status;

// Matrix holders
double matrix_a[N][N], matrix_b[N][N], matrix_c[N][N];

int main(int argc, char **argv)
{
    int processCount, processId, slaveTaskCount, source, dest, rows, offset;

    // For timing
    struct timeval start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    slaveTaskCount = processCount - 1;

    // Validate number of slave processes
    if (slaveTaskCount <= 0) {
        if (processId == 0) {
            printf("Error: At least 2 processes are required (1 master + at least 1 slave).\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (N % slaveTaskCount != 0) {
        if (processId == 0) {
            printf("Error: Number of slave processes (%d) must exactly divide matrix size (%d).\n", slaveTaskCount, N);
            printf("Please run with np = %d (1 master + %d slaves).\n", slaveTaskCount + 1, slaveTaskCount);
        }
        MPI_Finalize();
        return 1;
    }

    rows = N / slaveTaskCount;

    // Root (Master) process
    if (processId == 0) {
        srand(time(NULL));

        // Fill matrices A and B with random numbers
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix_a[i][j] = rand() % 10;
                matrix_b[i][j] = rand() % 10;
            }
        }

        printf("\n\t\tMatrix - Matrix Multiplication using MPI\n");

        // Print Matrix A
        printf("\nMatrix A\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f\t", matrix_a[i][j]);
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

        offset = 0;

        // Start timing
        gettimeofday(&start, NULL);

        // Send data to slave tasks
        for (dest = 1; dest <= slaveTaskCount; dest++) {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix_a[offset][0], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix_b, N * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            offset += rows;
        }

        // Receive results from slave tasks
        for (int i = 1; i <= slaveTaskCount; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix_c[offset][0], rows * N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        // End timing
        gettimeofday(&end, NULL);

        // Calculate time in milliseconds
        double elapsed = (end.tv_sec - start.tv_sec) * 1000.0;
        elapsed += (end.tv_usec - start.tv_usec) / 1000.0;

        // Print the result matrix
        printf("\nResult Matrix C = Matrix A * Matrix B:\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                printf("%.0f\t", matrix_c[i][j]);
            printf("\n");
        }

        printf("\nTotal Execution Time: %.3f ms\n", elapsed);
    }

    // Slave Processes
    if (processId > 0) {
        source = 0;

        // Receive data from root process
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_a, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_b, N * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        // Perform matrix multiplication on assigned rows
        for (int k = 0; k < N; k++) {
            for (int i = 0; i < rows; i++) {
                matrix_c[i][k] = 0.0;
                for (int j = 0; j < N; j++) {
                    matrix_c[i][k] += matrix_a[i][j] * matrix_b[j][k];
                }
            }
        }

        // Send results back to root process
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&matrix_c, rows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
