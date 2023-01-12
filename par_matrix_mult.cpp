#include <stdio.h>
#include <algorithm>
#include <mpi.h>
#include <ctime>

// Master process rank
#define MASTER_PROCESS 0

// Tags used to send message
#define MASTER_TAG 0
#define WORKER_TAG 1

// Error codes
#define INVALID_PROCESSES 1

// Matrix size
#define N 4

MPI_Status status;

int matrixA[N][N];
int matrixB[N][N];
int resultMatrix[N][N];

// Prints matrix
void showMatrix(int matrix[N][N]) {
	printf("\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
}

// Given a matrix, it will generate random values for all its fields
void generateMatrixes(int matrix[N][N]) {
	for(int i = 0; i < N; i ++){
        for(int j = 0; j < N; j ++){
          matrix[i][j] = std::rand() % 10;
        }
    }
}

// Multiplicates part of the matrix A and B
void matrixMultiplication(int interval) {
    for (int k = 0; k < N; k ++) {
        for (int i = 0; i < interval; i ++) {
            for (int j = 0; j < N; j ++) {
                resultMatrix[i][k] += matrixA[i][j] * matrixB[j][k];
            }
        }
    }
}

int main(int argc, char *argv[]) {
	int communicator_size, process_rank;

    // Offset to know which part of the matrix the worker will use
	int rowsOffset;

    // The amount of rows each worker will work with.
	int workingRows;

    // The amount of workers 
	int workersCount;

    // We will try to divide the amount of rows equally between the workers, but we might have a few rows left.
    // Therefore, we keep track of the remainder to split extra rows among the workers.
	int remainder;

    // The minimum amount of rows that each worker should work with.
	int intervalLength;

    // Initializes MPI environment, defining number of processes available and their ranks.
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // If we don't have more than one process available, we abort the execution
	if (communicator_size < 1) {
		MPI_Abort(MPI_COMM_WORLD, INVALID_PROCESSES);
	} else if (communicator_size == 1) { // If we only have one process, we run a sequential execution.
        long long int start = clock();

		printf("\nCreating matrix A, size %dx%d", N, N);
		generateMatrixes(matrixA);
		showMatrix(matrixA);

		printf("\nCreating matrix B, size %dx%d", N, N);
		generateMatrixes(matrixB);
		showMatrix(matrixB);

        printf("\nMultiplicating\n");

        matrixMultiplication(N);

        printf("\nResult:");
		showMatrix(resultMatrix);

        long long int end = clock();
		double diff = (double)((end - start) / (1.0 * 1000000));

		printf("\n%dx%d - %f seconds\n", N, N, diff);
        
        MPI_Finalize();
        return 0;
    } 

    // One process will be the master; the others will be workers
    workersCount = communicator_size - 1;

    // The interval that each worker will process
    intervalLength = N / workersCount;

    // The remaining rows that will be distributed between the workers
    remainder = N % workersCount;

    // Master process code
	if (process_rank == MASTER_PROCESS) {

        // Generates matrix A
		printf("\nCreating matrix A, size %dx%d", N, N);
		generateMatrixes(matrixA);
		showMatrix(matrixA);

        // Generates matrix B
		printf("\nCreating matrix B, size %dx%d", N, N);
		generateMatrixes(matrixB);
		showMatrix(matrixB);

		printf("\nMultiplicating\n");
        
        // Starts timer
		long long int start = clock();

        // For each worker process, send them the amount of rows they have to process and the part
        // of the matrix A they will use.
		rowsOffset = 0;
		for (int process_id = 1; process_id <= workersCount; process_id++) {
            workingRows = process_id <= remainder ? intervalLength + 1 : intervalLength;
            MPI_Send(&workingRows, 1, MPI_INT, process_id, MASTER_TAG, MPI_COMM_WORLD);
            MPI_Send(&matrixA[rowsOffset], workingRows * N, MPI_INT, process_id, MASTER_TAG, MPI_COMM_WORLD);
            rowsOffset += workingRows;
		}

        // Broadcasts the whole matrix B to all workers.
        MPI_Bcast(&matrixB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

        // Receives the multiplication result of each matrix part and assigns it to the resulting matrix
        rowsOffset = 0;
		for (int process_id = 1; process_id <= workersCount; process_id ++) {
            MPI_Recv(&workingRows, 1, MPI_INT, process_id, WORKER_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&resultMatrix[rowsOffset], workingRows * N, MPI_INT, process_id, WORKER_TAG, MPI_COMM_WORLD, &status);
            rowsOffset += workingRows;
		}

        // Prints resulting matrix
		printf("\nResult:");
		showMatrix(resultMatrix);

        // Finishes timer and prints execution length
		long long int end = clock();
		double diff = (double)((end - start) / (1.0 * 1000000));
		printf("\n%dx%d - %f seconds\n", N, N, diff);

	} else { // Worker process code

        // Receives the amount of rows it should process and the part of the matrixA it will work with.
		MPI_Recv(&workingRows, 1, MPI_INT, MASTER_PROCESS, MASTER_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrixA, workingRows * N, MPI_INT, MASTER_PROCESS, MASTER_TAG, MPI_COMM_WORLD, &status);

        // Receives the matrixB broadcast. The broadcast method is the same to both send and receive data.
        MPI_Bcast(&matrixB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

        // Multiplies the part of the matrix
        matrixMultiplication(workingRows);

        // Sends the amount of rows it worked with and the resulting matrix
		MPI_Send(&workingRows, 1, MPI_INT, MASTER_PROCESS, WORKER_TAG, MPI_COMM_WORLD);
		MPI_Send(&resultMatrix, workingRows * N, MPI_INT, MASTER_PROCESS, WORKER_TAG, MPI_COMM_WORLD);
    }

    // Finalizes the MPI execution
	MPI_Finalize();
	return 0;
}