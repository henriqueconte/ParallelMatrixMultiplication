#include <stdio.h>
#include <mpi.h>
#include <algorithm>
#include <ctime>

#define GRN   "\x1B[32m"
#define RESET "\x1B[0m"
#define CYN   "\x1B[36m"

#define MASTER_PROCESS 0

#define MASTER_TAG 0
#define WORKER_TAG 1

#define N 4
#define MICRO 1000000

#define NOT_ENOUGH_PROCESSES_NUM_ERROR 1

MPI_Status status;

int matrixA[N][N];
int matrixB[N][N];
int resultMatrix[N][N];

void generateMatrixes(int matrix[N][N]) {
	for(int i = 0; i < N; i ++){
        for(int j = 0; j < N; j ++){
          matrix[i][j] = std::rand() % 10;
        }
    }
}

void showMatrix(int matrix[N][N]) {
	printf("\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
}

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
	int offset;
	int rows_num;
	int workers_num;
	int remainder;
	int intervalLength;
	int message_tag;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // TODO: Create case where we run in only one process
	if (communicator_size < 1) {
		MPI_Abort(MPI_COMM_WORLD, NOT_ENOUGH_PROCESSES_NUM_ERROR);
	} else if (communicator_size == 1) {
        // TODO: Calculate timing for one process execution.
        matrixMultiplication(N);
        return 0;
    } 

    workers_num = communicator_size - 1;
    intervalLength = N / workers_num;
    remainder = N % workers_num;

	if (process_rank == MASTER_PROCESS) {
		printf("%sGenerating matrixes%s\n", CYN, RESET);
		
		printf("\n%sGenerating matrix %sA%s with size %s%dx%d",CYN, GRN, CYN, RESET, N, N);
		generateMatrixes(matrixA);
		showMatrix(matrixA);

		printf("\n%sGenerating matrix %sB%s with size %s%dx%d",CYN, GRN, CYN, RESET, N, N);
		generateMatrixes(matrixB);
		showMatrix(matrixB);

		printf("\nStarting multiplication ... \n");
		long long int start = clock();
		offset = 0;

		for (int process_id = 1; process_id <= workers_num; process_id++) {
            rows_num = process_id <= remainder ? intervalLength + 1 : intervalLength;
            MPI_Send(&rows_num, 1, MPI_INT, process_id, MASTER_TAG, MPI_COMM_WORLD);
            MPI_Send(&matrixA[offset], rows_num * N, MPI_INT, process_id, MASTER_TAG, MPI_COMM_WORLD);
            offset += rows_num;
		}
        MPI_Bcast(&matrixB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

        offset = 0;
		for (int process_id = 1; process_id <= workers_num; process_id ++) {
            MPI_Recv(&rows_num, 1, MPI_INT, process_id, WORKER_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&resultMatrix[offset], rows_num * N, MPI_INT, process_id, WORKER_TAG, MPI_COMM_WORLD, &status);
            offset += rows_num;
		}
		printf("\n%sResult %sA*B%s", CYN, GRN, RESET);
		showMatrix(resultMatrix);
		long long int end = clock();
		double diff = (double)((end - start) / (1.0 * MICRO));

		printf("\n%dx%d - %f seconds\n", N, N, diff);
	} 
	
	if (process_rank != MASTER_PROCESS) {
		MPI_Recv(&rows_num, 1, MPI_INT, MASTER_PROCESS, MASTER_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrixA, rows_num * N, MPI_INT, MASTER_PROCESS, MASTER_TAG, MPI_COMM_WORLD, &status);
        MPI_Bcast(&matrixB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

        matrixMultiplication(rows_num);

		MPI_Send(&rows_num, 1, MPI_INT, MASTER_PROCESS, WORKER_TAG, MPI_COMM_WORLD);
		MPI_Send(&resultMatrix, rows_num * N, MPI_INT, MASTER_PROCESS, WORKER_TAG, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}