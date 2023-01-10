#include <stdio.h>
#include <mpi.h>
#include <algorithm>
#include <ctime>

#define GRN   "\x1B[32m"
#define RESET "\x1B[0m"
#define CYN   "\x1B[36m"

#define MASTER_RANK 0

#define MASTER_TAG 0
#define WORKER_TAG 1

#define N 4
#define MICRO 1000000

#define NOT_ENOUGH_PROCESSES_NUM_ERROR 1

MPI_Status status;

int matrixA[N][N];
int matrixB[N][N];
int resultMatrix[N][N];

int GenerateRandomNumber() {
	return std::rand() % 9 + 1;
}

template<int rows, int cols> 
void generateMatrixes(int (&matrix)[rows][cols]) {
	for(int i = 0; i < cols; i ++){
        for(int j = 0; j < rows; j ++){
          matrix[i][j] = GenerateRandomNumber();
        }
    }
}

template<int rows, int cols> 
void PrintMatrix(int (&matrix)[rows][cols]){
	printf("\n");
	for(int i = 0; i < rows; i ++){
		for(int j = 0; j < cols; j ++){
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]) {
	int communicator_size;
	int process_rank;
	int process_id;
	int offset;
	int rows_num;
	int workers_num;
	int remainder;
	int intervalLength;
	int message_tag;
	int i;
	int j;
	int k;


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // TODO: Create case where we run in only one process
	if (communicator_size < 2) {
		MPI_Abort(MPI_COMM_WORLD, NOT_ENOUGH_PROCESSES_NUM_ERROR);
	}

    workers_num = communicator_size - 1;
    intervalLength = N / workers_num;
    remainder = N % workers_num;

	if (process_rank == MASTER_RANK) {
		printf("%sGenerating matrixes%s\n", CYN, RESET);
		
		printf("\n%sGenerating matrix %sA%s with size %s%dx%d",CYN, GRN, CYN, RESET, N, N);
		generateMatrixes(matrixA);
		PrintMatrix(matrixA);

		printf("\n%sGenerating matrix %sB%s with size %s%dx%d",CYN, GRN, CYN, RESET, N, N);
		generateMatrixes(matrixB);
		PrintMatrix(matrixB);

		printf("\nStarting multiplication ... \n");
		long long int start = clock();
		offset = 0;

		message_tag = MASTER_TAG;
		for (process_id = 1; process_id <= workers_num; process_id++) {
                rows_num = process_id <= remainder ? intervalLength + 1 : intervalLength;
                MPI_Send(&rows_num, 1, MPI_INT, process_id, message_tag, MPI_COMM_WORLD);
                MPI_Send(&matrixA[offset], rows_num * N, MPI_INT, process_id, message_tag, MPI_COMM_WORLD);
                offset += rows_num;
		}
        MPI_Bcast(&matrixB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

		message_tag = WORKER_TAG;
        offset = 0;
		for (process_id = 1; process_id <= workers_num; process_id ++) {
                MPI_Recv(&rows_num, 1, MPI_INT, process_id, message_tag, MPI_COMM_WORLD, &status);
                MPI_Recv(&resultMatrix[offset], rows_num * N, MPI_INT, process_id, message_tag, MPI_COMM_WORLD, &status);
            offset += rows_num;
		}
		printf("\n%sResult %sA*B%s", CYN, GRN, RESET);
		PrintMatrix(resultMatrix);
		long long int end = clock();
		double diff = (double)((end - start) / (1.0 * MICRO));

		printf("\n%dx%d - %f seconds\n", N, N, diff);
	} 
	
	if (process_rank != MASTER_RANK) {
		message_tag = MASTER_TAG;
		MPI_Recv(&rows_num, 1, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&matrixA, rows_num * N, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD, &status);
        MPI_Bcast(&matrixB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

        // printf("rows num: %d \n", rows_num);
		for (k = 0; k < N; k ++) {
			for (i = 0; i < rows_num; i ++) {
                printf("Process rank: %d, Pair: (%d, %d) \n", process_rank, i, k);
				for (j = 0; j < N; j ++) {
					resultMatrix[i][k] += matrixA[i][j] * matrixB[j][k];
				}
			}
		}

		message_tag = WORKER_TAG;
		// MPI_Send(&offset, 1, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD);
		MPI_Send(&rows_num, 1, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD);
		MPI_Send(&resultMatrix, rows_num * N, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}