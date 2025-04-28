// MPI Implementation#include <stdio.h>

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "headers/utils.h"
#include "headers/metrics.h"
#include "headers/forest.h"

#include "headers/tree/tree.h"
#include "headers/tree/utils.h"
#include "headers/tree/train_utils.h"

int main(int argc, char *argv[]) {
    int max_matrix_rows_print = 0; // Default: print nothing
    int num_classes = -1; 
    char *new_forest_path = "output/model"; 
    char *trained_forest_path = NULL; 
    char *store_predictions_path = "output/predictions.csv"; 
    char *store_metrics_path = "output/metrics_output.txt"; 
    float train_proportion = 0.8;
    int num_trees = 10;
    char* max_features = "sqrt";
    int min_samples_split = 2;
    int max_depth = 10;
    int seed = 0;
    char *dataset_path = "../data/classification_dataset.csv";  
    int num_rows, num_columns;
	float *data;
    float *train_data, *test_data;
	int *targets;
    int train_size, test_size;
	int sample_size;
	int train_total_length, test_total_length;
    
    // Parse command-line arguments
    int parse_result = parse_arguments(argc, argv, &max_matrix_rows_print, &num_classes, &num_trees,
                                        &max_depth, &min_samples_split, &max_features,
                                        &trained_forest_path, &store_predictions_path,
                                        &store_metrics_path, &new_forest_path, &dataset_path,
                                        &train_proportion, &seed);
    if (parse_result != 0) {
        printf("Error parsing arguments. Please check the command line options.\n");
        return 1;
    }

	int rank, process_number;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &process_number);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// PROCESS 0
	if(rank == 0) {
		// check if the directory of the dataset exists
		struct stat st = {0};
		char parent_dir[256] = {0};
		char *slash;
		strncpy(parent_dir, new_forest_path, sizeof(parent_dir) - 1);
		slash = strrchr(parent_dir, '/');
		if (slash) {
			*slash = '\0';  // Terminate the string at the last slash
			if (stat(parent_dir, &st) == -1) {
				if (mkdir(parent_dir, 0700) == -1) {
					perror("Parent directory creation failed");
					return 1;
				}
				printf("Parent directory created: %s\n", parent_dir);
			}
		}

		// Now try to create the full path
		if (stat(new_forest_path, &st) == -1) {
			if (mkdir(new_forest_path, 0700) == 0) {
				printf("Directory created: %s\n", new_forest_path);
			} else {
				perror("mkdir failed");
			}
		}

		data = read_csv(dataset_path, &num_rows, &num_columns);
		
		if (num_classes <= 0) {
			printf("Inferring number of classes from the dataset...\n");
			for (int i = 0; i < num_rows; i++) {
				int label = (int)data[i * num_columns + (num_columns - 1)];  // Access the label
				if (label > num_classes) {
					num_classes = label;
				}
			}
			num_classes++;
		}

    	stratified_split(data, num_rows, num_columns, num_classes, train_proportion, &train_data, &train_size, &test_data, &test_size, seed);
		// Broadcast important dimensions first
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		// Sample 75% of the train data and send it over
		for (int p = 1; p < process_number; p++) {
			sample_size = (int)(0.75 * train_size);
			int *sampled_indices = (int *)malloc(sample_size * sizeof(int));

			// Random sampling without replacement
			for (int i = 0; i < sample_size; i++) {
				int idx;
				int unique;
				do {
					unique = 1;
					idx = rand() % train_size;
					// Check if idx was already chosen
					for (int j = 0; j < i; j++) {
						if (sampled_indices[j] == idx) {
							unique = 0;
							break;
						}
					}
				} while (!unique);
				sampled_indices[i] = idx;
			}

			// Allocate a temporary buffer for sampled data
			float *sampled_data = (float *)malloc(sample_size * num_columns * sizeof(float));

			for (int i = 0; i < sample_size; i++) {
				int original_row = sampled_indices[i];
				for (int j = 0; j < num_columns; j++) {
					sampled_data[i * num_columns + j] = train_data[original_row * num_columns + j];
				}
			}

			// First, send the sample_size to process p
			MPI_Send(&sample_size, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

			// Now send the actual flattened sampled data
			MPI_Send(sampled_data, sample_size * num_columns, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
			free(sampled_indices);
			free(sampled_data);
		}
		targets = (int *)malloc(test_size * sizeof(int));
	
		for (int i = 0; i < test_size; i++) {
			targets[i] = (int)test_data[i * num_columns + (num_columns - 1)];
		}

		if (num_classes <= 0) {
			printf("Inferring number of classes from the dataset...\n");
			for (int i = 0; i < test_size; i++) {
				if (targets[i] > num_classes) num_classes = (int)targets[i];
			}
			num_classes++;
		}

		MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(targets, test_size, MPI_INT, 0, MPI_COMM_WORLD);

    	summary(dataset_path, train_proportion, train_size, num_columns - 1, num_classes,
            num_trees, max_depth, min_samples_split, max_features, store_predictions_path, 
            store_metrics_path, new_forest_path, trained_forest_path, seed);
	}
// OTHER PROCESSES
	else {
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Recv(&sample_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Allocate memory
		train_data = (float *)malloc(sample_size * num_columns * sizeof(float));
		if (!train_data) {
			perror("Malloc failed for train_data");
			MPI_Abort(MPI_COMM_WORLD, 1);}
		// Receive train_data
		MPI_Recv(train_data, sample_size * num_columns, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		printf("\n");
		printf("Num_rows = %d, Num_columns = %d, myrank = %d", train_size, num_columns, rank);
		printf("\n");
		
		// Allocate memory for targets
		targets = (int *)malloc(test_size * sizeof(int));
		if (!targets) {
			perror("Malloc failed for targets");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(targets, test_size, MPI_INT, 0, MPI_COMM_WORLD);
	}
// RECAP
	for (int p = 0; p < process_number; p++) {
		if (rank == p) {
			printf("\n========== Process %d Info ==========\n", rank);
			printf("Rank: %d/%d\n", rank, process_number-1);
			printf("Train Dataset dimensions: %d rows, %d columns\n", sample_size, num_columns);
			printf("Test Dataset dimensions: %d rows, %d columns\n", test_size, num_columns);
			
			// Print first row of train data as sample
			printf("First row of training data:\n");
			for(int i = 0; i < num_columns; i++) {
				int index = (0 * num_columns) + i;
				printf("  Column %d: %f\n", i, train_data[index]);
			}
			
			// Memory info
			if (rank == 0) {
				printf("Process 0 is the coordinator\n");
			} else {
				printf("Process received %d floats in train_data\n", sample_size * num_columns);
				printf("Process received %d integers in targets\n", test_size);
			}
			printf("=======================================\n");
		}
		MPI_Barrier(MPI_COMM_WORLD); // synchronize all processes before next one prints
	}
	free(targets);
	free(train_data);
	MPI_Finalize();
	return 0;
}

