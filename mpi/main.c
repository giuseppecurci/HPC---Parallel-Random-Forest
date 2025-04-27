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
		// capire questa parte qua
    	stratified_split(data, num_rows, num_columns, train_proportion, &train_data, &train_size, &test_data, &test_size, seed);
		
		// Compute total length of train data
		int train_total_length = train_size * num_columns;

		// Broadcast important dimensions first
		MPI_Bcast(&train_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Now broadcast only the train_data
		MPI_Bcast(train_data, train_total_length, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
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

	}

	else {
		int train_total_length, test_total_length;
		
		MPI_Bcast(&train_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);


		
		printf("\n");
		printf("Num_rows = %d, Num_columns = %d, myrank = %d", train_size, num_columns, rank);
		printf("\n");
	
		// Now allocate the right size
		train_total_length = train_size * num_columns;
		train_data = (float *)malloc(train_total_length * sizeof(float));
		if (!train_data) {
			perror("Malloc failed for train_data");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Receive train data
		MPI_Bcast(train_data, train_total_length, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		// Allocate memory for targets
		if (rank != 0) {
			targets = (int *)malloc(test_size * sizeof(int));
		}
		
		MPI_Bcast(targets, test_size, MPI_INT, 0, MPI_COMM_WORLD);
		// Fino a qui checkato e testato tutto
		free(train_data);
	}


	for (int p = 0; p < process_number; p++) {
    if (rank == p) {
        printf("\n========== Process %d Info ==========\n", rank);
        printf("Rank: %d/%d\n", rank, process_number-1);
        printf("Dataset dimensions: %d rows, %d columns\n", train_size, num_columns);
        printf("Train set: %d samples\n", train_size);
        printf("Test set: %d samples\n", test_size);
        
        // Print first row of train data as sample
        printf("First row of training data:\n");
        for(int i = 0; i < num_columns; i++) {
            int index = (0 * num_columns) + i;
            printf("  Column %d: %f\n", i, train_data[index]);
        }
        
        // Print first few target values
        printf("First 5 target values (or fewer if less available):\n");
        int target_display = test_size < 5 ? test_size : 5;
        for(int i = 0; i < target_display; i++) {
            printf("  targets[%d] = %d\n", i, targets[i]);
        }
        
        // Memory info
        if (rank == 0) {
            printf("Process 0 is the coordinator\n");
        } else {
            printf("Process received %d floats in train_data\n", train_size * num_columns);
            printf("Process received %d integers in targets\n", test_size);
        }
        printf("=======================================\n");
    }
    MPI_Barrier(MPI_COMM_WORLD); // synchronize all processes before next one prints
}
	free(targets);
	MPI_Finalize();
	return 0;
}
