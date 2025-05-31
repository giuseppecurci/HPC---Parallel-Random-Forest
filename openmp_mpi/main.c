// MPI Implementation 
#include <stdio.h>
#include <float.h>  // For DBL_MAX
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

#include "headers/utils.h"
#include "headers/metrics.h"
#include "headers/forest.h"
#include "headers/memory_ser.h"

#include "headers/tree/tree.h"
#include "headers/tree/utils.h"
#include "headers/tree/train_utils.h"
#include "headers/tree/memory_ser.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <float.h>  // For DBL_MAX
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

#include "headers/utils.h"
#include "headers/metrics.h"
#include "headers/forest.h"
#include "headers/memory_ser.h"

#include "headers/tree/tree.h"
#include "headers/tree/utils.h"
#include "headers/tree/train_utils.h"
#include "headers/tree/memory_ser.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[]) {
    int max_matrix_rows_print = 0; // Default: print nothing
    int num_classes = -1;
    char *new_forest_path = "output/model";
    char *trained_forest_path = NULL;
    char *store_predictions_path = "output/predictions.csv";
    char *store_metrics_path = "output/metrics_output.txt";
    float train_proportion = 0.8;
    float train_tree_proportion = 0.75;
    int num_trees = 10;
    char* max_features = "sqrt";
    int min_samples_split = 2;
    int max_depth = 10;
    int seed = 0;
    char *dataset_path = "../data/classification_dataset.csv";
    int thread_count = 1;
    int num_rows, num_columns;
	float *data;
    float *train_data, *test_data;
	int *targets;
    int train_size, test_size;
	int sample_size;

	// Variables for timing (used by worker processes)
	double train_start, train_end;
	double infer_start, infer_end;
	double train_time = 0.0, inference_time = 0.0, total_time = 0.0;

	// Parse command-line arguments
    int parse_result = parse_arguments(argc, argv, &max_matrix_rows_print, &num_classes, &num_trees,
                                        &max_depth, &min_samples_split, &max_features,
                                        &trained_forest_path, &store_predictions_path,
                                        &store_metrics_path, &new_forest_path, &dataset_path,
                                        &train_proportion, &train_tree_proportion, &seed, &thread_count);
    if (parse_result != 0) {
        printf("Error parsing arguments. Please check the command line options.\n");
        return 1;
    }

	int rank, process_number;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &process_number);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		// Master process: read data and distribute
		check_dir_existence(new_forest_path);

		data = read_csv(dataset_path, &num_rows, &num_columns);
		
		if (num_classes <= 0) {
			for (int i = 0; i < num_rows; i++) {
				int label = (int)data[i * num_columns + (num_columns - 1)];  // Access the label
				if (label > num_classes) {
					num_classes = label;
				}
			}
			num_classes++;
		}

    	stratified_split(data, num_rows, num_columns, num_classes, train_proportion,
						&train_data, &train_size, &test_data, &test_size, seed);
		if (test_data == NULL) {
			printf("test_data is NULL after stratified_split. Aborting.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}	
		
		// Broadcast dimensions first
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		// input sample size
		sample_size = (int)(train_tree_proportion * train_size);
		
		// Sample and send data to worker processes
		for (int p = 1; p < process_number; p++) {
			// Allocate sampled_data buffer
			float *sampled_data = (float *)malloc(sample_size * num_columns * sizeof(float));
			if (sampled_data == NULL) {
				fprintf(stderr, "Failed to allocate memory for sampled data\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			
			// Call sampling function with different seed for each process
			int actual_sample_size = sample_data_without_replacement(
				train_data, train_size, num_columns, train_tree_proportion, sampled_data, seed + p);
				
			if (actual_sample_size <= 0) {
				fprintf(stderr, "Error in sampling data for process %d\n", p);
				free(sampled_data);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			
			// Send the data size first
			MPI_Send(&actual_sample_size, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
			
			// Send the sampled data
			MPI_Send(sampled_data, actual_sample_size * num_columns, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
			
			// Free the buffer
			free(sampled_data);
		}

		// Prepare target data for broadcasting
		targets = (int *)malloc(test_size * sizeof(int));
		for (int i = 0; i < test_size; i++) {
			targets[i] = (int)test_data[i * num_columns + (num_columns - 1)];
		}

		// Broadcast test data and targets
		MPI_Bcast(targets, test_size, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(test_data, test_size * num_columns, MPI_FLOAT, 0, MPI_COMM_WORLD);

    	summary(dataset_path, train_proportion, train_tree_proportion, train_size, num_columns - 1, num_classes,
            num_trees, max_depth, min_samples_split, max_features, store_predictions_path,
            store_metrics_path, new_forest_path, trained_forest_path, seed);
			
		// Calculate distribution of trees across worker processes (ranks 1 to process_number-1)
		int worker_count = process_number - 1;
		int *tree_counts = (int *)malloc(worker_count * sizeof(int));
		int *tree_displs = (int *)malloc(worker_count * sizeof(int));
		
		// Distribute trees among worker processes only
		distribute_trees(num_trees, worker_count, tree_counts, tree_displs);

		for (int p = 0; p < worker_count; p++) {
			MPI_Send(&tree_counts[p], 1, MPI_INT, p + 1, 0, MPI_COMM_WORLD);
		}
		
		free(tree_counts);
		free(tree_displs);
	}
	else {
		// Worker process: receive data and train trees
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		// Receive sample size and training data
		MPI_Recv(&sample_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// Allocate memory for training data
		train_data = (float *)malloc(sample_size * num_columns * sizeof(float));
		if (!train_data) {
			perror("Malloc failed for train_data");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		// Receive training data
		MPI_Recv(train_data, sample_size * num_columns, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// Allocate memory for targets
		targets = (int *)malloc(test_size * sizeof(int));
		if (!targets) {
			perror("Malloc failed for targets");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Receive targets and test data
		MPI_Bcast(targets, test_size, MPI_INT, 0, MPI_COMM_WORLD);

		// Allocate memory for test data
		test_data = (float *)malloc(test_size * num_columns * sizeof(float));
		if (!test_data) {
			perror("Malloc failed for test_data");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		// Receive test data from process 0
		MPI_Bcast(test_data, test_size * num_columns, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// Receive number of trees assigned to this process
		int num_trees_assigned;
		MPI_Recv(&num_trees_assigned, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (num_trees_assigned > 0) {
			// Create an array to store the trained trees
			Tree *trees = (Tree *)malloc(num_trees_assigned * sizeof(Tree));
			if (!trees) {
				perror("Malloc failed for trees array");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			
			// Start training timing
			train_start = MPI_Wtime();
			
			// Train each tree and store it directly in the array
			for (int t = 0; t < num_trees_assigned; t++) {
				// Train the tree and store it directly in the array at position t
				train_tree_1d(&trees[t], train_data, sample_size, num_columns, num_classes,
						   max_depth, min_samples_split, max_features, thread_count);
			}
			
			// End training timing
			train_end = MPI_Wtime();
			
			// Start inference timing
			infer_start = MPI_Wtime();
			
			// Process all trees to make predictions (we don't store results, just time it)
			for (int t = 0; t < num_trees_assigned; t++) {
				int *tree_preds = tree_inference_1d(&trees[t], test_data, test_size, num_columns);
				if (!tree_preds) {
					perror("tree_inference_1d returned NULL");
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
				// We don't need to store predictions, just free them
				free(tree_preds);
			}
			
			// End inference timing
			infer_end = MPI_Wtime();
			
			// Calculate times
			train_time = train_end - train_start;
			inference_time = infer_end - infer_start;
			total_time = train_time + inference_time;

			// Free the trees when done
			for (int t = 0; t < num_trees_assigned; t++) {
				destroy_tree(&trees[t]);
			}
			free(trees);
		}
		
		// Free worker process memory
		free(train_data);
		free(test_data);
		free(targets);
	}
	
	// Barrier to ensure all processes are done before collecting timing results
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Collect timing results from all worker processes and report maximum times
	double global_max_train_time, global_max_inference_time, global_max_total_time;
	
	// Use MPI_Reduce to find maximum times across all processes
	double local_train_time = (rank == 0) ? 0.0 : train_time;
	double local_inference_time = (rank == 0) ? 0.0 : inference_time;
	double local_total_time = (rank == 0) ? 0.0 : total_time;
	
	MPI_Reduce(&local_train_time, &global_max_train_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&local_inference_time, &global_max_inference_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&local_total_time, &global_max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	// Process 0 reports the maximum times and cleans up master process memory
	if (rank == 0) {
		printf("\n=== MAXIMUM TIMING RESULTS ACROSS ALL WORKER PROCESSES ===\n");
		printf("Maximum training time: %.6f seconds\n", global_max_train_time);
		printf("Maximum inference time: %.6f seconds\n", global_max_inference_time);
		printf("Maximum total time: %.6f seconds\n", global_max_total_time);
		printf("============================================================\n");
		printf("trainsize = %d, train_tree_prop = %f, num columns = %d\n", train_size, train_tree_proportion, num_columns);

		char csv_store_time_metrics_path[256];
		snprintf(csv_store_time_metrics_path, sizeof(csv_store_time_metrics_path),
				 "output/store_time_metrics_%d_processes_%d_num_threads.csv", process_number, thread_count);
		int tree_data_size = (int)(train_size * train_tree_proportion) * num_columns;
		store_run_params_processes_threads(csv_store_time_metrics_path, global_max_train_time, global_max_inference_time, num_trees, tree_data_size, process_number, thread_count);
		
		// Free allocated memory for master process
		free(data);
		// Don't free train_data and test_data if they were allocated by stratified_split
		// The stratified_split function should handle its own memory management
		// If you're sure these were allocated with malloc, then uncomment:
		// free(train_data);
		// free(test_data);
		free(targets);
	}
	
	MPI_Finalize();
	return 0;
}
