// MPI Implementation 
#include <stdio.h>
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

int main(int argc, char *argv[]) {
    int max_matrix_rows_print = 0; // Default: print nothing
    int num_classes = -1; 
    char *new_forest_path = "output/model"; 
    char *trained_forest_path = "output/model"; 
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
	int sample_size, mode;

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

	if (rank == 0) {
		printf("Hello from rank %d", rank);
		check_dir_existence(new_forest_path);
		mode = check_bin_files_exist(new_forest_path);

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

    	stratified_split(data, num_rows, num_columns, num_classes, train_proportion,
						&train_data, &train_size, &test_data, &test_size, seed);
		if (test_data == NULL) {
			printf("test_data is NULL after stratified_split. Aborting.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}	
		// Broadcast dimensions first
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		printf("broadcasted dimensions\n");
		// input sample size 0.8
		sample_size = (int)(0.8 * train_size);
		// Sample 75% of the train data and send it over
		for (int p = 1; p < process_number; p++) {
			// Allocate sampled_data buffer
			float *sampled_data = (float *)malloc(sample_size * num_columns * sizeof(float));
			if (sampled_data == NULL) {
				fprintf(stderr, "Failed to allocate memory for sampled data\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			
			// Call sampling function
			int actual_sample_size = sample_data_without_replacement(
				train_data, train_size, num_columns, 0.8, sampled_data);
				
			printf("Sampled data size, about to send = %d\n", actual_sample_size);
			if (actual_sample_size <= 0) {
				fprintf(stderr, "Error in sampling data for process %d\n", p);
				free(sampled_data);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			
			// Send the data size first
			MPI_Send(&actual_sample_size, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
			printf("Sent the sampled data size");
			
			// Send the sampled data
			MPI_Send(sampled_data, actual_sample_size * num_columns, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
			
			// Free the buffer
			free(sampled_data);
		}
		printf("BROADCASTED train datasets\n");

		targets = (int *)malloc(test_size * sizeof(int));
		for (int i = 0; i < test_size; i++) {
			targets[i] = (int)test_data[i * num_columns + (num_columns - 1)];
		}

		MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(targets, test_size, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(test_data, test_size * num_columns, MPI_FLOAT, 0, MPI_COMM_WORLD);
    	summary(dataset_path, train_proportion, train_size, num_columns - 1, num_classes,
            num_trees, max_depth, min_samples_split, max_features, store_predictions_path, 
            store_metrics_path, new_forest_path, trained_forest_path, seed);

		// BASED ON MODE WE DO DIFFERENT STUFF
		// IDEA: processo 0 legge tutti gli alberi e ha un puntatore per albero, iterare sui suoi alberi e mandare la proporzione giusta
		MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (mode == 1) {
			printf("Found .bin files in directory: %s\n", new_forest_path);
			
			// Create and deserialize the forest
			Forest *random_forest = (Forest *)malloc(sizeof(Forest));
			if (!random_forest) {
				perror("malloc failed for random_forest");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			
			create_forest(random_forest, num_trees, max_depth, min_samples_split, max_features);
			
			deserialize_forest(random_forest, trained_forest_path);
			
			// Calculate distribution of trees across worker processes (ranks 1 to process_number-1)
			int worker_count = process_number - 1;
			int *tree_counts = (int *)malloc(worker_count * sizeof(int));
			int *tree_displs = (int *)malloc(worker_count * sizeof(int));
			
			// Distribute trees among worker processes only
			distribute_trees(random_forest->num_trees, worker_count, tree_counts, tree_displs);
			
			// Print tree distribution for debugging
			printf("Distributing trees: ");
			for (int p = 0; p < worker_count; p++) {
				printf("Process %d: %d trees, ", p+1, tree_counts[p]);
			}
			printf("\n");
			
			// Send trees to worker processes
			for (int p = 0; p < worker_count; p++) {
				int target_rank = p + 1; // Worker process rank
				
				if (tree_counts[p] > 0) {
					// Send number of trees to each process
					MPI_Send(&tree_counts[p], 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
					
					// Send each tree individually
					for (int t = 0; t < tree_counts[p]; t++) {
						int tree_idx = tree_displs[p] + t;
						void *buffer;
						int buffer_size;
						
						// Serialize the tree
						serialize_tree_to_buffer(&random_forest->trees[tree_idx], &buffer, &buffer_size);
						
						// Send buffer size first
						MPI_Send(&buffer_size, 1, MPI_INT, target_rank, 1, MPI_COMM_WORLD);
						
						// Then send the buffer
						MPI_Send(buffer, buffer_size, MPI_BYTE, target_rank, 2, MPI_COMM_WORLD);
						
						// Free the buffer
						free(buffer);
					}
				}
			}
			if (trained_forest_path != new_forest_path) {
				free(trained_forest_path);
			}
			// Allocate space for collecting predictions from all workers
			int **all_predictions = (int **)malloc((process_number - 1) * sizeof(int *));
			for (int p = 0; p < process_number - 1; p++) {
				all_predictions[p] = (int *)malloc(tree_counts[p] * test_size * sizeof(int));
				if (!all_predictions[p]) {
					fprintf(stderr, "Failed to allocate memory for all_predictions[%d]\n", p);
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
			}

			// Receive predictions from each process
			for (int p = 1; p < process_number; p++) {
				int trees_from_p = tree_counts[p - 1];  // offset since tree_counts[0] is for rank 1

				if (trees_from_p > 0) {
					MPI_Recv(all_predictions[p - 1], trees_from_p * test_size, MPI_INT,
							 p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					printf("Process 0: Received predictions from process %d\n", p);
				}
			}
			aggregate_and_save_predictions(process_number, test_size, num_classes,
                               all_predictions, targets,
                               store_predictions_path, store_metrics_path, rank);
			free(tree_counts);
			free(tree_displs);
			free_forest(random_forest);

    	}	
		else {
			printf("Mode == 0");
		}
	}
	else {
		printf("PRINT CIAO DA PROCESSO %d", rank);
		MPI_Bcast(&test_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Recv(&sample_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Received test_size = %d, num_columns = %d, sample_size = %d", test_size, num_columns, sample_size);
		// Allocate memory
		train_data = (float *)malloc(sample_size * num_columns * sizeof(float));
		if (!train_data) {
			perror("Malloc failed for train_data");
			MPI_Abort(MPI_COMM_WORLD, 1);}
		// Receive train_data
		MPI_Recv(train_data, sample_size * num_columns, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		printf("\n");
		printf("Num_rows = %d, Num_columns = %d, myrank = %d", sample_size, num_columns, rank);
		printf("\n");
		
		// Allocate memory for targets
		targets = (int *)malloc(test_size * sizeof(int));
		if (!targets) {
			perror("Malloc failed for targets");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(targets, test_size, MPI_INT, 0, MPI_COMM_WORLD);

		// Allocate memory for test data
		test_data = (float *)malloc(test_size * num_columns * sizeof(float));
		if (!test_data) {
			perror("Malloc failed for test_data");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		// Receive test data from process 0
		MPI_Bcast(test_data, test_size * num_columns, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		// receive mode
		MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (mode == 1) {
			// Worker process code for mode == 1 (existing trees)
			// Receive number of trees assigned to this process
			// Fixed worker process code for mode == 1 (existing trees)
			// Receive number of trees assigned to this process
			int num_trees_assigned;
			MPI_Recv(&num_trees_assigned, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("Process %d: Assigned %d trees to process\n", rank, num_trees_assigned);

			// Create an array to point at trees and store the received trees
			Tree *trees = NULL;
			if (num_trees_assigned > 0) {
				trees = (Tree *)malloc(num_trees_assigned * sizeof(Tree));
				if (!trees) {
					perror("Malloc failed for trees array");
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
				
				// Receive each tree
				for (int t = 0; t < num_trees_assigned; t++) {
					// Receive buffer size first
					int buffer_size;
					MPI_Recv(&buffer_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					
					// Allocate buffer and receive serialized tree
					void *buffer = malloc(buffer_size);
					if (!buffer) {
						perror("Malloc failed for tree buffer");
						MPI_Abort(MPI_COMM_WORLD, 1);
					}
					
					MPI_Recv(buffer, buffer_size, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					
					// Deserialize the tree
					deserialize_tree_from_buffer(buffer, &trees[t]);
					free(buffer);
					
					printf("Process %d: Received tree %d of %d\n", rank, t+1, num_trees_assigned);
				}
				
				// MOVED OUTSIDE: Allocate space for predictions after all trees are received
				int *local_predictions = (int *)malloc(num_trees_assigned * test_size * sizeof(int));
				if (!local_predictions) {
					perror("Malloc failed for local_predictions");
					MPI_Abort(MPI_COMM_WORLD, 1);
				}

				// Process all trees AFTER they are all received
				for (int t = 0; t < num_trees_assigned; t++) {
					int *tree_preds = tree_inference_1d(&trees[t], test_data, test_size, num_columns);
					if (!tree_preds) {
						perror("tree_inference_1d returned NULL");
						MPI_Abort(MPI_COMM_WORLD, 1);
					}
					
					for (int i = 0; i < test_size; i++) {
						local_predictions[t * test_size + i] = tree_preds[i];
					}
					free(tree_preds);  // Free memory returned by tree_inference_1d
				}
				
				// Send the predictions to process 0
				MPI_Send(local_predictions, num_trees_assigned * test_size, MPI_INT, 0, 3, MPI_COMM_WORLD);
				printf("Process %d: Sent predictions to process 0\n", rank);

				// Free memory - MOVED OUTSIDE both loops
				free(local_predictions);
				
				// Free each tree before freeing the array
				for (int t = 0; t < num_trees_assigned; t++) {
					// Assuming there's a function to free tree resources - call it here
					// free_tree(&trees[t]);
				}
				free(trees);
			}
		}
		else {
			// NO TREES IN DIRECTORY
			printf("haha");
		}
	}

	MPI_Finalize();
	return 0;
}
