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
#include <libgen.h>
#include <string.h>

#include "headers/utils.h"
#include "headers/metrics.h"
#include "headers/forest.h"
#include "headers/memory_ser.h"
#include "headers/tree/tree.h"
#include "headers/tree/utils.h"
#include "headers/tree/train_utils.h"
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
    int num_rows, num_columns;
    float *data = NULL;
    float *train_data = NULL, *test_data = NULL;
    int *targets = NULL;
    int train_size, test_size;
    int sample_size, mode;
    int n_threads = 1;

    // Variables for timing
    double train_start, train_end;
    double infer_start, infer_end;
    double global_start, global_end;
    double train_time = 0.0, inference_time = 0.0, total_time = 0.0;

    // Parse command-line arguments
    int parse_result = parse_arguments(argc, argv, &max_matrix_rows_print, &num_classes, &num_trees,
                                        &max_depth, &min_samples_split, &max_features,
                                        &trained_forest_path, &store_predictions_path,
                                        &store_metrics_path, &new_forest_path, &dataset_path,
                                        &train_proportion, &train_tree_proportion, &seed, &n_threads);
    if (parse_result != 0) {
        printf("Error parsing arguments. Please check the command line options.\n");
        return 1;
    }

    int rank, process_number, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &process_number);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Variables that all processes will need
    int num_trees_assigned = 0;
    float *my_train_data = NULL;
    int my_sample_size = 0;

    // Process 0 reads the dataset and determines basic parameters
    if (rank == 0) {
        printf("Process 0 starting with %d total processes\n", process_number);
        fflush(stdout);
        
        check_dir_existence(new_forest_path);
        if (trained_forest_path == NULL) {
            mode = 0;
        } else {
            mode = 1;
        }

        printf("Process 0: Reading dataset from %s\n", dataset_path);
        fflush(stdout);
        
        data = read_csv(dataset_path, &num_rows, &num_columns);
        if (data == NULL) {
            fprintf(stderr, "Process 0: Failed to read CSV data\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        
        printf("Process 0: Dataset loaded - %d rows, %d columns\n", num_rows, num_columns);
        fflush(stdout);
        
        // Determine number of classes if not specified
        if (num_classes <= 0) {
            for (int i = 0; i < num_rows; i++) {
                int label = (int)data[i * num_columns + (num_columns - 1)];
                if (label > num_classes) {
                    num_classes = label;
                }
            }
            num_classes++;
        }

        printf("Process 0: Number of classes determined: %d\n", num_classes);
        fflush(stdout);

        summary(dataset_path, train_proportion, train_tree_proportion, 0, num_columns - 1, num_classes,
                num_trees, max_depth, min_samples_split, max_features, store_predictions_path,
                store_metrics_path, new_forest_path, trained_forest_path, seed);
    }
	
	global_start = MPI_Wtime();

    // Broadcast dataset dimensions and parameters to all processes
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All non-root processes allocate memory for the dataset
    if (rank != 0) {
        data = (float *)malloc(num_rows * num_columns * sizeof(float));
        if (data == NULL) {
            fprintf(stderr, "Process %d: Failed to allocate memory for dataset\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("Process %d: Allocated memory for dataset - %d rows, %d columns\n", 
               rank, num_rows, num_columns);
        fflush(stdout);
    }

    // Broadcast the entire dataset to all processes
    printf("Process %d: Broadcasting dataset...\n", rank);
    fflush(stdout);
    
    MPI_Bcast(data, num_rows * num_columns, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    printf("Process %d: Dataset broadcast complete\n", rank);
    fflush(stdout);
    
    stratified_split(data, num_rows, num_columns, num_classes, train_proportion,
                    &train_data, &train_size, &test_data, &test_size, seed);
                    
    // Free the original dataset as it's no longer needed
    free(data);
    data = NULL;

	// Only process 0 extracts targets from test data
    if (rank == 0) {
        targets = (int *)malloc(test_size * sizeof(int));
        if (!targets) {
            fprintf(stderr, "Process 0: Malloc failed for targets\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        for (int i = 0; i < test_size; i++) {
            targets[i] = (int)test_data[i * num_columns + (num_columns - 1)];
        }
        
        printf("Process 0: Extracted targets for evaluation\n");
        fflush(stdout);
    } else {
        targets = NULL;  // Other processes don't need targets
    }

    // Calculate sample size for training
    sample_size = (int)(train_tree_proportion * train_size);


    printf("Process %d: Sample size per tree: %d (%.2f%% of %d)\n", 
           rank, sample_size, train_tree_proportion * 100, train_size);
    fflush(stdout);

    // Calculate tree distribution among ALL processes
    int *tree_counts = (int *)malloc(process_number * sizeof(int));
    int *tree_displs = (int *)malloc(process_number * sizeof(int));
    
    if (tree_counts == NULL || tree_displs == NULL) {
        fprintf(stderr, "Process %d: Failed to allocate memory for tree distribution\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Distribute trees among ALL processes
    distribute_trees(num_trees, process_number, tree_counts, tree_displs);
    num_trees_assigned = tree_counts[rank];

    printf("Process %d: Assigned %d trees to train\n", rank, num_trees_assigned);
    fflush(stdout);

    // Each process samples its own training data if it has trees assigned
    if (num_trees_assigned > 0) {
        my_train_data = (float *)malloc(sample_size * num_columns * sizeof(float));
        if (!my_train_data) {
            fprintf(stderr, "Process %d: Malloc failed for my_train_data\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Each process uses a different seed for sampling to ensure different samples
        my_sample_size = sample_data_without_replacement(
            train_data, train_size, num_columns, train_tree_proportion, my_train_data, seed + rank);
        
        if (my_sample_size <= 0) {
            fprintf(stderr, "Process %d: Error in sampling data (got size: %d)\n", rank, my_sample_size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("Process %d: Sampled %d rows for training\n", rank, my_sample_size);
        fflush(stdout);
    }

    // Free train_data as each process now has its own sample
    if (train_data) {
        free(train_data);
        train_data = NULL;
    }
	// Synchronize all processes before starting computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ALL PROCESSES PERFORM COMPUTATION
    if (mode == 0 && num_trees_assigned > 0) {
        printf("Process %d: Starting training of %d trees\n", rank, num_trees_assigned);
        fflush(stdout);
        
        // Allocate array to store trained trees
        Tree *trees = (Tree *)malloc(num_trees_assigned * sizeof(Tree));
        if (!trees) {
            fprintf(stderr, "Process %d: Malloc failed for trees array\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Start training timing
        train_start = MPI_Wtime();
        
        for (int t = 0; t < num_trees_assigned; t++) {
            printf("Process %d: Training tree %d/%d\n", rank, t+1, num_trees_assigned);
            printf("====================================================\n");
            fflush(stdout);
            double tree_start = MPI_Wtime();
            
            train_tree_1d(&trees[t], my_train_data, my_sample_size, num_columns, num_classes,
                         max_depth, min_samples_split, max_features, n_threads);
            
            double tree_end = MPI_Wtime();
            printf("Process %d: Finished tree %d/%d in %.4f seconds\n", 
                   rank, t+1, num_trees_assigned, tree_end - tree_start);
            fflush(stdout);
        }

        train_end = MPI_Wtime();
		
		printf("Process %d: Finished all the training\n", rank);
		fflush(stdout);
        
        // Free training data after training is complete
        if (my_train_data) {
			printf("Process %d: About to free my train data \n", rank);
			fflush(stdout);
            free(my_train_data);
            my_train_data = NULL;
        }
        
        // Start inference timing
        infer_start = MPI_Wtime();
        
        // Allocate space for predictions from all trees
        int *local_predictions = (int *)malloc(num_trees_assigned * test_size * sizeof(int));
        if (!local_predictions) {
            fprintf(stderr, "Process %d: Malloc failed for local_predictions\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Make predictions with all trees
        for (int t = 0; t < num_trees_assigned; t++) {
            int *tree_preds = tree_inference_1d(&trees[t], test_data, test_size, num_columns);
            if (!tree_preds) {
                fprintf(stderr, "Process %d: tree_inference_1d returned NULL for tree %d\n", rank, t);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Copy predictions for this tree
            for (int i = 0; i < test_size; i++) {
                local_predictions[t * test_size + i] = tree_preds[i];
            }
            
            // Free the predictions for this tree
            free(tree_preds);
        }
        
        infer_end = MPI_Wtime();
        
        // Calculate times
        train_time = train_end - train_start;
        inference_time = infer_end - infer_start;
        
        printf("Process %d: Completed - train_time: %.6f, inference_time: %.6f, total_time: %.6f\n", 
               rank, train_time, inference_time, total_time);
        fflush(stdout);
        
		if (rank == 0) {
				// Process 0 collects all predictions
				printf("Process 0: Collecting predictions from all processes\n");
				fflush(stdout);
				
				// Use the existing tree_counts and tree_displs arrays
				// (no need to recreate them)
				
				// Allocate memory for all predictions
				int **all_predictions = (int **)malloc(process_number * sizeof(int *));
				if (!all_predictions) {
					fprintf(stderr, "Process 0: Failed to allocate all_predictions array\n");
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
				
				// Process 0's own predictions
				all_predictions[0] = local_predictions;
				
				// Receive predictions from other processes
				for (int p = 1; p < process_number; p++) {
					if (tree_counts[p] > 0) {
						all_predictions[p] = (int *)malloc(tree_counts[p] * test_size * sizeof(int));
						if (!all_predictions[p]) {
							fprintf(stderr, "Process 0: Failed to allocate predictions array for process %d\n", p);
							MPI_Abort(MPI_COMM_WORLD, 1);
						}
						
						printf("Process 0: Receiving %d predictions from process %d\n", 
							   tree_counts[p] * test_size, p);
						fflush(stdout);
						
						MPI_Recv(all_predictions[p], tree_counts[p] * test_size, MPI_INT, p, 0, 
								 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					} else {
						all_predictions[p] = NULL;  // No predictions from this process
					}
				}
				
				printf("Process 0: All predictions collected, starting aggregation\n");
				fflush(stdout);
				
				// Aggregate predictions and save results
				aggregate_and_save_predictions(process_number, test_size, num_classes,
											 all_predictions, tree_counts, targets,
											 store_predictions_path, store_metrics_path, rank);
				
				printf("Process 0: All predictions aggregated, results available\n");
				fflush(stdout);
				
				global_end = MPI_Wtime(); 
				
				// Clean up prediction arrays
				for (int p = 1; p < process_number; p++) {
					if (all_predictions[p]) {
						free(all_predictions[p]);
					}
				}
				free(all_predictions);
				
				printf("Process 0: Prediction aggregation and saving completed\n");
				fflush(stdout);
				
			} else {
				// Other processes send their predictions to process 0
				printf("Process %d: Sending %d predictions to process 0\n", 
					   rank, num_trees_assigned * test_size);
				fflush(stdout);
				
				MPI_Send(local_predictions, num_trees_assigned * test_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
				
				printf("Process %d: Predictions sent to process 0\n", rank);
				fflush(stdout);
				
			}

        // Clean up memory
        free(local_predictions);
        
        // Free memory for all trained trees
		printf("Process %d: about to destroy all my trees", rank);
        fflush(stdout);
		
//        for (int t = 0; t < num_trees_assigned; t++) {
//			printf("Process %d: about to destroy %d my tree", rank, t);
//          destroy_tree(&trees[t]);
//			printf("Process %d: destroyed my %d tree", rank, t);
//        	fflush(stdout);
//        }
		printf("Process %d: about to destroy my tree array", rank);

        free(trees);
		printf("Process %d: destroyed my trees array", rank);
        

	} else if (num_trees_assigned == 0) {
        printf("Process %d: No trees assigned, skipping computation\n", rank);
        fflush(stdout);
        
        // Initialize timing variables for processes with no trees
        train_time = 0.0;
        inference_time = 0.0;
        total_time = 0.0;
    }
    
    // Clean up test data and targets for ALL processes
    if (test_data) {
        free(test_data);
        test_data = NULL;
    }
    if (targets) {
        free(targets);
        targets = NULL;
    }
    
    // Collect timing results from all processes
    double global_max_train_time = 0.0, global_max_inference_time = 0.0, global_max_total_time = 0.0;
    MPI_Reduce(&train_time, &global_max_train_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&inference_time, &global_max_inference_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     
	// Process 0 reports the maximum times and cleans up
    if (rank == 0) {
		global_max_total_time = global_end - global_start; 
        printf("\n=== MAXIMUM TIMING RESULTS ACROSS ALL PROCESSES ===\n");
        printf("Maximum training time: %.6f seconds\n", global_max_train_time);
        printf("Maximum inference time: %.6f seconds\n", global_max_inference_time);
        printf("Maximum total time: %.6f seconds\n", global_max_total_time);
        printf("====================================================\n");
        printf("trainsize = %d, train_tree_prop = %f, num columns = %d\n", train_size, train_tree_proportion, num_columns);
        fflush(stdout);
        
        // Store timing metrics
        char dataset_copy[256];
        strncpy(dataset_copy, dataset_path, sizeof(dataset_copy));
        dataset_copy[sizeof(dataset_copy) - 1] = '\0';

        char *base = basename(dataset_copy);
        char *dot = strrchr(base, '.');
        if (dot != NULL) {
           *dot = '\0';
        }
        char csv_store_time_metrics_path[512];
        snprintf(csv_store_time_metrics_path, sizeof(csv_store_time_metrics_path),
             "output/store_time_metrics_%s_%d_processes_%d_threads.csv", base, process_number, n_threads);
        int tree_data_size = (int)(train_size * train_tree_proportion) * num_columns;
        store_run_params_processes_threads(csv_store_time_metrics_path, global_max_train_time, global_max_inference_time, global_max_total_time,num_trees, tree_data_size, process_number, n_threads);
        
    }
	free(tree_counts);
	free(tree_displs);
    
    MPI_Finalize();
    return 0;
}

