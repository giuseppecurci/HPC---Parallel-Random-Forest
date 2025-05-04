/**
 * @file main.c
 * @brief Main program for training a decision tree on a classification dataset and evaluating its performance.
 *
 * This file provides the main function that:
 * - Parses command-line arguments.
 * - Loads a dataset from a CSV file.
 * - Trains a decision tree classifier using the loaded dataset.
 * - Makes predictions using the trained tree.
 * - Computes and saves performance metrics such as accuracy, precision, and recall.
 * - Serializes the trained tree for future use.
 *
 * The program can be run with the following command-line arguments:
 * - `--print_matrix <num_rows>`: Print the first `num_rows` rows of the dataset.
 * - `--num_classes <num_classes>`: Specify the number of classes in the dataset.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

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
    float train_tree_proportion = 0.75; // Default: 75% of the training data
    int num_trees = 10;
    char* max_features = "sqrt";
    int min_samples_split = 2;
    int max_depth = 10;
    int seed = 0;
    
    char *dataset_path = "../data/classification_dataset.csv";  
    int num_rows, num_columns;
    float **train_data, **test_data;
    int train_size, test_size;

    struct timeval start_time, end_time;
    double train_time, inference_time;
    
    // Parse command-line arguments
    int parse_result = parse_arguments(argc, argv, &max_matrix_rows_print, &num_classes, &num_trees,
                                        &max_depth, &min_samples_split, &max_features,
                                        &trained_forest_path, &store_predictions_path,
                                        &store_metrics_path, &new_forest_path, &dataset_path,
                                        &train_proportion, &train_tree_proportion, &seed);
    if (parse_result != 0) {
        printf("Error parsing arguments. Please check the command line options.\n");
        return 1;
    }

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

    float **data = read_csv(dataset_path, &num_rows, &num_columns);

    if (num_classes <= 0) {
        printf("Inferring number of classes from the dataset...\n");
        for (int i = 0; i < num_rows; i++) {
            if (data[i][num_columns - 1] > num_classes) num_classes = (int)data[i][num_columns - 1];
        }
        num_classes++;
    }

    stratified_split(data, num_rows, num_columns, num_classes, train_proportion, &train_data, &train_size, &test_data, &test_size, seed);
    if (data == NULL) {
        return 1;  
    }
    printf("Loaded data\n--------------\n");
    if (max_matrix_rows_print != 0) {  
        print_matrix(data, num_rows, num_columns, max_matrix_rows_print);
        printf("--------------\n");
    }

    int* targets = (int *)malloc(test_size * sizeof(int));
    for (int i = 0; i < test_size; i++) {
        targets[i] = (int)test_data[i][num_columns - 1];
    }

    int train_tree_size = train_size * train_tree_proportion;

    summary(dataset_path, train_proportion, train_tree_proportion, train_size, num_columns - 1, num_classes,
            num_trees, max_depth, min_samples_split, max_features, store_predictions_path, 
            store_metrics_path, new_forest_path, trained_forest_path, seed);
    
	Forest *random_forest = (Forest *)malloc(sizeof(Forest));
    create_forest(random_forest, num_trees, max_depth, min_samples_split, max_features);
    
    if (trained_forest_path == NULL){
        gettimeofday(&start_time, NULL);
        train_forest(random_forest, train_data, train_size, num_columns, train_tree_size, num_classes, seed);
        gettimeofday(&end_time, NULL);
        train_time = (end_time.tv_sec - start_time.tv_sec) + 
                   (end_time.tv_usec - start_time.tv_usec) / 1e6;
        serialize_forest(random_forest, new_forest_path);
    } else {
        printf("Loading tree from %s\n", trained_forest_path);
        deserialize_forest(random_forest, trained_forest_path);
        if (random_forest == NULL) {
            printf("Failed to load tree from %s\n", trained_forest_path);
            return 1;
        }
    }

    int* predictions;
    gettimeofday(&start_time, NULL);
    predictions = forest_inference(random_forest, test_data, test_size, num_classes);
    gettimeofday(&end_time, NULL);
    inference_time = (end_time.tv_sec - start_time.tv_sec) + 
                   (end_time.tv_usec - start_time.tv_usec) / 1e6;
    save_predictions(predictions, test_size, store_predictions_path);
    compute_metrics(predictions, targets, test_size, num_classes, store_metrics_path);

    printf("\nTime taken to train the forest: %.6f seconds\n", train_time);
    printf("    'time_sampling_data for trees': %.6f seconds\n", total_time_sampling_data);
    printf("    'find_best_split': %.6f seconds\n", total_time_find_best_split);
    printf("        'merge_sort': %.6f seconds\n", total_time_merge_sort);
    printf("        'best_split_num_var': %.6f seconds\n", total_time_best_split_num_var);
    printf("            'split_for_entropy': %.6f seconds\n", total_time_split_for_entropy);
    printf("            'entropy': %.6f seconds\n", total_time_entropy);
    printf("    'split_data': %.6f seconds\n", total_time_split_data);
    printf("Time taken for inference: %.6f seconds\n", inference_time);
    printf("Total time taken: %.6f seconds\n", train_time + inference_time);
    
    // Free allocated memory
    free_forest(random_forest);
    for (int i = 0; i < MAX_ROWS; i++) free(data[i]);
    free(data);
    for (int i = 0; i < train_size; i++) {
        free(train_data[i]);
    }
    free(train_data);

    for (int i = 0; i < test_size; i++) {
        free(test_data[i]);
    }
    free(test_data);
    free(predictions);
    free(targets);
    return 0;
}
