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

#include "headers/utils.h"
#include "headers/metrics.h"

#include "headers/tree/tree.h"
#include "headers/tree/utils.h"
#include "headers/tree/train_utils.h"

int main(int argc, char *argv[]) {
    int max_matrix_rows_print = 0; // Default: print nothing
    int num_classes = -1; 
    char *new_tree_path = "random_tree.bin"; // Default path for the new tree
    char *trained_tree_path = NULL; // Path to the trained tree
    char *store_predictions_path = "predictions.csv"; // Default path for predictions
    char *store_metrics_path = "metrics_output.txt"; // Default path for metrics
    float train_proportion = 0.8;
    int seed = 0;
    
    char *dataset_path = "data/classification_dataset.csv";  
    int num_rows, num_columns;
    float **train_data, **test_data;
    int train_size, test_size;
    
    // Parse command-line arguments
    int parse_result = parse_arguments(argc, argv, &max_matrix_rows_print, &num_classes,
                                        &trained_tree_path, &store_predictions_path,
                                        &store_metrics_path, &new_tree_path, &dataset_path,
                                        &train_proportion, &seed);
    if (parse_result != 0) {
        printf("Error parsing arguments. Please check the command line options.\n");
        return 1;
    }

    float **data = read_csv(dataset_path, &num_rows, &num_columns);
    stratified_split(data, num_rows, num_columns, train_proportion, &train_data, &train_size, &test_data, &test_size, seed);
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

    if (num_classes <= 0) {
        printf("Inferring number of classes from the dataset...\n");
        for (int i = 0; i < test_size; i++) {
            if (targets[i] > num_classes) num_classes = (int)targets[i];
        }
        num_classes++;
    }

    summary(dataset_path, train_proportion, train_size, num_columns,
            num_classes, store_predictions_path, store_metrics_path,
            new_tree_path, trained_tree_path, seed);

    Tree *random_tree = (Tree *)malloc(sizeof(Tree));
    if (trained_tree_path == NULL){
        printf("Starting tree training\n");
        train_tree(random_tree, train_data, train_size, num_columns, num_classes);
        serialize_tree(random_tree, new_tree_path);
    } else {
        printf("Loading tree from %s\n", trained_tree_path);
        random_tree = deserialize_tree(trained_tree_path);
        if (random_tree == NULL) {
            printf("Failed to load tree from %s\n", trained_tree_path);
            return 1;
        }
    }

    int* predictions;
    printf("Starting inference.\n");
    predictions = tree_inference(random_tree, test_data, test_size);
    save_predictions(predictions, test_size, store_predictions_path);
    compute_metrics(predictions, targets, test_size, num_classes, store_metrics_path);
    printf("Inference completed.\n");
    
    // Free allocated memory
    destroy_tree(random_tree);
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