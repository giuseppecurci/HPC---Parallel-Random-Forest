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
#include "headers/utils.h"
#include "headers/metrics.h"

#include "headers/tree/tree.h"
#include "headers/tree/utils.h"
#include "headers/tree/train_utils.h"

int main(int argc, char *argv[]) {
    const char *filename = "data/classification_dataset.csv";  
    int num_rows, num_columns;
    int max_matrix_rows_print = 0; // Default: print nothing
    int num_classes = 0;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print_matrix") == 0 && i + 1 < argc) {
            max_matrix_rows_print = atoi(argv[i + 1]);  // Convert to integer
        }
        else if (strcmp(argv[i], "--num_classes") == 0 && i + 1 < argc) {
            num_classes = atoi(argv[i + 1]);
        }
    }

    if (num_classes <= 0) {
        printf("Number of classes must be a positive integer. Use --num_classes argument.\n");
        return 1;
    }

    printf("Number of classes: %d\n", num_classes);

    float **data = read_csv(filename, &num_rows, &num_columns);
    if (data == NULL) {
        return 1;  
    }
    printf("Loaded data\n");

    // Optionally, print the matrix
    if (max_matrix_rows_print != 0) {  
        print_matrix(data, num_rows, num_columns, max_matrix_rows_print);
    }

    Tree *random_tree = (Tree *)malloc(sizeof(Tree));
    train_tree(random_tree, data, num_rows, num_columns, num_classes);
    //Tree *random_tree = deserialize_tree("random_tree.bin");
    int *predictions;
    int* targets = (int *)malloc(num_rows * sizeof(int));
    printf("Getting targets\n");
    for (int i = 0; i < num_rows; i++) {
        targets[i] = (int)data[i][num_columns - 1];
    }
    printf("Starting inference:\n");
    predictions = tree_inference(random_tree, data, num_rows);
    save_predictions(predictions, num_rows, "predictions.csv");
    printf("Inference completed\n");
    compute_metrics(predictions, targets, num_rows, num_classes);
    serialize_tree(random_tree, "random_tree.bin");
    destroy_tree(random_tree);
    
    // Free allocated memory
    for (int i = 0; i < MAX_ROWS; i++) free(data[i]);
    free(data);
    free(predictions);
    free(targets);
    printf("Memory freed\n");
    return 0;
}