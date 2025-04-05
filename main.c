#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "headers/read_csv.h"
#include "headers/merge_sort.h"
#include "headers/utils.h"
#include "headers/metrics.h"
#include "headers/tree.h"

int main(int argc, char *argv[]) {
    const char *filename = "data/classification_dataset.csv";  // Replace with your actual CSV file path
    int num_rows, num_columns;
    int max_matrix_rows_print = 0, max_array_elements_print = 0;  // Default: print nothing
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

    // Call read_csv to get the matrix
    float **data = read_csv(filename, &num_rows, &num_columns);
    if (data == NULL) {
        return 1;  // If there was an error reading the file
    }
    printf("Loaded data\n");

    // Optionally, print the matrix
    if (max_matrix_rows_print != 0) {  // Only print if not explicitly disabled (0 rows)
        print_matrix(data, num_rows, num_columns, max_matrix_rows_print);
    }

    Tree *random_tree = (Tree *)malloc(sizeof(Tree));
    train_tree(random_tree, data, num_rows, num_columns, num_classes);
    //print_tree(random_tree);
    int *predictions;
    int *targets = malloc(num_rows * sizeof(int));
    printf("Getting targets\n");
    for (int i = 0; i < num_rows; i++) {
        targets[i] = (int)data[i][num_columns - 1];
    }
    printf("Starting inference:\n");
    predictions = tree_inference(random_tree, data, num_rows);
    compute_metrics(predictions, targets, num_rows, num_classes);
    destroy_tree(random_tree);
    // Free allocated memory
    for (int i = 0; i < num_rows; i++) free(data[i]);
    return 0;
}