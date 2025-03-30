#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "headers/read_csv.h"
#include "headers/merge_sort.h"
#include "headers/utils.h"
#include "headers/metrics.h"

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

    // Find the best split
    //BestSplit best_split = find_best_split(data, num_rows, num_columns, num_classes);
    //printf("Best entropy: %.6f, Best split: %.6f (Feature: %d)\n", 
    //       best_split.entropy, best_split.threshold, best_split.feature_index);

    int* predictions = (int *)malloc(num_rows * sizeof(int));
    int* targets = (int *)malloc(num_rows * sizeof(int));
    for (int i = 0; i < num_rows; i++)
    {
        predictions[i] = i % num_classes;
        targets[i] = data[i][num_columns - 1];
    }
    float *acc = accuracy(predictions, targets, num_rows, num_classes);
    float **pr = precision_recall(predictions, targets, num_rows, num_classes);
    for (int i = 0; i < num_classes; i++)
    {
        printf("Accuracy for class %d: %.6f\n", i, acc[i]);
        printf("Precision for class %d: %.6f\n", i, pr[0][i]);
        printf("Recall for class %d: %.6f\n", i, pr[1][i]);
        printf("*********************\n");
    }

    // Free allocated memory
    for (int i = 0; i < num_rows; i++) free(data[i]);
    free(acc);
    for (int i = 0; i < 2; i++) free(pr[i]);
    return 0;
}