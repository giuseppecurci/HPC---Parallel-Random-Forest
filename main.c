#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "headers/read_csv.h"
#include "headers/merge_sort.h"
#include "headers/utils.h"

int main(int argc, char *argv[]) {
    const char *filename = "data/classification_dataset.csv";  // Replace with your actual CSV file path
    int num_rows, num_columns;
    int max_matrix_rows_print = 0, max_array_elements_print = 0;  // Default: print nothing
    int num_classes = 0;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print_matrix") == 0 && i + 1 < argc) {
            max_matrix_rows_print = atoi(argv[i + 1]);  // Convert to integer
        } else if (strcmp(argv[i], "--print_sorted_array") == 0 && i + 1 < argc) {
            max_array_elements_print = atoi(argv[i + 1]);
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

    float *target_array = malloc(num_rows * sizeof(float));
    for (int i = 0; i < num_rows; i++) {
        target_array[i] = data[i][20];
    }

    // Extract feature column for sorting
    int feature_column = 1;  // Which feature to sort
    float *unsorted_array = malloc(num_rows * sizeof(float));
    for (int i = 0; i < num_rows; i++) {
        unsorted_array[i] = data[i][feature_column];
    }
    
    // Sort the array
    merge_sort(unsorted_array, num_rows);
    
    // Print sorted array
    if (max_array_elements_print != 0) {
        printf("Sorted array, feature %d:\n", feature_column);
        print_array(unsorted_array, num_rows, max_array_elements_print);
    }

    // Calculate split sizes
    //float left_split[] = {0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //float right_split[] = {0.0, 2.0, 0.0, 0.0, 0.0};
    //int left_size = sizeof(left_split) / sizeof(left_split[0]);
    //int right_size = sizeof(right_split) / sizeof(right_split[0]);

    //float entropy = get_entropy(left_split, right_split, left_size, right_size, num_classes);
    //printf("Entropy: %.6f\n", entropy);
    //float test_sorted_array[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    //float test_target_array[] = {0.0, 0.0, 0.0, 1.0, 1.0};
    
    float *best_split = get_best_split_num_var(unsorted_array, target_array, num_rows, num_classes);
    printf("Best entropy: %.6f, Best split: %.6f\n", best_split[0], best_split[1]);
    // Free memory
    printf("Freeing memory\n");
    free(unsorted_array);
    for (int i = 0; i < num_rows; i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}