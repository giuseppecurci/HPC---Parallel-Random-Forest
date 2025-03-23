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

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print_matrix") == 0 && i + 1 < argc) {
            max_matrix_rows_print = atoi(argv[i + 1]);  // Convert to integer
        } else if (strcmp(argv[i], "--print_sorted_array") == 0 && i + 1 < argc) {
            max_array_elements_print = atoi(argv[i + 1]);
        }
    }

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

    // Extract feature column for sorting
    printf("Getting Unsorted array\n");
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
    int left_size = 10, right_size = 5;
    float left_split[] = {0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float right_split[] = {0.0, 0.0, 0.0, 0.0, 0.0};

    float entropy = get_entropy(left_split, right_split, left_size, right_size);
    printf("Entropy: %.6f\n", entropy);

    // Free memory
    printf("Freeing memory\n");
    free(unsorted_array);
    for (int i = 0; i < num_rows; i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}