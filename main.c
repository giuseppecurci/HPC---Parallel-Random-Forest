#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "headers/read_csv.h"
#include "headers/merge_sort.h"
#include "headers/utils.h"

int main(int argc, char *argv[]) {
    const char *filename = "data/classification_dataset.csv";  // Replace with your actual CSV file path
    int num_rows, num_columns;
    int print_matrix_flag = 0, print_sorted_array_flag = 0; // Flags to print the matrix or array for debugging

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print_matrix") == 0) {
            print_matrix_flag = 1;
        } else if (strcmp(argv[i], "--print_sorted_array") == 0) {
            print_sorted_array_flag = 1;
        }
    }

    // Call read_csv to get the matrix
    float **data = read_csv(filename, &num_rows, &num_columns);
    if (data == NULL) {
        return 1;  // If there was an error reading the file
    }
    printf("Loaded data\n");

    // Optionally, print the matrix to verify the data
    if (print_matrix_flag) {
        print_matrix(data, num_rows, num_columns);
    }

	// Extract feature column for testing merge sort
    printf("Getting Unsorted array\n");
    int feature_column = 1; // Which feature to sort
    float *unsorted_array = malloc(num_rows * sizeof(double));
    for (int i = 0; i < num_rows; i++) {
        unsorted_array[i] = data[i][feature_column];
    }
    printf("\n");
    
    // Sort the array
    merge_sort(unsorted_array, num_rows);
    
    // Print sorted array
    if (print_sorted_array_flag) {
        printf("Sorted array, feature %d:\n", feature_column);
        print_array(unsorted_array, num_rows);
    }

    // Free the dynamically allocated memory
    printf("Freeing memory\n");
    free(unsorted_array);
    for (int i = 0; i < num_rows; i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}
