#include <stdio.h>
#include <stdlib.h>
#include "headers/read_csv.h"
#include "headers/merge_sort.h"

int main() {
    const char *filename = "data/classification_dataset.csv";  // Replace with your actual CSV file path
    int num_rows, num_columns;

    // Call read_csv to get the matrix
    float **data = read_csv(filename, &num_rows, &num_columns);
    if (data == NULL) {
        return 1;  // If there was an error reading the file
    }

    // Optionally, print the matrix to verify the data
    printf("Loaded data:\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_columns; j++) {
            printf("%.4f ", data[i][j]);
        }
        printf("\n");
    }

	// Extract feature column for testing merge sort
    printf("Unsorted array:\n");
    int feature_column = 1; // Which feature to sort
    double *unsorted_array = malloc(num_rows * sizeof(double));
    for (int i = 0; i < num_rows; i++) {
        unsorted_array[i] = data[i][feature_column];
        printf("%.4f ", unsorted_array[i]);
    }
    printf("\n");
    
    // Sort the array
    merge_sort(unsorted_array, num_rows);
    
    // Print sorted array
    printf("Sorted array:\n");
    print_array(unsorted_array, num_rows);	

    // Free the dynamically allocated memory
    for (int i = 0; i < num_rows; i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}
