#include "../headers/utils.h"

void print_matrix(float **data, int num_rows, int num_columns) {
    // Print column headers
    printf("Feature Matrix:\n");
    for (int j = 0; j < num_columns - 1; j++) {
        printf("feat %d | ", j + 1);
    }
    printf("target\n");

    // Print horizontal separator
    for (int j = 0; j < num_columns; j++) {
        printf("--------");
    }
    printf("\n");

    // Print matrix with vertical separators
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_columns; j++) {
            printf("%6.4f ", data[i][j]);  // Adjust width for alignment
            if (j < num_columns - 1) {
                printf("| ");
            }
        }
        printf("\n");

        // Print horizontal separator (except after the last row)
        if (i < num_rows - 1) {
            for (int j = 0; j < num_columns; j++) {
                printf("--------");
            }
            printf("\n");
        }
    }
}


void print_array(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}