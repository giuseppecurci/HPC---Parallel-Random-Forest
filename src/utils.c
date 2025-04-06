#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../headers/utils.h"

void print_matrix(float **data, int num_rows, int num_columns, int max_rows) {
    // If max_rows is -1 or greater than num_rows, print all rows
    if (max_rows == -1 || max_rows > num_rows) {
        max_rows = num_rows;
    }

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

    // Print the matrix with separators
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < num_columns; j++) {
            printf("%6.4f ", data[i][j]);
            if (j < num_columns - 1) {
                printf("| ");
            }
        }
        printf("\n");

        // Print horizontal separator (except after the last printed row)
        if (i < max_rows - 1) {
            for (int j = 0; j < num_columns; j++) {
                printf("--------");
            }
            printf("\n");
        }
    }
}

void print_array(float *arr, int size, int max_elements) {
    // If max_elements is -1 or greater than size, print all elements
    if (max_elements == -1 || max_elements > size) {
        max_elements = size;
    }

    for (int i = 0; i < max_elements; i++) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}

// Function to read CSV and return the data matrix
float** read_csv(const char *filename, int *num_rows, int *num_columns) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    char line[MAX_LINE];
    int row = 0;
    *num_columns = 0;

    // First, determine the number of columns by reading the first line
    if (fgets(line, sizeof(line), file)) {
        // Count the columns based on the number of commas
        char *token = strtok(line, ",");
        while (token) {
            (*num_columns)++;
            token = strtok(NULL, ",");
        }
    }

    // Go back to the start of the file to read the actual data
    rewind(file);

    // Skip the header row
    fgets(line, sizeof(line), file);

    // Dynamically allocate memory for the data matrix
    float **data = (float**)malloc(sizeof(float*) * MAX_ROWS);  
    for (int i = 0; i < MAX_ROWS; i++) {
        data[i] = (float*)malloc(sizeof(float) * (*num_columns));
    }

    // Read data row by row
    while (fgets(line, sizeof(line), file)) {
        char *token;
        int col = 0;

        token = strtok(line, ",");
        while (token) {
            // Convert token to a float and store it in the data array
            data[row][col] = atof(token); // Convert string to float
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    *num_rows = row;  // Set the actual number of rows read

    fclose(file);
    return data;
}