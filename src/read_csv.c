#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../headers/read_csv.h"

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
