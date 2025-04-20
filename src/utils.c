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

int parse_arguments(int argc, char *argv[], int *max_matrix_rows_print, int *num_classes,
                    char **trained_tree_path, char **store_predictions_path, char **store_metrics_path,
                    char **new_tree_path, char **dataset_path, float *train_proportion) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print_matrix") == 0 && i + 1 < argc) {
            *max_matrix_rows_print = atoi(argv[i + 1]);  // Convert to integer
        }
        else if (strcmp(argv[i], "--num_classes") == 0 && i + 1 < argc) {
            *num_classes = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "--trained_tree_path") == 0 && i + 1 < argc) {
            *trained_tree_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--store_predictions_path") == 0 && i + 1 < argc) {
            *store_predictions_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--store_metrics_path") == 0 && i + 1 < argc) {
            *store_metrics_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--new_tree_path") == 0 && i + 1 < argc) {
            *new_tree_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--dataset_path") == 0 && i + 1 < argc) {
            *dataset_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--train_proportion") == 0 && i + 1 < argc) {
            *train_proportion = atof(argv[i + 1]);
            if (*train_proportion <= 0 || *train_proportion >= 1) {
                printf("Train proportion must be between 0 and 1, instead %f was provided.\n", *train_proportion);
                return 1;  // Return 1 to indicate an error
            } 
        }
    }

    return 0;  // Return 0 if everything is parsed successfully
}

void stratified_split(float **data, int num_rows, int num_columns, float train_proportion,
                      float ***train_data, int *train_size, float ***test_data, int *test_size) {
    // The last column is the target, so we need to extract it
    int target_index = num_columns - 1;

    // Allocate memory for the target values
    float *targets = (float *)malloc(num_rows * sizeof(float));
    if (!targets) {
        perror("Memory allocation failed for targets");
        exit(EXIT_FAILURE);
    }

    // Extract target values
    for (int i = 0; i < num_rows; i++) {
        targets[i] = data[i][target_index];
    }

    // Create an array of indices and shuffle them
    int *indices = (int *)malloc(num_rows * sizeof(int));
    if (!indices) {
        perror("Memory allocation failed for indices");
        free(targets);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_rows; i++) {
        indices[i] = i;
    }

    // Seed the random number generator
    srand(SEED);

    // Shuffle the indices to randomize the order
    for (int i = num_rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Create an array of stratified groups based on target values
    // Group the indices by target values
    int num_groups = 0;
    for (int i = 0; i < num_rows; i++) {
        int found = 0;
        for (int j = 0; j < num_groups; j++) {
            if (targets[indices[i]] == targets[indices[j]]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            num_groups++;
        }
    }

    // Allocate memory for train and test sets
    *train_size = (int)(num_rows * train_proportion);
    *test_size = num_rows - *train_size;

    *train_data = (float **)malloc(*train_size * sizeof(float *));
    *test_data = (float **)malloc(*test_size * sizeof(float *));
    if (!*train_data || !*test_data) {
        perror("Memory allocation failed for train/test data");
        free(targets);
        free(indices);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for each row in train and test sets
    for (int i = 0; i < *train_size; i++) {
        (*train_data)[i] = (float *)malloc(num_columns * sizeof(float));
    }
    for (int i = 0; i < *test_size; i++) {
        (*test_data)[i] = (float *)malloc(num_columns * sizeof(float));
    }

    // Split the data into train and test based on stratified groups
    int train_index = 0, test_index = 0;
    for (int i = 0; i < num_rows; i++) {
        if (i < *train_size) {
            for (int j = 0; j < num_columns; j++) {
                (*train_data)[train_index][j] = data[indices[i]][j];
            }
            train_index++;
        } else {
            for (int j = 0; j < num_columns; j++) {
                (*test_data)[test_index][j] = data[indices[i]][j];
            }
            test_index++;
        }
    }

    // Clean up memory
    free(targets);
    free(indices);
}