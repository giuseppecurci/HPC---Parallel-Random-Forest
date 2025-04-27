#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../headers/utils.h"

int parse_arguments(int argc, char *argv[], int *max_matrix_rows_print, int *num_classes, int *num_trees,
                    int *max_depth, int *min_samples_split, char **max_features,
                    char **trained_forest_path, char **store_predictions_path, char **store_metrics_path,
                    char **new_forest_path, char **dataset_path, float *train_proportion, int *seed) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print_matrix") == 0 && i + 1 < argc) {
            *max_matrix_rows_print = atoi(argv[i + 1]);  // Convert to integer
        }
        else if (strcmp(argv[i], "--num_classes") == 0 && i + 1 < argc) {
            *num_classes = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "--num_trees") == 0 && i + 1 < argc) {
            *num_trees = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "--trained_forest_path") == 0 && i + 1 < argc) {
            *trained_forest_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--store_predictions_path") == 0 && i + 1 < argc) {
            *store_predictions_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--store_metrics_path") == 0 && i + 1 < argc) {
            *store_metrics_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--new_forest_path") == 0 && i + 1 < argc) {
            *new_forest_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--dataset_path") == 0 && i + 1 < argc) {
            *dataset_path = argv[i + 1]; 
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            *seed = atoi(argv[i + 1]); 
        }
        else if (strcmp(argv[i], "--max_depth") == 0 && i + 1 < argc) {
            *max_depth = atoi(argv[i + 1]); 
        }
        else if (strcmp(argv[i], "--min_samples_split") == 0 && i + 1 < argc) {
            *min_samples_split = atoi(argv[i + 1]); 
        }
        else if (strcmp(argv[i], "--max_features") == 0 && i + 1 < argc) {
            *max_features = argv[i + 1];
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

// Function to read CSV and return the data array. Access with the formula: index = row * num_columns + column

float* read_csv(const char *filename, int *num_rows, int *num_columns) {
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
        // Count columns based on the number of commas
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
    
    // Count the total number of rows first
    int total_rows = 0;
    while (fgets(line, sizeof(line), file)) {
        total_rows++;
    }
    *num_rows = total_rows;
    
    // Go back to start of data
    rewind(file);
    fgets(line, sizeof(line), file); // Skip header again
    
    // Allocate a single contiguous block of memory
    float *data = (float*)malloc(sizeof(float) * total_rows * (*num_columns));
    if (!data) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }
    
    // Read data row by row
    row = 0;
    while (fgets(line, sizeof(line), file)) {
        char *token;
        int col = 0;
        token = strtok(line, ",");
        
        while (token && col < *num_columns) {
            // Calculate the flat array index
            int index = row * (*num_columns) + col;
            
            // Convert token to a float and store it in the data array
            data[index] = atof(token);
            
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    
    fclose(file);
    return data;
}

void stratified_split(float *data, int num_rows, int num_columns, float train_proportion,
                     float **train_data, int *train_size, float **test_data, int *test_size, int seed) {
    // The last column is the target, so we need to extract it
    int target_index = num_columns - 1;
    
    // Allocate memory for the target values
    float *targets = (float *)malloc(num_rows * sizeof(float));
    if (!targets) {
        perror("Memory allocation failed for targets");
        exit(EXIT_FAILURE);
    }
    
    // Extract target values from flat array
    for (int i = 0; i < num_rows; i++) {
        targets[i] = data[i * num_columns + target_index];
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
    srand(seed);
    
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
    
    // Calculate sizes for train and test sets
    *train_size = (int)(num_rows * train_proportion);
    *test_size = num_rows - *train_size;
    
    // Allocate memory for train and test sets as contiguous arrays
    *train_data = (float *)malloc(*train_size * num_columns * sizeof(float));
    *test_data = (float *)malloc(*test_size * num_columns * sizeof(float));
    
    if (!*train_data || !*test_data) {
        perror("Memory allocation failed for train/test data");
        free(targets);
        free(indices);
        exit(EXIT_FAILURE);
    }
    
    // Split the data into train and test based on stratified groups
    int train_index = 0, test_index = 0;
    for (int i = 0; i < num_rows; i++) {
        if (i < *train_size) {
            // Copy entire row to train data
            for (int j = 0; j < num_columns; j++) {
                (*train_data)[train_index * num_columns + j] = data[indices[i] * num_columns + j];
            }
            train_index++;
        } else {
            // Copy entire row to test data
            for (int j = 0; j < num_columns; j++) {
                (*test_data)[test_index * num_columns + j] = data[indices[i] * num_columns + j];
            }
            test_index++;
        }
    }
    
    // Clean up memory
    free(targets);
    free(indices);
}
