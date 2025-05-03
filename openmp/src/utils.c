#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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

void summary(char* dataset_path, float train_proportion, int train_size, int num_columns,
             int num_classes, int num_trees, int max_depth, int min_samples_split, char* max_features, 
             char* store_predictions_path, char* store_metrics_path, char* new_tree_path, 
             char* trained_tree_path, int seed, int thread_count) {
        printf("Summary setup:\n");
        printf(" - Dataset: %s\n", dataset_path);
        printf(" - Train/test size: %.2f/%.2f\n", train_proportion, 1-train_proportion);
        printf(" - Training samples: %d\n", train_size);
        printf(" - Number of features: %d\n", num_columns);
        printf(" - Number of classes: %d\n", num_classes);
        printf(" - Predictions path: %s\n", store_predictions_path);
        printf(" - Metrics path: %s\n", store_metrics_path);
        if (trained_tree_path != NULL) {
            printf(" - Trained Forest path: %s\n", trained_tree_path);
        } else {
            printf(" - New Forest path: %s\n", new_tree_path);
        }
        printf(" - Number of trees: %d\n", num_trees);
        printf(" - Max depth: %d\n", max_depth);
        printf(" - Min samples split: %d\n", min_samples_split);
        printf(" - Max features: %s\n", max_features);
        printf(" - Seed: %d\n", seed);
        printf(" - Thread count: %d\n", thread_count);
        printf("--------------\n");
    };

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

int parse_arguments(int argc, char *argv[], int *max_matrix_rows_print, int *num_classes, int *num_trees,
                    int *max_depth, int *min_samples_split, char **max_features,
                    char **trained_forest_path, char **store_predictions_path, char **store_metrics_path,
                    char **new_forest_path, char **dataset_path, float *train_proportion, int *seed, int *thread_count) {

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
                return 1; 
            } 
        }
        else if (strcmp(argv[i], "--thread_count") == 0 && i + 1 < argc) {
            *thread_count = atoi(argv[i + 1]);
        }
    }

    return 0;  // Return 0 if everything is parsed successfully
}

void stratified_split(float **data, int num_rows, int num_columns, int num_classes, float train_proportion,
                      float ***train_data, int *train_size, float ***test_data, int *test_size, int seed) {

    int target_index = num_columns - 1; 

    srand(seed);

    int class_counts[num_classes];
    memset(class_counts, 0, num_classes * sizeof(int));
    int **class_indices = (int **)malloc(num_classes * sizeof(int *));
    
    // First pass: count samples per class
    for (int i = 0; i < num_rows; i++) {
        int label = (int)data[i][target_index];
        class_counts[label]++;
    }

    // Allocate space for indices
    for (int i = 0; i < num_classes; i++) {
        class_indices[i] = (int *)malloc(class_counts[i] * sizeof(int));
    }

    // Reset counts to use for filling indices
    int class_fill_ptrs[num_classes];
    memset(class_fill_ptrs, 0, num_classes * sizeof(int));

    // Second pass: collect indices per class
    for (int i = 0; i < num_rows; i++) {
        int label = (int)data[i][target_index];
        class_indices[label][class_fill_ptrs[label]++] = i;
    }

    // Compute train and test sizes
    *train_size = 0;
    *test_size = 0;
    for (int i = 0; i < num_classes; i++) {
        *train_size += (int)(class_counts[i] * train_proportion);
        *test_size += class_counts[i] - (int)(class_counts[i] * train_proportion);
    }

    *train_data = (float **)malloc(*train_size * sizeof(float *));
    *test_data = (float **)malloc(*test_size * sizeof(float *));
    for (int i = 0; i < *train_size; i++) {
        (*train_data)[i] = (float *)malloc(num_columns * sizeof(float));
    }
    for (int i = 0; i < *test_size; i++) {
        (*test_data)[i] = (float *)malloc(num_columns * sizeof(float));
    }

    // Now split inside each class
    int train_idx = 0;
    int test_idx = 0;
    for (int c = 0; c < num_classes; c++) {
        int n = class_counts[c];

        // Shuffle indices inside the class
        for (int i = n - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = class_indices[c][i];
            class_indices[c][i] = class_indices[c][j];
            class_indices[c][j] = tmp;
        }

        int num_train = (int)(n * train_proportion);

        for (int i = 0; i < n; i++) {
            int row_idx = class_indices[c][i];
            if (i < num_train) {
                for (int j = 0; j < num_columns; j++) {
                    (*train_data)[train_idx][j] = data[row_idx][j];
                }
                train_idx++;
            } else {
                for (int j = 0; j < num_columns; j++) {
                    (*test_data)[test_idx][j] = data[row_idx][j];
                }
                test_idx++;
            }
        }
    }

    // Clean up
    for (int i = 0; i < num_classes; i++) {
        free(class_indices[i]);
    }
    free(class_indices);
}

void store_run_params(char* csv_store_time_metrics_path, float time, int num_trees, int train_size, int thread_count) {
    struct stat buffer;
    int file_exists = (stat(csv_store_time_metrics_path, &buffer) == 0);

    float speedup = -1.0f;
    float efficiency = -1.0f;

    // If thread_count > 1, try to find the matching serial time
    if (thread_count > 1 && file_exists) {
        FILE *read_file = fopen(csv_store_time_metrics_path, "r");
        if (read_file != NULL) {
            char line[256];
            // Skip header
            fgets(line, sizeof(line), read_file);
            while (fgets(line, sizeof(line), read_file)) {
                float recorded_time;
                int recorded_threads, recorded_trees, recorded_data_size;
                int parsed = sscanf(line, "%f,%d,%d,%d", &recorded_time, &recorded_threads, &recorded_trees, &recorded_data_size);
                if (parsed >= 4 && recorded_threads == 1 &&
                    recorded_trees == num_trees &&
                    recorded_data_size == train_size) {
                        speedup = recorded_time / time;
                        efficiency = speedup / thread_count;
                        break;
                }
            }
            fclose(read_file);
        }
    }

    FILE *file = fopen(csv_store_time_metrics_path, "a");
    if (file == NULL) {
        perror("Error opening file for appending time metrics");
        return;
    }

    // Write header if file is new
    if (!file_exists) {
        fprintf(file, "Time,Threads,Num Trees,Data Size,Speedup,Efficiency\n");
    }

    if (speedup > 0 && efficiency > 0) {
        fprintf(file, "%.6f,%d,%d,%d,%.3f,%.3f\n", time, thread_count, num_trees, train_size, speedup, efficiency);
    } else {
        fprintf(file, "%.6f,%d,%d,%d,%d,%d\n", time, thread_count, num_trees, train_size, 1, 1);
    }

    fclose(file);
}