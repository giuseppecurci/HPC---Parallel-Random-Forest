#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../headers/utils.h"
#include <sys/stat.h>

int parse_arguments(int argc, char *argv[], int *max_matrix_rows_print, int *num_classes, int *num_trees,
                    int *max_depth, int *min_samples_split, char **max_features,
                    char **trained_forest_path, char **store_predictions_path, char **store_metrics_path,
                    char **new_forest_path, char **dataset_path, float *train_proportion, float *train_tree_proportion, int *seed, int *thread_count) {

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
        else if (strcmp(argv[i], "--thread_count") == 0 && i + 1 < argc) {
            *thread_count = atoi(argv[i + 1]);
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

void stratified_split(float *data, int num_rows, int num_columns, int num_classes, float train_proportion,
                      float **train_data, int *train_size, float **test_data, int *test_size, int seed) {

    int target_index = num_columns - 1;

    srand(seed);

    int *class_counts = (int *)calloc(num_classes, sizeof(int));
    int **class_indices = (int **)malloc(num_classes * sizeof(int *));

    // First pass: count samples per class
    for (int i = 0; i < num_rows; i++) {
        int label = (int)data[i * num_columns + target_index];
        class_counts[label]++;
    }

    // Allocate space for indices
    for (int i = 0; i < num_classes; i++) {
        class_indices[i] = (int *)malloc(class_counts[i] * sizeof(int));
    }

    // Reset fill pointers
    int *class_fill_ptrs = (int *)calloc(num_classes, sizeof(int));

    // Second pass: collect indices per class
    for (int i = 0; i < num_rows; i++) {
        int label = (int)data[i * num_columns + target_index];
        class_indices[label][class_fill_ptrs[label]++] = i;
    }

    // Compute train and test sizes
    *train_size = 0;
    *test_size = 0;
    for (int i = 0; i < num_classes; i++) {
        *train_size += (int)(class_counts[i] * train_proportion);
        *test_size += class_counts[i] - (int)(class_counts[i] * train_proportion);
    }

    // Allocate memory for train and test sets (flat arrays)
    *train_data = (float *)malloc((*train_size) * num_columns * sizeof(float));
    *test_data = (float *)malloc((*test_size) * num_columns * sizeof(float));

    if (!*train_data || !*test_data) {
        perror("Memory allocation failed for train/test data");
        exit(EXIT_FAILURE);
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
                    (*train_data)[train_idx * num_columns + j] = data[row_idx * num_columns + j];
                }
                train_idx++;
            } else {
                for (int j = 0; j < num_columns; j++) {
                    (*test_data)[test_idx * num_columns + j] = data[row_idx * num_columns + j];
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
    free(class_counts);
    free(class_fill_ptrs);
}
void summary(char* dataset_path, float train_proportion, float train_tree_proportion, int train_size, int num_columns,
             int num_classes, int num_trees, int max_depth, int min_samples_split, char* max_features, 
             char* store_predictions_path, char* store_metrics_path, char* new_tree_path, 
             char* trained_tree_path, int seed) {
        printf("Summary setup:\n");
        printf(" - Dataset: %s\n", dataset_path);
        printf(" - Train/test size: %.2f/%.2f\n", train_proportion, 1-train_proportion);
        printf(" - Training samples: %d\n", train_size);
        printf(" - Training samples per tree (%.2f%%): %d\n", train_tree_proportion, (int)(train_size * train_tree_proportion));
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
        printf("--------------\n");
    };
/**
 * Samples data without replacement from the training dataset
 *
 * @param train_data The full training dataset
 * @param train_size Number of samples in the training dataset
 * @param num_columns Number of features per sample
 * @param sample_proportion Proportion of data to sample (e.g., 0.75 for 75%)
 * @param sampled_data Output buffer for the sampled data (must be pre-allocated)
 * @return Returns 0 on success, non-zero on failure
 */

int sample_data_without_replacement(float *train_data, int train_size, int num_columns, 
                                    float sample_proportion, float *sampled_data, int seed) {
    if (train_data == NULL || sampled_data == NULL || train_size <= 0 || 
        num_columns <= 0 || sample_proportion <= 0 || sample_proportion > 1) {
        fprintf(stderr, "Invalid parameters for data sampling\n");
        return 1;
    }

    int sample_size = (int)(sample_proportion * train_size);
    if (sample_size <= 0) {
        fprintf(stderr, "Sample size is too small\n");
        return 1;
    }

    // Allocate and initialize index array
    int *indices = (int *)malloc(train_size * sizeof(int));
    if (indices == NULL) {
        fprintf(stderr, "Failed to allocate memory for indices\n");
        return 1;
    }

    for (int i = 0; i < train_size; i++) {
        indices[i] = i;
    }

    // Fisher-Yates shuffle
    srand(seed);
    for (int i = train_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Copy the first sample_size rows using flat memory layout
    for (int i = 0; i < sample_size; i++) {
        int original_row = indices[i];
        for (int j = 0; j < num_columns; j++) {
            sampled_data[i * num_columns + j] = train_data[original_row * num_columns + j];
        }
    }

    free(indices);
    return sample_size;
}

void distribute_trees(int num_trees, int size, int *counts, int *displs) {
    int base = num_trees / size;
    int rem = num_trees % size;

    for (int i = 0; i < size; i++) {
        counts[i] = base + (i < rem ? 1 : 0);  // Spread remainder evenly
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }
};

void store_run_params_processes_threads(char* csv_store_time_metrics_path, float train_time, float inference_time, int num_trees, int train_size, int process_count, int num_threads) {
    struct stat buffer;
    int file_exists = (stat(csv_store_time_metrics_path, &buffer) == 0);
    float speedup = -1.0f;
    float efficiency = -1.0f;
    float total_time = train_time + inference_time;
	

    // Baseline is process_count == 2 && num_threads == 1
    if (file_exists && !(process_count == 2 && num_threads == 1)) {
        FILE *read_file = fopen(csv_store_time_metrics_path, "r");
        if (read_file != NULL) {
            char line[256];
            // Skip header
            fgets(line, sizeof(line), read_file);
            while (fgets(line, sizeof(line), read_file)) {
                float recorded_train_time, recorded_inference_time, recorded_total_time;
                int recorded_processes, recorded_threads, recorded_trees, recorded_data_size;
                float dummy_speedup, dummy_efficiency;

                int parsed = sscanf(line, "%f,%f,%f,%d,%d,%d,%d,%f,%f",
                    &recorded_train_time,
                    &recorded_inference_time,
                    &recorded_total_time,
                    &recorded_processes,
                    &recorded_threads,
                    &recorded_trees,
                    &recorded_data_size,
                    &dummy_speedup,
                    &dummy_efficiency);

                if (parsed == 9 &&
                    recorded_processes == 2 &&
                    recorded_threads == 1 &&
                    recorded_trees == num_trees &&
                    recorded_data_size == train_size) {
                        speedup = recorded_total_time / total_time;
                        efficiency = speedup / (process_count * num_threads);
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
        fprintf(file, "Train Time,Inference Time,Total Time,Processes,Num Threads,Num Trees,Data Size,Speedup,Efficiency\n");
    } else {
        // Ensure previous line ends with newline
        fseek(file, -1, SEEK_END);
        int last_char = fgetc(file);
        if (last_char != '\n') {
            fputc('\n', file);
        }
    }

    printf("Current speedup = %f, current efficiency = %f, current threads = %d, current processes = %d\n", speedup, efficiency, num_threads, process_count);

    if (process_count == 2 && num_threads == 1) {
        fprintf(file, "%.6f,%.6f,%.6f,%d,%d,%d,%d,1.000,1.000\n",
            train_time, inference_time, total_time,
            process_count, num_threads, num_trees, train_size);
    } else if (speedup > 0 && efficiency > 0) {
        fprintf(file, "%.6f,%.6f,%.6f,%d,%d,%d,%d,%.3f,%.3f\n",
            train_time, inference_time, total_time,
            process_count, num_threads, num_trees, train_size,
            speedup, efficiency);
    } else {
        fprintf(file, "%.6f,%.6f,%.6f,%d,%d,%d,%d,-1.000,-1.000\n",
            train_time, inference_time, total_time,
            process_count, num_threads, num_trees, train_size);
    }

    fclose(file);
}

