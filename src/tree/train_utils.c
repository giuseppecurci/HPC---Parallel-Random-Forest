#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "train_utils.h" 
#include "merge_sort.h" 

int argmax(int *arr, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

float compute_entropy(float *split, int size, int num_classes) {
    if (size == 0) return 0.0;  // Prevent division by zero

    // Allocate memory for class counts
    int *class_counts = calloc(num_classes, sizeof(int));
    if (!class_counts) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Count occurrences of each class
    for (int i = 0; i < size; i++) {
        int class_index = (int)split[i];  // Assuming classes are labeled as integers (0,1,2,...)
        if (class_index >= 0 && class_index < num_classes) {
            class_counts[class_index]++;
        }
    }

    // Compute entropy
    float entropy = 0.0;
    for (int i = 0; i < num_classes; i++) {
        if (class_counts[i] > 0) {
            float p = (float)class_counts[i] / size;
            entropy -= p * log2f(p);
        }
    }

    free(class_counts);  // Free allocated memory
    return entropy;
}

float get_entropy(float *left_split, float *right_split, int left_size, int right_size, int num_classes) {
    float left_entropy = compute_entropy(left_split, left_size, num_classes);
    float right_entropy = compute_entropy(right_split, right_size, num_classes);
    float weighted_entropy = (left_size * left_entropy + right_size * right_entropy) / (left_size + right_size);

    //printf("Left entropy: %.6f\n", left_entropy);
    //printf("Right entropy: %.6f\n", right_entropy);
    //printf("Weighted entropy: %.6f\n", weighted_entropy);
    
    // Weighted entropy sum
    return weighted_entropy;
}

float* get_best_split_num_var(
    float *sorted_array, 
    float *target_array, 
    int size, 
    int num_classes)
    {
        float* best_split = malloc(6 * sizeof(float));
        best_split[0] = INFINITY;  
        best_split[1] = 0.0;
        best_split[2] = best_split[3] = best_split[4] = best_split[5] = -1;

        int left_class_counts[num_classes];
        int right_class_counts[num_classes];
        memset(left_class_counts, 0, num_classes * sizeof(int));
        memset(right_class_counts, 0, num_classes * sizeof(int));
        
        for (int i = 0; i < size; i++)
        {
            float avg = (sorted_array[i] + sorted_array[i + 1]) / 2;
            int left_size = i + 1;
            int right_size = size - i - 1; 

            float* left_split = malloc((left_size) * sizeof(float));
            float* right_split = malloc((right_size) * sizeof(float));
            for (int j = 0; j < left_size; j++)
            {
                left_split[j] = target_array[j];
                left_class_counts[(int)target_array[j]]++;
            }
            for (int j = 0; j < right_size; j++)
            {
                right_split[j] = target_array[j + left_size];
                right_class_counts[(int)target_array[j + left_size]]++;
            }

            float entropy = get_entropy(left_split, right_split, left_size, right_size, num_classes);
            if (entropy < best_split[0])
            {
                best_split[0] = entropy;
                best_split[1] = avg;
                best_split[2] = left_size;
                best_split[3] = right_size;
                best_split[4] = argmax(left_class_counts, num_classes);
                best_split[5] = argmax(right_class_counts, num_classes);
            }
            for (int j = 0; j < num_classes; j++) {
                right_class_counts[j] = 0;
                left_class_counts[j] = 0;
                }
            free(left_split);
            free(right_split);
        }

        return best_split;
    }

BestSplit find_best_split(float **data, int num_rows, int num_columns, 
                          int num_classes, int *class_pred_left, int *class_pred_right,
                          int *best_size_left, int *best_size_right) 
                          {
    BestSplit best_split = {INFINITY, 0.0, -1};
    int target_column = num_columns - 1;  // Assuming target column is the last one

    for (int feature_col = 0; feature_col < num_columns; feature_col++) {
        if (feature_col == target_column) continue;  // Skip target column

        // Allocate arrays for sorting
        float *feature_values = malloc(num_rows * sizeof(float));
        float *target_values = malloc(num_rows * sizeof(float));
        if (!feature_values || !target_values) {
            fprintf(stderr, "Memory allocation failed!\n");
            exit(EXIT_FAILURE);
        }

        // Extract feature column and corresponding target values
        for (int i = 0; i < num_rows; i++) {
            feature_values[i] = data[i][feature_col];
            target_values[i] = data[i][target_column];
        }

        // Sort the feature and target values together
        merge_sort(feature_values, target_values, num_rows);

        // Find best split for this feature
        float *feature_best_split = get_best_split_num_var(feature_values, target_values, num_rows, num_classes);
        // Update the global best split if a lower entropy is found
        if (feature_best_split[0] < best_split.entropy) {
            best_split.entropy = feature_best_split[0];
            best_split.threshold = feature_best_split[1];
            *best_size_left = (int) feature_best_split[2];
            *best_size_right = (int) feature_best_split[3];
            *class_pred_left = (int) feature_best_split[4];
            *class_pred_right = (int) feature_best_split[5];
            best_split.feature_index = feature_col;
        }

        // Free allocated memory
        free(feature_best_split);
        free(feature_values);
        free(target_values);
    }

    return best_split;
}

void split_data(float** data, float** left_data, float** right_data, int num_rows, int target_index, float threshold) {
    int left_index = 0;
    int right_index = 0;

    for (int i = 0; i < num_rows; i++) {
        if (data[i][target_index] <= threshold) {
            left_data[left_index] = data[i];
            left_index++;
        } else {
            right_data[right_index] = data[i];
            right_index++;
        }
    }
    
}