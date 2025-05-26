#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "../../headers/tree/train_utils.h" 
#include "../../headers/tree/tree.h"
#include "../../headers/tree/utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int argmax(int *arr, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

void merge(float *features, float *targets, float *temp_features, float *temp_targets, int left, int mid, int right) {
    int i = left;      // index left subarray
    int j = mid + 1;   // index right subarray
    int k = left;      // index temp array
    
    // merge the subarrays
    while (i <= mid && j <= right) {
        if (features[i] <= features[j]) {
            temp_features[k] = features[i];
            temp_targets[k] = targets[i];  // Maintain alignment
            i++;
        } else {
            temp_features[k] = features[j];
            temp_targets[k] = targets[j];  // Maintain alignment
            j++;
        }
        k++;
    }
    
    // copy the remaining elements from left subarray
    while (i <= mid) {
        temp_features[k] = features[i];
        temp_targets[k] = targets[i];
        i++;
        k++;
    }
    
    // copy the remaining elements from right subarray
    while (j <= right) {
        temp_features[k] = features[j];
        temp_targets[k] = targets[j];
        j++;
        k++;
    }
    
    // copy back the sorted elements to original array
    for (i = left; i <= right; i++) {
        features[i] = temp_features[i];
        targets[i] = temp_targets[i];
    }
}

void merge_sort_helper(float *features, float *targets, float *temp_features, float *temp_targets, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;  // Avoid integer overflow

        // Recursively sort the subarrays
        merge_sort_helper(features, targets, temp_features, temp_targets, left, mid);
        merge_sort_helper(features, targets, temp_features, temp_targets, mid + 1, right);

        // Merge the sorted subarrays
        merge(features, targets, temp_features, temp_targets, left, mid, right);
    }
}

void merge_sort(float *features, float *targets, int size) {
    float *temp_features = (float *)malloc(size * sizeof(float));
    float *temp_targets = (float *)malloc(size * sizeof(float));
    if (temp_features == NULL || temp_targets == NULL) {
        printf("Memory allocation failed\n");
        free(temp_features);
        free(temp_targets);
        return;
    }
    merge_sort_helper(features, targets, temp_features, temp_targets, 0, size - 1);
    
    free(temp_features);
    free(temp_targets);
}

// quiquiqui
//
float* get_best_split_num_var(
    float *sorted_array, 
    float *target_array, 
    int size, 
    int num_classes,
    int num_threads)
{
    float* best_split = malloc(6 * sizeof(float));
    best_split[0] = INFINITY;  
    best_split[1] = 0.0;
    best_split[2] = best_split[3] = best_split[4] = best_split[5] = -1;
    
    int left_class_counts[num_classes];
    int right_class_counts[num_classes];
    
    #pragma omp parallel for num_threads(num_threads) private(left_class_counts, right_class_counts)
    for (int i = 0; i < size - 1; i++)
    {   
        float avg = (sorted_array[i] + sorted_array[i + 1]) / 2;
        int left_size = i + 1;
        int right_size = size - i - 1; 
        
        memset(left_class_counts, 0, num_classes * sizeof(int));
        memset(right_class_counts, 0, num_classes * sizeof(int));
        
        for (int j = 0; j < left_size; j++) {
            left_class_counts[(int)target_array[j]]++;
        }
        for (int j = 0; j < right_size; j++) {
            right_class_counts[(int)target_array[j + left_size]]++;
        }
    
        float entropy = get_entropy(left_class_counts, right_class_counts, left_size, right_size, num_classes);
        
        #pragma omp critical 
        {
            if (
                (entropy + EPSILON < best_split[0]) || 
                (fabs(entropy - best_split[0]) < EPSILON && avg < best_split[1])
            )
            {
                best_split[0] = entropy;
                best_split[1] = avg;
                best_split[2] = left_size;
                best_split[3] = right_size;
                best_split[4] = argmax(left_class_counts, num_classes);
                best_split[5] = argmax(right_class_counts, num_classes);
            }
        }
    }
    return best_split;
}

float get_entropy(int *left_class_counts, int *right_class_counts, int left_size, int right_size, int num_classes) {
    float left_entropy = compute_entropy(left_class_counts, left_size, num_classes);
    float right_entropy = compute_entropy(right_class_counts, right_size, num_classes);
    float weighted_entropy = (left_size * left_entropy + right_size * right_entropy) / (left_size + right_size);
    
    return weighted_entropy;
}

float compute_entropy(int *class_counts, int size, int num_classes) {
    if (size == 0) return 0.0;  // Prevent division by zero
    
    float entropy = 0.0;
    for (int i = 0; i < num_classes; i++) {
        if (class_counts[i] > 0) {
            float p = (float)class_counts[i] / size;
            entropy -= p * log2f(p);
        }
    }
    return entropy;
}



void shuffle(int *array, int size) {
    // Fisher-Yates shuffle algorithm
    for (int i = size - 1; i > 0; i--) {
        // Generate a random index between 0 and i (inclusive)
        int j = rand() % (i + 1);
        
        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}



void split_data(float** data, float** left_data, float** right_data, int num_rows, int num_columns, int target_index, float threshold) {
    int left_index = 0;
    int right_index = 0;

    for (int i = 0; i < num_rows; i++) {
        if (data[i][target_index] <= threshold) {
            left_data[left_index] = (float *)malloc(sizeof(float) * num_columns);  // Allocate memory for each row
            memcpy(left_data[left_index], data[i], sizeof(float) * num_columns);  // Deep copy the row
            left_index++;
        } else {
            right_data[right_index] = (float *)malloc(sizeof(float) * num_columns);  // Allocate memory for each row
            memcpy(right_data[right_index], data[i], sizeof(float) * num_columns);  // Deep copy the row
            right_index++;
        }
    }
}
