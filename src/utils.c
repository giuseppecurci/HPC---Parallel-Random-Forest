#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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
    printf("Weighted entropy: %.6f\n", weighted_entropy);
    
    // Weighted entropy sum
    return weighted_entropy;
}

float* get_best_split_num_var(float *sorted_array, float *target_array, int size, int num_classes)
    {
        float* best_split = malloc(2 * sizeof(float));
        best_split[0] = INFINITY;  
        best_split[1] = 0.0;

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
            }
            for (int j = 0; j < right_size; j++)
            {
                right_split[j] = target_array[j + left_size];
            }

            float entropy = get_entropy(left_split, right_split, left_size, right_size, num_classes);
            if (entropy < best_split[0])
            {
                best_split[0] = entropy;
                best_split[1] = avg;
            }
            free(left_split);
            free(right_split);
        }
        return best_split;
    }