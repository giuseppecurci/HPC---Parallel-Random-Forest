#include "../headers/utils.h"
#include <math.h>

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

float get_entropy(float *left_split, float *right_split, int left_size, int right_size) {
    // Calculate the entropy of the left and right child nodes
    float left_entropy = 0.0, right_entropy = 0.0;
    float p_left = 0.0, p_right = 0.0;

    // Calculate the entropy of the left child node
    for (int i = 0; i < left_size; i++) {
        if (left_split[i] == 1.0) {
            p_left += 1.0;
        }
    }

    if (p_left > 0.0 && p_left != left_size) {
        p_left = p_left/left_size;
        left_entropy = -p_left * log2f(p_left) - (1-p_left) * log2f(1-p_left);
    }
    else {
        left_entropy = 0.0;
    }

    // Calculate the entropy of the right child node
    for (int i = 0; i < right_size; i++) {
        if (right_split[i] == 1.0) {
            p_right += 1.0;
        }
    }
    if (p_right > 0.0 && p_right != right_size) {
        p_right = p_right/right_size;
        right_entropy = -p_right * log2f(p_right) - (1-p_right) * log2f(1-p_right);
    }
    else {
        right_entropy = 0.0;
    }

    printf("Left Entropy: %.6f\n", left_entropy);
    printf("Right Entropy: %.6f\n", right_entropy);

    // Calculate the weighted average of the child nodes' entropy
    return (left_size * left_entropy + right_size * right_entropy) / (left_size + right_size);
}