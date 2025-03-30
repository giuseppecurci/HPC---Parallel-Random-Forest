//prevent multiple inclusions of the same header file//
#ifndef UTILS_H 
#define UTILS_H

#include <stdio.h>
#include <math.h>

typedef struct {
    float entropy;
    float threshold;
    int feature_index;
} BestSplit;

int argmax(int *arr, int size);
void print_matrix(float **data, int num_rows, int num_columns, int max_rows);
void print_array(float *arr, int size, int max_elements);
float compute_entropy(float *split, int size, int num_classes);
float get_entropy(float *left_split, float *right_split, int left_size, int right_size, int num_classes);
float* get_best_split_num_var(float *sorted_array, float *target_array, int size, int num_classes);
BestSplit find_best_split(float **data, int num_rows, int num_columns, int num_classes, 
                          int *class_pred_left, int *class_pred_right, int *best_size_left, 
                          int *best_size_right);
void split_data(float** data, float** left_data, float** right_data, int num_rows, int target_index, float threshold);

#endif