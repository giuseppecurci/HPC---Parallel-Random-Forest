#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H

#include "tree.h"

// Function declarations
int argmax(int *arr, int size);
void merge(float *features, float *targets, float *temp_features, float *temp_targets, int left, int mid, int right);
void merge_sort_helper(float *features, float *targets, float *temp_features, float *temp_targets, int left, int right);
void merge_sort(float *features, float *targets, int size);
float compute_entropy(float *split, int size, int num_classes);
float get_entropy(float *left_split, float *right_split, int left_size, int right_size, int num_classes);
float* get_best_split_num_var(float *sorted_array, float *target_array, int size, int num_classes);
void shuffle(int *array, int size);
BestSplit find_best_split(float **data, int num_rows, int num_columns, 
                          int num_classes, int *class_pred_left, int *class_pred_right,
                          int *best_size_left, int *best_size_right, char *max_features);
void split_data(float **data, float **left_data, float **right_data, int num_rows, int num_columns, int target_index, float threshold);

#endif // TRAIN_UTILS_H

