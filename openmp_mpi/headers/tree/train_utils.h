/**
 * @file train_utils.h
 * @brief Header file for parallel decision tree training utilities.
 *
 * This file contains utility functions and data structures used in the training
 * of decision trees in a parallel computing context (e.g., MPI). It includes 
 * operations for entropy calculation, dataset splitting, merge sort on features,
 * and finding the best split based on entropy.
 */

#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H

#define EPSILON 1e-9

#include "tree.h"

/**
 * @brief Finds the index of the maximum value in an array.
 * 
 * @param arr Pointer to the array of integers.
 * @param size The size of the array.
 * @return The index of the maximum value in the array.
 */
int argmax(int *arr, int size);

/**
 * @brief Merges two sorted halves of the feature and target arrays.
 * 
 * This function is used by the merge sort helper to combine two sorted
 * subarrays into one, preserving the relationship between features and targets.
 * 
 * @param features The feature array to be merged.
 * @param targets The corresponding target array.
 * @param temp_features Temporary array used for merging features.
 * @param temp_targets Temporary array used for merging targets.
 * @param left The starting index of the left subarray.
 * @param mid The midpoint index separating the two halves.
 * @param right The ending index of the right subarray.
 */
void merge(float *features, float *targets, float *temp_features, float *temp_targets, int left, int mid, int right);

/**
 * @brief Recursive merge sort helper for sorting feature and target arrays.
 * 
 * This function is used internally by the merge sort function to recursively
 * divide and sort the input arrays.
 * 
 * @param features The feature array to be sorted.
 * @param targets The corresponding target array.
 * @param temp_features Temporary array used for merging features.
 * @param temp_targets Temporary array used for merging targets.
 * @param left The starting index of the subarray.
 * @param right The ending index of the subarray.
 */
void merge_sort_helper(float *features, float *targets, float *temp_features, float *temp_targets, int left, int right);

/**
 * @brief Sorts the features and targets using merge sort.
 * 
 * This function sorts the input feature and target arrays together to
 * preserve the correspondence between them, which is essential for split evaluation.
 * 
 * @param features The feature array to be sorted.
 * @param targets The corresponding target array.
 * @param size The size of the arrays.
 */
void merge_sort(float *features, float *targets, int size);

/**
 * @brief Computes the entropy of a set of class probabilities or counts.
 * 
 * This version of entropy calculation assumes the input is a float array
 * representing a class distribution or soft counts.
 * 
 * @param split The array representing the class distribution.
 * @param size The number of classes or elements in the array.
 * @param num_classes The total number of possible classes.
 * @return The computed entropy.
 */
float compute_entropy(int *class_counts, int size, int num_classes);

/**
 * @brief Computes the weighted entropy for a data split.
 * 
 * This function calculates the entropy for both the left and right splits of the dataset
 * and returns their weighted sum based on the sizes of the splits.
 * 
 * @param left_split Class distribution for the left split.
 * @param right_split Class distribution for the right split.
 * @param left_size The size of the left split.
 * @param right_size The size of the right split.
 * @param num_classes The total number of target classes.
 * @return The weighted entropy for the split.
 */
float get_entropy(int *left_class_counts, int *right_class_counts, int left_size, int right_size, int num_classes);

/**
 * @brief Finds the best threshold for splitting a sorted feature.
 * 
 * This function evaluates different thresholds for a single feature and 
 * returns an array containing the best split's metrics such as entropy and threshold.
 * 
 * @param sorted_array Sorted values of a single feature.
 * @param target_array Target values corresponding to the sorted features.
 * @param size The number of elements in the arrays.
 * @param num_classes The number of target classes.
 * @return A float array with the best entropy, threshold, and split sizes and predictions.
 */
float* get_best_split_num_var(float *sorted_array, float *target_array, int size, int num_classes, int thread_count);

/**
 * @brief Fisher-Yates shuffle algorithm to randomize an array.
 * 
 * This function shuffles an array of integers in-place in linear time.
 * 
 * @param array The array to shuffle.
 * @param size The number of elements in the array.
 */
void shuffle(int *array, int size);

/**
 * @brief Finds the best split across all features of the dataset.
 * 
 * This function evaluates all features, determines the best split based on
 * entropy, and returns a structure with the best split information. Also stores 
 * predicted classes and sizes for each side of the split.
 * 
 * @param data The dataset as a 2D array.
 * @param num_rows The number of samples (rows).
 * @param num_columns The number of features (columns).
 * @param num_classes The total number of target classes.
 * @param class_pred_left Pointer to store the predicted class of the left split.
 * @param class_pred_right Pointer to store the predicted class of the right split.
 * @param best_size_left Pointer to store the size of the left split.
 * @param best_size_right Pointer to store the size of the right split.
 * @param max_features Strategy for selecting the subset of features to evaluate.
 * @return A BestSplit structure containing the optimal split parameters.
 */
BestSplit find_best_split(float **data, int num_rows, int num_columns, 
                          int num_classes, int *class_pred_left, int *class_pred_right,
                          int *best_size_left, int *best_size_right, char *max_features);

/**
 * @brief Splits the dataset into left and right based on a threshold.
 * 
 * This function partitions the dataset into two groups depending on whether
 * the values in the target column are less than or equal to a threshold or not.
 * 
 * @param data The dataset to be split.
 * @param left_data Output pointer to the left dataset.
 * @param right_data Output pointer to the right dataset.
 * @param num_rows The number of rows in the original dataset.
 * @param num_columns The number of columns in the dataset.
 * @param target_index The index of the feature used for splitting.
 * @param threshold The threshold used to split the data.
 */
void split_data(float **data, float **left_data, float **right_data, int num_rows, int num_columns, int target_index, float threshold);


/**
 * @brief Finds the best feature and threshold to split the data.
 * 
 * This function evaluates different features and thresholds to find
 * the split that minimizes entropy in the resulting subsets.
 * 
 * @param data The input dataset as a float array.
 * @param num_rows Number of samples in the dataset.
 * @param num_columns Number of features in the dataset (including the label).
 * @param num_classes Number of unique classes in the dataset.
 * @param class_pred_left Pointer to store the predicted class for the left split.
 * @param class_pred_right Pointer to store the predicted class for the right split.
 * @param best_size_left Pointer to store the number of samples in the left split.
 * @param best_size_right Pointer to store the number of samples in the right split.
 * @param max_features Strategy for selecting features to consider for splitting.
 * @return A BestSplit structure containing information about the best split found.
 */
BestSplit find_best_split_1d(float *data, int num_rows, int num_columns, int num_classes, 
                            int *class_pred_left, int *class_pred_right, 
                            int *best_size_left, int *best_size_right, char *max_features, int num_threads);

#endif // TRAIN_UTILS_H

