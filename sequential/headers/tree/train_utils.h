/**
 * @file train_utils.h
 * @brief Header file for the decision tree training.
 *
 * This file defines helper functions and structures used in the decision tree
 * training process. It includes functions for finding the best split, computing
 * entropy, and splitting the dataset based on a threshold.
 */

#ifndef TRAIN_TREE_H
#define TRAIN_TREE_H

#include "tree.h"

typedef struct {
    float entropy;
    float threshold;
    int feature_index;
} BestSplit;

/**
 * @brief Finds the index of the maximum value in an array.
 * 
 * This function searches through the array and returns the index of the
 * largest value.
 * 
 * @param arr Pointer to the array of integers.
 * @param size The size of the array.
 * @return The index of the maximum value in the array.
 */
int argmax(int *arr, int size);

/**
 * @brief Sorts the features and targets using merge sort.
 * 
 * This function initializes temporary arrays and calls the recursive helper
 * function to sort the feature and target arrays together.
 * 
 * @param features The array of feature values to be sorted.
 * @param targets The array of target values corresponding to features.
 * @param size The size of the arrays.
 */
void merge_sort(float *features, float *targets, int size);

/**
 * @brief Computes the entropy of a set of labels.
 * 
 * This function calculates the entropy of a split, which measures the
 * impurity or uncertainty in the dataset subset. It uses the formula for Shannon entropy.
 * 
 * @param split The array of target values for the split.
 * @param size The size of the split array.
 * @param num_classes The number of possible target classes.
 * @return The entropy value for the split.
 */
float compute_entropy(float *split, int size, int num_classes);

/**
 * @brief Computes the weighted entropy for a split of data.
 * 
 * This function computes the entropy for both the left and right splits and
 * returns the weighted sum of the entropies, which is used for decision tree splitting.
 * 
 * @param left_split The array of target values for the left split.
 * @param right_split The array of target values for the right split.
 * @param left_size The size of the left split.
 * @param right_size The size of the right split.
 * @param num_classes The number of possible target classes.
 * @return The weighted entropy for the split.
 */
float get_entropy(float *left_split, float *right_split, int left_size, int right_size, int num_classes);

/**
 * @brief Finds the best split for a feature using a sorted array.
 * 
 * This function evaluates all possible split points for a given feature and
 * returns the split with the minimum entropy. It calculates the best threshold,
 * sizes of left and right splits, and the predicted classes for each split.
 * 
 * @param sorted_array The sorted array of feature values.
 * @param target_array The array of target values corresponding to the features.
 * @param size The size of the arrays.
 * @param num_classes The number of possible target classes.
 * @return An array containing the best split's entropy, threshold, sizes of left and right splits,
 *         and the predicted class for each side.
 */
float* get_best_split_num_var(float *sorted_array, float *target_array, int size, int num_classes);

/**
 * @brief Finds the best split for all features in the dataset.
 * 
 * This function iterates over all features, finds the best split for each,
 * and returns the split with the lowest entropy. The prediction of the left and 
 * right split are stored for efficiency.
 * 
 * @param data The dataset to be split.
 * @param num_rows The number of rows in the dataset.
 * @param num_columns The number of columns in the dataset (including target).
 * @param num_classes The number of target classes.
 * @param class_pred_left Pointer to store the predicted class for the left split.
 * @param class_pred_right Pointer to store the predicted class for the right split.
 * @param best_size_left Pointer to store the size of the left split.
 * @param best_size_right Pointer to store the size of the right split.
 * @return The best split found, containing entropy, threshold, and other split parameters.
 */
BestSplit find_best_split(float **data, int num_rows, int num_columns, int num_classes, 
                          int *class_pred_left, int *class_pred_right, int *best_size_left, 
                          int *best_size_right, char* max_features);

/**
 * @brief Splits the dataset into left and right subarrays based on a threshold.
 * 
 * This function divides the dataset into two parts: one where the feature
 * values are less than or equal to the threshold, and one where they are
 * greater than the threshold. The arrays are allocated dynamically and deep-copied
 * from data.
 * 
 * @param data The dataset to be split.
 * @param left_data The left subarray after splitting.
 * @param right_data The right subarray after splitting.
 * @param num_rows The number of rows in the dataset.
 * @param num_columns The number of columns in the dataset.
 * @param target_index The index of the target column.
 * @param threshold The threshold value for splitting the data.
 */                          
void split_data(float** data, float** left_data, float** right_data, int num_rows, int num_columns, int target_index, float threshold);

/**
 * @brief Fisher-Yates algorithm implementation to shuffle an array in O(n)
 *
 * @param array The array to shuffle
 * @size the size of the array
 *
 */
void shuffle(int *array, int size);
#endif 
