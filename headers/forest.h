/**
 * @file tree.h
 * @brief Header file for the decision tree structure and functions.
 *
 * This file defines the structures and function prototypes for building,
 * training, and performing inference with a decision tree.
 */

#ifndef FOREST_H
#define FOREST_H

#include "tree.h"

/**
 * @struct Forest
 * @brief Collection of decision trees and hyperparameters.
 *
 * This structure holds information about a collection of decision trees,
 * including the number of trees, the maximum depth for each tree, and the
 * minimum number of samples required to split a node. It also contains
 * hyperparameters for the random forest algorithm, such as the number of
 * features to consider when looking for the best split.
 * 
 */
typedef struct Forest {
    int num_trees;          /**< Number of trees in the forest. */
    int max_depth;          /**< Maximum depth for each tree. */
    int min_samples_split;  /**< Minimum number of samples required to split a node. */
    char* max_features;     /**< Number of features to consider when looking for the best split. Possible values: {“sqrt”, “log2”, "int"} */
    Tree* trees;            /**< Array of decision trees in the forest. */
} Forest;

void create_forest(Forest *forest, int num_trees, int max_depth, int min_samples_split, char* max_features);

void train_forest(Forest *forest, float **data, int num_rows, int num_columns, int num_classes);

int* forest_inference(Forest *forest, float **data, int num_rows, int num_classes);

void free_forest(Forest *forest);

void serialize_forest(Forest *forest, const char *filename);

Forest* deserialize_forest(const char *filename);

#endif