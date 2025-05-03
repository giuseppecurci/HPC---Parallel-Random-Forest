/**
 * @file tree.h
 * @brief Header file for the decision tree structure and functions.
 *
 * This file defines the structures and function prototypes for building,
 * training, and performing inference with a decision tree.
 */

#ifndef FOREST_H
#define FOREST_H

#include "tree/tree.h"

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

/**
 * @brief Creates a random forest with the specified parameters.
 *
 * @param forest Pointer to the Forest structure to be initialized.
 * @param num_trees Number of trees in the forest.
 * @param max_depth Maximum depth for each tree.
 * @param min_samples_split Minimum number of samples required to split a node.
 * @param max_features Number of features to consider when looking for the best split.
 */
void create_forest(Forest *forest, int num_trees, int max_depth, int min_samples_split, char* max_features);

/**
 * @brief Trains the random forest on the provided dataset.
 *
 * @param forest Pointer to the Forest structure to be trained.
 * @param data 2D array of float data (last column is the class label).
 * @param num_rows Number of data samples.
 * @param num_columns Number of columns in each data sample (features + 1 label).
 * @param num_classes Total number of classes.
 */
void train_forest(Forest *forest, float **data, int num_rows, int num_columns, int num_classes, int thread_count);

/**
 * @brief Performs inference on the provided dataset using the trained random forest.
 *
 * @param forest Pointer to the trained Forest structure.
 * @param data 2D array of float data (last column is the class label).
 * @param num_rows Number of data samples.
 * @param num_classes Total number of classes.
 * @return Array of predicted class labels for each sample in the dataset.
 */
int* forest_inference(Forest *forest, float **data, int num_rows, int num_classes);

/**
 * @brief Frees the memory allocated for the random forest and its trees.
 *
 * @param forest Pointer to the Forest structure to be freed.
 */
void free_forest(Forest *forest);

/**
 * @brief Serializes the random forest to a binary file.
 *
 * @param forest Pointer to the Forest structure to be serialized.
 * @param filename Path to the output file where the forest will be saved.
 */
void serialize_forest(Forest *forest, const char *filename);

/**
 * @brief Deserializes a random forest from a binary file.
 *
 * @param forest Pointer to the Forest structure to be deserialized.
 * @param filename Path to the input file from which the forest will be loaded.
 */
void deserialize_forest(Forest *forest, const char *filename);

#endif
