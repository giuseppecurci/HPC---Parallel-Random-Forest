/**
 * @file utils.h
 * @brief Utility functions for matrix printing, array printing, and CSV file reading.
 * 
 * This file provides various utility functions such as:
 * - Printing a matrix with specified formatting.
 * - Printing an array with a specified maximum number of elements.
 * - Reading a CSV file and storing the data into a matrix.
 * 
 * These functions can be used for debugging and data processing purposes.
 * 
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>
#include <time.h>

//Max number of characters that can be stored in the buffer line
#define MAX_LINE 1024
#define MAX_ROWS 20000000

/**
 * @brief Prints a matrix with specified number of rows and columns.
 * 
 * This function prints a feature matrix with the option to limit the number of rows printed.
 * The target column is included in the output with a separator between features and the target.
 * 
 * @param data The matrix to be printed.
 * @param num_rows The number of rows in the matrix.
 * @param num_columns The number of columns in the matrix.
 * @param max_rows The maximum number of rows to print. If set to -1, all rows are printed.
 */
void print_matrix(float **data, int num_rows, int num_columns, int max_rows);

/**
 * @brief Prints the elements of a single-dimensional array.
 * 
 * This function prints the elements of an array up to a specified maximum number of elements.
 * 
 * @param arr The array to be printed.
 * @param size The total size of the array.
 * @param max_elements The maximum number of elements to print. If set to -1, all elements are printed.
 */
void print_array(float *arr, int size, int max_elements);

/**
 * @brief Reads data from a CSV file and returns it as a matrix.
 * 
 * This function reads data from a CSV file, allocates memory for a matrix, and stores the data.
 * The first row of the CSV is treated as a header, and the data rows are read into the matrix.
 * 
 * @param filename The name of the CSV file to be read.
 * @param num_rows Output parameter that will store the number of rows in the matrix.
 * @param num_columns Output parameter that will store the number of columns in the matrix.
 * @return A pointer to the matrix containing the data from the CSV file, or NULL if an error occurs.
 */
float** read_csv(const char *filename, int *num_rows, int *num_columns);

/**
 * @brief Performs a stratified split of the data into training and testing sets.
 * 
 * This function splits the data into training and testing sets while maintaining the distribution
 * of target classes in both sets. The split is done based on a specified proportion for the training set.
 * 
 * @param data The input data matrix to be split.
 * @param num_rows The number of rows in the input data matrix.
 * @param num_columns The number of columns in the input data matrix.
 * @param train_proportion The proportion of data to be used for training (between 0 and 1).
 * @param train_data Output parameter that will store the training data matrix.
 * @param train_size Output parameter that will store the size of the training data matrix.
 * @param test_data Output parameter that will store the testing data matrix.
 * @param test_size Output parameter that will store the size of the testing data matrix.
 * @param seed Random seed for reproducibility.
 */
void stratified_split(float **data, int num_rows, int num_columns, int num_classes, float train_proportion,
                      float ***train_data, int *train_size, float ***test_data, int *test_size, int seed);

/**
 * @brief Parses command-line arguments for various options.
 * 
 * This function parses command-line arguments to set various parameters. It returns 0 on success
 * and 1 on error. The function checks for the following arguments:
 * @param max_matrix_rows_print Maximum number of rows to print from the matrix (--print_matrix).
 * @param num_classes Number of classes in the dataset (--num_classes int).
 * @param num_trees Number of trees in the forest. (--num_trees int).
 * @param max_depth Maximum depth of the trees. (--max_depth int).
 * @param min_samples_split Minimum number of samples required to split a node. (--min_samples_split int).
 * @param max_features Number of features to consider when looking for the best split. (--max_features char*).
 * @param trained_tree_path Path for the trained tree to deserialize (--trained_tree_path).
 * @param store_predictions_path Path for storing predictions (--store_predictions_path).
 * @param store_metrics_path Path for storing performance metrics (--store_metrics_path).
 * @param new_tree_path Path for the new tree to train and then serialize (--new_tree_path).
 * @param dataset_path Path for the dataset to be used (--dataset_path).
 * @param train_proportion Proportion of data to be used for training (--train_proportion).
 * @param num_trees Number of trees to be used in the forest (--num_trees).
 * @param seed Random seed for reproducibility (--seed).
 */
int parse_arguments(int argc, char *argv[], int *max_matrix_rows_print, int *num_classes, int *num_trees,
                    int *max_depth, int *min_samples_split, char **max_features, char **trained_tree_path, 
                    char **store_predictions_path, char **store_metrics_path, char **new_tree_path, 
                    char **dataset_path, float *train_proportion, int *seed, int *thread_count);

/**
 * @brief Prints config used for a run.
 * 
 * This function prints the configuration used for the run and the various paths used to store results.
 * 
 * @param dataset_path Path to the dataset.
 * @param train_proportion Proportion of data used for training.
 * @param train_size Size of the training data.
 * @param num_columns Number of columns in the dataset.
 * @param num_trees Number of trees in the forest.
 * @param max_depth Maximum depth of the trees.
 * @param min_samples_split Minimum number of samples required to split a node.
 * @param max_features Number of features to consider when looking for the best split.
 * @param num_classes Number of classes in the dataset.
 * @param store_predictions_path Path to store predictions.
 * @param store_metrics_path Path to store performance metrics.
 * @param new_tree_path Path for the new tree.
 * @param trained_tree_path Path for the trained tree.
 * @param seed Random seed used for the run.
 */
void summary(char* dataset_path, float train_proportion, int train_size, int num_columns,
             int num_classes, int num_trees, int max_depth, int min_samples_split, char* max_features, 
             char* store_predictions_path, char* store_metrics_path, char* new_tree_path, 
             char* trained_tree_path, int seed, int thread_count);
#endif