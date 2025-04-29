/**
 * @file utils.h
 * @brief Header file containing utility functions for managing decision trees.
 *
 * This file includes the function prototypes for managing and manipulating decision trees, 
 * including functions for making predictions, freeing memory, printing the tree structure, 
 * saving predictions to a file, and serializing/deserializing tree structures to and from 
 * binary files for persistence.
 *
 * The utilities provided include:
 * - Tree traversal and printing
 * - Memory management (destroying nodes and trees)
 * - Serialization and deserialization of trees
 * - Saving prediction results to files
 * 
*/

#ifndef UTILS_TREE_H
#define UTILS_TREE_H

#include "tree.h"
#include <stdio.h>

/**
 * @brief Assigns the most frequent class label in the current node's data as its prediction.
 * 
 * This function is not used in the current implementation but provides functionality for 
 * assigning the class prediction of a node based on the most frequent class label in the data.
 * In the current implementation, the prediction is set during the training phase and this function
 * is not currently used.
 * 
 * @param data         2D array of float data where the last column represents the class label.
 * @param num_rows     Number of data samples.
 * @param num_columns  Number of columns in each data sample (features + 1 for class label).
 * @param num_classes  Total number of classes.
 * @param node         Pointer to the node where prediction will be stored.
 */
void get_class_pred(float** data, int num_rows, int num_columns, int num_classes, Node *node);

/**
 * @brief Recursively frees memory allocated for the nodes and the tree.
 * 
 * This function frees all dynamically allocated memory for the tree structure.
 * It is a recursive function that destroys nodes and their children.
 * 
 * @param tree Pointer to the tree that needs to be destroyed.
 */
void destroy_tree(Tree *tree);

/**
 * @brief Recursively frees memory allocated for a node and its children.
 * 
 * This function frees all dynamically allocated memory for a node and its children 
 * in a recursive manner.
 * 
 * @param node Pointer to the node that needs to be destroyed.
 */
void destroy_node(Node *node);

/**
 * @brief Prints the structure and contents of the tree.
 * 
 * This function prints out the tree structure, starting from the root, and prints the 
 * node information, such as feature, threshold, prediction, number of samples, etc.
 * 
 * @param tree Pointer to the tree to be printed.
 */
void print_tree(Tree *tree);

/**
 * @brief Recursively prints node information (feature, threshold, prediction, etc.).
 * 
 * This function prints the details of each node in the tree, including the feature index, 
 * threshold, predicted class, entropy, depth, and the number of samples in that node.
 * 
 * @param node Pointer to the node to be printed.
 */
void print_node(Node *node);

/**
 * @brief Recursively serializes a node and its children to a binary file.
 * 
 * This function serializes the data of a node and its children to a binary file. 
 * A marker (-1) is used to indicate null pointers, which are later used for reconstruction 
 * during deserialization.
 * 
 * @param node Pointer to the node to be serialized.
 * @param fp File pointer to the binary file for serialization.
 */
void serialize_node(Node *node, FILE *fp);

/**
 * @brief Serializes a decision tree to a binary file.
 * 
 * This function serializes the entire tree to a binary file. The tree structure is 
 * recursively serialized using the `serialize_node` function.
 * 
 * @param tree Pointer to the tree to be serialized.
 * @param filename Path to the output binary file.
 */
void serialize_tree(Tree *tree, const char *filename);

/**
 * @brief Recursively deserializes a node and its children from a binary file.
 * 
 * This function reconstructs a node and its children by reading data from the binary file. 
 * It uses a marker to identify null nodes and handles the reconstruction of the tree structure.
 * 
 * @param fp File pointer to the binary file for deserialization.
 * @return Pointer to the reconstructed node.
 */
Node *deserialize_node(FILE *fp);

/**
 * @brief Deserializes a tree structure from a binary file.
 * 
 * This function deserializes a complete tree structure from a binary file and reconstructs 
 * the tree starting from the root node.
 * 
 * @param filename Path to the binary file to load the tree from.
 * @return Pointer to the reconstructed Tree structure.
 */
Tree *deserialize_tree(const char *filename);

/**
 * @brief Saves an array of predictions to a file, one per line.
 * 
 * This function saves the predicted class labels to a file, each label on a separate line.
 * 
 * @param predictions Array of predicted class labels.
 * @param num_rows Number of predictions.
 * @param filename Path to the output file.
 */
void save_predictions(const int *predictions, int num_rows, const char *filename);

#endif 
