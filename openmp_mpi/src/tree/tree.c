/**
 * @file tree.c
 * @brief Implementation of decision tree training, growing, and inference functions.
 *
 * This file contains functions to build and train a decision tree using recursive
 * splitting based on feature values. It includes functions for node creation,
 * growing the tree, training the tree, and performing inference on new data.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include "../../headers/tree/tree.h"
#include "../../headers/tree/train_utils.h"

// Node creation function (remains unchanged)
Node *create_node(int feature, float threshold, Node *left, Node *right, int pred, int depth, float entropy, int num_samples) {
    Node *node = (Node *)malloc(sizeof(Node));
    node->feature = feature;
    node->threshold = threshold;
    node->left = left;
    node->right = right;
    node->pred = pred;
    node->entropy = entropy;
    node->depth = depth;
    node->num_samples = num_samples;
    return node;
}

void grow_tree_1d(Node *parent, float *data, int num_columns, int num_classes, 
               int max_depth, int min_samples_split, char* max_features, int n_threads) {
    if (parent->num_samples < min_samples_split || parent->depth >= max_depth) {
        return;
    }
    
    int best_size_left = 0;
    int best_size_right = 0;
    int best_class_pred_left = -1;
    int best_class_pred_right = -1;
    
    BestSplit best_split = find_best_split_1d(data, parent->num_samples, num_columns, num_classes, 
                                           &best_class_pred_left, &best_class_pred_right, 
                                           &best_size_left, &best_size_right, max_features, n_threads);
    
    if (best_split.entropy >= parent->entropy) {
        return;
    }
    // Allocate memory for left and right datasets
    float *left_data = (float *)malloc(parent->num_samples * num_columns * sizeof(float));
    float *right_data = (float *)malloc(parent->num_samples * num_columns * sizeof(float));
 
	// qua c'è da fare lo split safe in caso
    
    int actual_left_size = 0, actual_right_size = 0;

    // Split the data
    split_data_1d_safe(data, left_data, right_data, parent->num_samples, num_columns, 
                best_split.feature_index, best_split.threshold, &actual_left_size, &actual_right_size);
    
    // Update parent node
    parent->feature = best_split.feature_index;
    parent->threshold = best_split.threshold;
    parent->entropy = best_split.entropy;
    
    // Create child nodes
    parent->left = create_node(-1, -1, NULL, NULL, best_class_pred_left, 
                             parent->depth + 1, INFINITY, best_size_left);
    parent->right = create_node(-1, -1, NULL, NULL, best_class_pred_right, 
                              parent->depth + 1, INFINITY, best_size_right);
    
    // Recursively grow the tree
    grow_tree_1d(parent->left, left_data, num_columns, num_classes, 
              max_depth, min_samples_split, max_features, n_threads);
    grow_tree_1d(parent->right, right_data, num_columns, num_classes, 
               max_depth, min_samples_split, max_features, n_threads);
    
    // Free memory
    free(left_data);
    free(right_data);
}

void split_data_1d(float *data, float *left_data, float *right_data, 
                  int num_rows, int num_columns, int feature_index, float threshold) {
    int actual_left_size, actual_right_size;
    split_data_1d_safe(data, left_data, right_data, num_rows, num_columns, 
                      feature_index, threshold, &actual_left_size, &actual_right_size);
}


// Refactored grow_tree function for 1D array data

// Refactored train_tree function for 1D array data
void train_tree_1d(Tree *tree, float *data, int num_rows, int num_columns, int num_classes, 
                int max_depth, int min_samples_split, char* max_features, int num_threads) {
    tree->root = create_node(-1, -1000, NULL, NULL, -1, 0, 1000, num_rows);
    grow_tree_1d(tree->root, data, num_columns, num_classes, max_depth, min_samples_split, max_features, num_threads);
}

// Refactored tree_inference function for 1D array data
int* tree_inference_1d(Tree *tree, float *data, int num_rows, int num_columns) {
    int *predictions = (int *)malloc(num_rows * sizeof(int));
    if (!predictions) {
        fprintf(stderr, "Memory allocation failed in tree_inference_1d!\n");
        exit(EXIT_FAILURE);
    }
   
    for (int i = 0; i < num_rows; i++) {
        Node *current_node = tree->root;
        while (current_node->left != NULL && current_node->right != NULL) {
            int feature_pos = i * num_columns + current_node->feature;
            if (data[feature_pos] <= current_node->threshold) {
                current_node = current_node->left;
            } else {
                current_node = current_node->right;
            }
        }
        predictions[i] = current_node->pred;
    }
    
    return predictions;
}

void split_data_1d_safe(float *data, float *left_data, float *right_data, 
                       int num_rows, int num_columns, int feature_index, float threshold,
                       int *actual_left_size, int *actual_right_size) {
    int left_index = 0;
    int right_index = 0;
    
    for (int i = 0; i < num_rows; i++) {
        // Calculate position of the feature value in the 1D array
        int row_start = i * num_columns;
        int feature_pos = row_start + feature_index;
        
        if (data[feature_pos] <= threshold) {
            // Copy entire row to left_data
            for (int j = 0; j < num_columns; j++) {
                left_data[left_index * num_columns + j] = data[row_start + j];
            }
            left_index++;
        } else {
            // Copy entire row to right_data
            for (int j = 0; j < num_columns; j++) {
                right_data[right_index * num_columns + j] = data[row_start + j];
            }
            right_index++;
        }
    }
    
    // Return actual sizes
    *actual_left_size = left_index;
    *actual_right_size = right_index;
}

