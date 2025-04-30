#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
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

// Refactored find_best_split function for 1D array data
BestSplit find_best_split_1d(float *data, int num_rows, int num_columns, 
                          int num_classes, int *class_pred_left, int *class_pred_right,
                          int *best_size_left, int *best_size_right, char *max_features) 
{
    BestSplit best_split = {INFINITY, 0.0, -1};
    int target_column = num_columns - 1;  // Assuming target column is the last one

    int features_to_consider = num_columns - 1; // Exclude target column
    int selected_features[features_to_consider]; // contains the indices of columns to consider
    int num_selected_features = 0;

    // Handle different max_features scenarios
    if (strcmp(max_features, "sqrt") == 0) {
        num_selected_features = (int) sqrt(features_to_consider);
    } else if (strcmp(max_features, "log2") == 0) {
        num_selected_features = (int) (log(features_to_consider) / log(2));
    } else {
        num_selected_features = atoi(max_features);
    }
    
    // Create a list of feature indices to consider    
    for (int i = 0; i < features_to_consider; i++) {
        selected_features[i] = i;
    }
    
    // Randomly shuffle all features 
    shuffle(selected_features, features_to_consider);

    // Loop over the first num_selected_features columns which were randomized
    for (int i = 0; i < num_selected_features; i++) {
        int feature_col = selected_features[i];

        if (feature_col == target_column){ 
            fprintf(stderr, "Error in function best_split you have selected the feature column\n");
            exit(EXIT_FAILURE);
        }
        
        // Allocate arrays for sorting
        float *feature_values = malloc(num_rows * sizeof(float));
        float *target_values = malloc(num_rows * sizeof(float));
        if (!feature_values || !target_values) {
            fprintf(stderr, "Memory allocation failed!\n");
            exit(EXIT_FAILURE);
        }

        // Extract feature column and corresponding target values from 1D array
        for (int j = 0; j < num_rows; j++) {
            feature_values[j] = data[j * num_columns + feature_col];
            target_values[j] = data[j * num_columns + target_column];
        }

        // Sort the feature and target values together
        merge_sort(feature_values, target_values, num_rows);

        // Find best split for this feature
        float *feature_best_split = get_best_split_num_var(feature_values, target_values, num_rows, num_classes);
        
        // Update the global best split if a lower entropy is found
        if (feature_best_split[0] < best_split.entropy) {
            best_split.entropy = feature_best_split[0];
            best_split.threshold = feature_best_split[1];
            *best_size_left = (int) feature_best_split[2];
            *best_size_right = (int) feature_best_split[3];
            *class_pred_left = (int) feature_best_split[4];
            *class_pred_right = (int) feature_best_split[5];
            best_split.feature_index = feature_col;
        }

        // Free allocated memory
        free(feature_best_split);
        free(feature_values);
        free(target_values);
    }

    return best_split;
}

// Refactored split_data function for 1D array data
void split_data_1d(float *data, float *left_data, float *right_data, 
                  int num_rows, int num_columns, int feature_index, float threshold) {
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
}

// Refactored grow_tree function for 1D array data
void grow_tree_1d(Node *parent, float *data, int num_columns, int num_classes, 
               int max_depth, int min_samples_split, char* max_features) {
    if (parent->num_samples < min_samples_split || parent->depth >= max_depth) {
        return;
    }
    
    int best_size_left = 0;
    int best_size_right = 0;
    int best_class_pred_left = -1;
    int best_class_pred_right = -1;
    
    BestSplit best_split = find_best_split_1d(data, parent->num_samples, num_columns, num_classes, 
                                           &best_class_pred_left, &best_class_pred_right, 
                                           &best_size_left, &best_size_right, max_features);
    
    if (best_split.entropy >= parent->entropy) {
        return;
    }
    
    // Allocate memory for left and right datasets
    float *left_data = (float *)malloc(best_size_left * num_columns * sizeof(float));
    float *right_data = (float *)malloc(best_size_right * num_columns * sizeof(float));
    
    if (!left_data || !right_data) {
        fprintf(stderr, "Memory allocation failed in grow_tree_1d!\n");
        free(left_data);
        free(right_data);
        exit(EXIT_FAILURE);
    }
    
    // Split the data
    split_data_1d(data, left_data, right_data, parent->num_samples, num_columns, 
                best_split.feature_index, best_split.threshold);
    
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
              max_depth, min_samples_split, max_features);
    grow_tree_1d(parent->right, right_data, num_columns, num_classes, 
               max_depth, min_samples_split, max_features);
    
    // Free memory
    free(left_data);
    free(right_data);
}

// Refactored train_tree function for 1D array data
void train_tree_1d(Tree *tree, float *data, int num_rows, int num_columns, int num_classes, 
                int max_depth, int min_samples_split, char* max_features) {
    tree->root = create_node(-1, -1000, NULL, NULL, -1, 0, 1000, num_rows);
    grow_tree_1d(tree->root, data, num_columns, num_classes, max_depth, min_samples_split, max_features);
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

