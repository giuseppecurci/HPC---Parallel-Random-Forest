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
#include "tree.h"
#include "train_utils.h"

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
};

void grow_tree(Node *parent, float **data, int num_columns, int num_classes) {
    if (parent->num_samples < MIN_SAMPLES_SPLIT || parent->depth >= MAX_DEPTH) {
        return;
    }
    
    int best_size_left = 0;
    int best_size_right = 0;
    int best_class_pred_left = -1;
    int best_class_pred_right = -1;
    BestSplit best_split = find_best_split(data, parent->num_samples, num_columns, num_classes, 
                                           &best_class_pred_left, &best_class_pred_right, 
                                           &best_size_left, &best_size_right);
    if (best_split.entropy > parent->entropy){
        return;
    }
    
    float** left_data = (float **)malloc(best_size_left * sizeof(float *));
    float** right_data = (float **)malloc(best_size_right * sizeof(float *));

    split_data(data, left_data, right_data, parent->num_samples, num_columns, best_split.feature_index, best_split.threshold);
    
    parent->feature = best_split.feature_index;
    parent->threshold = best_split.threshold;
    parent->entropy = best_split.entropy;
    parent->left = create_node(-1, -1, NULL, NULL, best_class_pred_left, parent->depth + 1, INFINITY, best_size_left);
    parent->right = create_node(-1, -1, NULL, NULL, best_class_pred_right, parent->depth + 1, INFINITY, best_size_right);

    grow_tree(parent->left, left_data, num_columns, num_classes);
    grow_tree(parent->right, right_data, num_columns, num_classes);
    for (int i = 0; i < best_size_left; i++) free(left_data[i]);
    free(left_data);
    for (int i = 0; i < best_size_right; i++) free(right_data[i]);
    free(right_data);

};

void train_tree(Tree *tree, float **data, int num_rows, int num_columns, int num_classes) {
    tree->root = create_node(-1, -1000, NULL, NULL, -1, 0, 1000, num_rows);
    printf("Starting tree training\n");
    printf("Number of rows: %d\n", num_rows);
    printf("Number of columns: %d\n", num_columns);
    printf("Number of classes: %d\n", num_classes);
    grow_tree(tree->root, data, num_columns, num_classes);
    printf("Tree trained\n");
};

int* tree_inference(Tree *tree, float **data, int num_rows) {
    int *predictions = (int *)malloc(num_rows * sizeof(int));
    for (int i = 0; i < num_rows; i++) {
        Node *current_node = tree->root;
        while (current_node->left != NULL && current_node->right != NULL) {
            if (data[i][current_node->feature] <= current_node->threshold) {
                current_node = current_node->left;
            } else {
                current_node = current_node->right;
            }
        }
        predictions[i] = current_node->pred;
    }
    return predictions;
}