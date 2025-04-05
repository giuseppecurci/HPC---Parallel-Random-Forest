#include "../headers/tree.h"
#include <stdlib.h>
#include <stdio.h>

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
    int *best_size_left = (int *)malloc(sizeof(int));
    *best_size_left = 0;
    int *best_size_right = (int *)malloc(sizeof(int));
    *best_size_right = 0;
    int *best_class_pred_left = (int *)malloc(sizeof(int));
    *best_class_pred_left = -1;
    int *best_class_pred_right = (int *)malloc(sizeof(int));
    *best_class_pred_right = -1;
    BestSplit best_split = find_best_split(data, parent->num_samples, num_columns, num_classes, 
                                           best_class_pred_left, best_class_pred_right, best_size_left, best_size_right);
    if (best_split.entropy > parent->entropy){
        free(best_size_left);
        free(best_size_right);
        free(best_class_pred_left);
        free(best_class_pred_right);
        return;
    }
    
    parent->feature = best_split.feature_index;
    parent->threshold = best_split.threshold;
    parent->entropy = best_split.entropy;
    // get_class_pred(data, parent->num_samples, num_columns, num_classes, parent);
    parent->left = create_node(-1, -1, NULL, NULL, *best_class_pred_left, parent->depth + 1, INFINITY, *best_size_left);
    parent->right = create_node(-1, -1, NULL, NULL, *best_class_pred_right, parent->depth + 1, INFINITY, *best_size_right);
    float** left_data = (float **)malloc(*best_size_left * sizeof(float *));
    float** right_data = (float **)malloc(*best_size_right * sizeof(float *));
    for (int i = 0; i < *best_size_left; i++) {
        left_data[i] = (float *)malloc(num_columns * sizeof(float));
    }
    for (int i = 0; i < *best_size_right; i++) {
        right_data[i] = (float *)malloc(num_columns * sizeof(float));
    }

    split_data(data, left_data, right_data, parent->num_samples, best_split.feature_index, best_split.threshold);
    
    grow_tree(parent->left, left_data, num_columns, num_classes);
    free(left_data);
    
    grow_tree(parent->right, right_data, num_columns, num_classes);
    free(right_data);

    free(best_size_left);
    free(best_size_right);
    free(best_class_pred_left);
    free(best_class_pred_right);
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

void get_class_pred(float** data, int num_rows, int num_columns, int num_classes, Node *node) {
    int classes_count[num_classes];
    for (int i = 0; i < num_rows; i++) {
        classes_count[(int)data[i][num_columns - 1]]++;
    }
    node->pred = argmax(classes_count, 3);
}

void destroy_tree(Tree *tree) {
    destroy_node(tree->root);
    free(tree);
};

void destroy_node(Node *node) {
    if (node == NULL) return;
    destroy_node(node->left);
    destroy_node(node->right);
    free(node);
};

void print_tree(Tree *tree) {
    printf("Printing tree\n");
    print_node(tree->root);
};

void print_node(Node *node) {
    if (node == NULL) return;
    printf("Feature: %d, Threshold: %.6f, Value: %d, Num samples: %d\n", node->feature, node->threshold, node->pred, node->num_samples);
    print_node(node->left);
    print_node(node->right);
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