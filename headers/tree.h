#ifndef TREE_H
#define TREE_H

#include "utils.h"

#define MAX_DEPTH 100
#define MIN_SAMPLES_SPLIT 10

// Node structure for the decision tree
typedef struct Node {
    int feature;          // The feature index used for splitting
    float threshold;      // The threshold for splitting
    struct Node *left;    // Pointer to left child node
    struct Node *right;   // Pointer to right child node
    int pred;             // Prediction value for leaf node
    float entropy;        // Entropy value of the node
    int depth;            // Depth of the node in the tree
    int num_samples;      // Number of samples at this node
} Node;

// Tree structure containing the root node and parameters for the tree
typedef struct Tree {
    Node *root;           // Root node of the tree
} Tree;

// Function to create a new node
Node *create_node(int feature, float threshold, Node *left, Node *right, 
                  int pred, int depth, float entropy, int num_samples);

// Function to grow the tree recursively
void grow_tree(Node *parent, float **data, int num_columns, int num_classes);

// Function to train the decision tree
void train_tree(Tree *tree, float **data, int num_rows, int num_columns, int num_classes);

// Function to get the predicted class for a node
void get_class_pred(float** data, int num_rows, int num_columns, int num_classes, Node *node);

// Function to destroy the tree and free allocated memory
void destroy_tree(Tree *tree);

// Function to recursively destroy nodes in the tree
void destroy_node(Node *node);

// Function to print the tree (for debugging purposes)
void print_tree(Tree *tree);

// Function to recursively print nodes in the tree
void print_node(Node *node);

#endif 