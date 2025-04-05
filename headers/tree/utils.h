#ifndef UTILS_TREE_H
#define UTILS_TREE_H

#include "tree.h"
#include <stdio.h>

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

void serialize_node(Node *node, FILE *fp);

void serialize_tree(Tree *tree, const char *filename);

Node *deserialize_node(FILE *fp);

Tree *deserialize_tree(const char *filename);

void save_predictions(const int *predictions, int num_rows, const char *filename);

#endif 