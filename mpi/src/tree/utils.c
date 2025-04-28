#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../headers/utils.h"
#include "../../headers/tree/utils.h"
#include "../../headers/tree/tree.h"
#include "../../headers/tree/train_utils.h"

/**
 * @file tree_utils.c
 * @brief Utility functions for managing and working with decision trees.
 *
 * This file includes functions for making predictions at the leaf level,
 * freeing memory for trees and nodes, printing the tree structure, saving
 * prediction results to file, and serializing/deserializing trees for storage.
 */

/**
 * @brief Recursively frees memory allocated for the nodes and the tree itself.
 */
void destroy_tree(Tree *tree) {
    destroy_node(tree->root);
};

/**
 * @brief Recursively frees memory allocated for a node and its children.
 */
void destroy_node(Node *node) {
    if (node == NULL) return;
    destroy_node(node->left);
    destroy_node(node->right);
    free(node);
};
