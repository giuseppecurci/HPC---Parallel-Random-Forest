/**
 * @file tree_utils.c
 * @brief Utility functions for managing and working with decision trees.
 *
 * This file includes functions for making predictions at the leaf level,
 * freeing memory for trees and nodes, printing the tree structure, saving
 * prediction results to file, and serializing/deserializing trees for storage.
 */

#include <stdlib.h>
#include <stdio.h>
#include "../headers/tree/utils.h"
#include "../headers/tree/tree.h"
#include "../headers/tree/train_utils.h"

/**
 * @brief Assigns the most frequent class label in the current node's data as its prediction.
 *        Not used in the current implementation.
 * @param data         2D array of float data (last column is the class label).
 * @param num_rows     Number of data samples.
 * @param num_columns  Number of columns in each data sample (features + 1 label).
 * @param num_classes  Total number of classes.
 * @param node         Pointer to the node where prediction will be stored.
 */
void get_class_pred(float** data, int num_rows, int num_columns, int num_classes, Node *node) {
    int classes_count[num_classes];
    for (int i = 0; i < num_rows; i++) {
        classes_count[(int)data[i][num_columns - 1]]++;
    }
    node->pred = argmax(classes_count, 3);
}

/**
 * @brief Recursively frees memory allocated for the nodes and the tree itself.
 */
void destroy_tree(Tree *tree) {
    destroy_node(tree->root);
    free(tree);
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

/**
 * @brief Prints the structure and contents of a tree.
 */
void print_tree(Tree *tree) {
    printf("Printing tree\n");
    print_node(tree->root);
};

/**
 * @brief Recursively prints node information (feature, threshold, prediction, etc.).
 */
void print_node(Node *node) {
    if (node == NULL) return;
    printf("Feature: %d, Threshold: %.6f, Value: %d, Num samples: %d\n", node->feature, node->threshold, node->pred, node->num_samples);
    print_node(node->left);
    print_node(node->right);
};

/**
 * @brief Saves an array of predictions to a file, one per line.
 *
 * @param predictions  Array of predicted class labels.
 * @param num_rows     Number of predictions.
 * @param filename     Path to the output file.
 */
void save_predictions(const int *predictions, int num_rows, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening file to save predictions");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "Predictions\n");
    for (int i = 0; i < num_rows; i++) {
        fprintf(fp, "%d\n", predictions[i]);
    }

    fclose(fp);
}

/**
 * @brief Recursively serializes a node and its children to a binary file.
 *
 * Uses a marker to indicate null pointers (-1) for reconstruction during deserialization.
 */
void serialize_node(Node *node, FILE *fp) {
    if (node == NULL) {
        int marker = -1;
        fwrite(&marker, sizeof(int), 1, fp);
        return;
    }

    int marker = 1;
    fwrite(&marker, sizeof(int), 1, fp);
    fwrite(&node->feature, sizeof(int), 1, fp);
    fwrite(&node->threshold, sizeof(float), 1, fp);
    fwrite(&node->pred, sizeof(int), 1, fp);
    fwrite(&node->entropy, sizeof(float), 1, fp);
    fwrite(&node->depth, sizeof(int), 1, fp);
    fwrite(&node->num_samples, sizeof(int), 1, fp);

    serialize_node(node->left, fp);
    serialize_node(node->right, fp);
}

/**
 * @brief Serializes a decision tree to a binary file.
 */
void serialize_tree(Tree *tree, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error opening file to save tree");
        exit(EXIT_FAILURE);
    }

    serialize_node(tree->root, fp);
    fclose(fp);
}

/**
 * @brief Recursively deserializes a node and its children from a binary file.
 *
 * @return Pointer to the reconstructed node.
 */
Node *deserialize_node(FILE *fp) {
    int marker;
    if (fread(&marker, sizeof(int), 1, fp) != 1)
        return NULL;

    if (marker == -1)
        return NULL;

    Node *node = malloc(sizeof(Node));
    if (!node) {
        perror("Error allocating memory for node");
        exit(EXIT_FAILURE);
    }

    fread(&node->feature, sizeof(int), 1, fp);
    fread(&node->threshold, sizeof(float), 1, fp);
    fread(&node->pred, sizeof(int), 1, fp);
    fread(&node->entropy, sizeof(float), 1, fp);
    fread(&node->depth, sizeof(int), 1, fp);
    fread(&node->num_samples, sizeof(int), 1, fp);

    node->left = deserialize_node(fp);
    node->right = deserialize_node(fp);

    return node;
}

/**
 * @brief Deserializes a tree structure from a binary file.
 *
 * @param filename  Path to the binary file.
 * @return Pointer to the reconstructed Tree structure.
 */
Tree *deserialize_tree(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file to load tree");
        exit(EXIT_FAILURE);
    }

    Tree *tree = malloc(sizeof(Tree));
    if (!tree) {
        perror("Error allocating memory for tree");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    tree->root = deserialize_node(fp);
    fclose(fp);
    return tree;
}