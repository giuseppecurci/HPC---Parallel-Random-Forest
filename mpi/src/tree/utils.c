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
#include <dirent.h>    
#include <string.h>     
#include "../../headers/tree/utils.h"
#include "../../headers/tree/tree.h"
#include "../../headers/tree/train_utils.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
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

void check_dir_existence(const char *new_forest_path) {
    struct stat st = {0};
    char parent_dir[256] = {0};
    char *slash;
    
    if (new_forest_path == NULL) {
        printf("Error: Path is NULL\n");
    }
    
    strncpy(parent_dir, new_forest_path, sizeof(parent_dir) - 1);
    slash = strrchr(parent_dir, '/');
    if (slash) {
        *slash = '\0';  // Terminate the string at the last slash
        if (stat(parent_dir, &st) == -1) {
            if (mkdir(parent_dir, 0700) == -1) {
                perror("Parent directory creation failed");
            }
            printf("Parent directory created: %s\n", parent_dir);
        }
    }
    
    // Now try to create the full path
    if (stat(new_forest_path, &st) == -1) {
        if (mkdir(new_forest_path, 0700) == 0) {
            printf("Directory created: %s\n", new_forest_path);
        } else {
            perror("mkdir failed");
        }
    }
}

int check_bin_files_exist(const char *directory_path) {
    DIR *dir;
    struct dirent *entry;
    int found_bin_files = 0;
    char *extension;
    
    // Check if the directory path is valid
    if (directory_path == NULL) {
        fprintf(stderr, "Error: Directory path is NULL\n");
        return -1;
    }
    
    // Open the directory
    dir = opendir(directory_path);
    if (dir == NULL) {
        perror("Error opening directory");
        return -1;
    }
    
    // Read the directory entries
    while ((entry = readdir(dir)) != NULL) {
        // Skip '.' and '..' directories
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Check for .bin extension
        extension = strrchr(entry->d_name, '.');
        if (extension != NULL && strcmp(extension, ".bin") == 0) {
            found_bin_files = 1;
            break;
        }
    }
    
    // Close the directory
    closedir(dir);
    
    return found_bin_files;
}
