/**
 * @file tree.h
 * @brief Header file defining decision tree structures and functions.
 * 
 * This file provides the data structures and functions needed to create,
 * train, and use decision trees for classification tasks. It includes
 * node structures, tree structures, and algorithms for finding optimal
 * splits and growing trees.
 */
#ifndef TREE_H
#define TREE_H

// Forward declaration of Tree struct
typedef struct Tree Tree;

/**
 * @brief Node structure representing a decision point in a tree.
 * 
 * Each node contains either a decision rule (for internal nodes) or
 * a prediction value (for leaf nodes).
 */
typedef struct Node {
    int feature;         /**< Index of the feature used for splitting at this node */
    float threshold;     /**< Threshold value for the feature to make the split decision */
    struct Node *left;   /**< Pointer to the left child node (samples where feature <= threshold) */
    struct Node *right;  /**< Pointer to the right child node (samples where feature > threshold) */
    int pred;            /**< Predicted class value (only used in leaf nodes) */
    float entropy;       /**< Entropy value at this node */
    int depth;           /**< Depth of this node in the tree */
    int num_samples;     /**< Number of training samples that reached this node */
} Node;

/**
 * @brief Structure to store information about the best split found.
 * 
 * This structure stores the entropy, threshold, and feature index
 * for the best split discovered during the tree building process.
 */
typedef struct {
    float entropy;       /**< Entropy value of the best split */
    float threshold;     /**< Threshold value used for the best split */
    int feature_index;   /**< Index of the feature used for the best split */
} BestSplit;

/**
 * @brief Tree structure representing a decision tree.
 * 
 * The tree consists of a root node which connects to all other nodes.
 */
struct Tree {
    Node *root;          /**< Pointer to the root node of the tree */
};

/**
 * @brief Creates a new tree node.
 * 
 * Allocates memory for a new node and initializes it with the provided values.
 * 
 * @param feature Index of the feature used for splitting at this node.
 * @param threshold Threshold value for the feature to make the split decision.
 * @param left Pointer to the left child node.
 * @param right Pointer to the right child node.
 * @param pred Predicted class value (only used in leaf nodes).
 * @param depth Depth of this node in the tree.
 * @param entropy Entropy value at this node.
 * @param num_samples Number of training samples that reached this node.
 * @return A pointer to the newly created node.
 */
Node* create_node(int feature, float threshold, Node *left, Node *right, int pred, int depth, float entropy, int num_samples);
/**
 * @brief Splits the data into left and right subsets based on a feature and threshold.
 * 
 * This function divides the input data into two subsets based on the specified
 * feature and threshold.
 * 
 * @param data The input dataset as a float array.
 * @param left_data Preallocated array to store samples going to the left split.
 * @param right_data Preallocated array to store samples going to the right split.
 * @param num_rows Number of samples in the dataset.
 * @param num_columns Number of features in the dataset (including the label).
 * @param feature_index Index of the feature to use for splitting.
 * @param threshold Threshold value for the feature to make the split decision.
 */
void split_data_1d(float *data, float *left_data, float *right_data, 
                  int num_rows, int num_columns, int feature_index, float threshold);

/**
 * @brief Recursively grows a decision tree from a node.
 * 
 * This function builds a decision tree by recursively finding the best splits
 * and creating child nodes until stopping criteria are met.
 * 
 * @param parent The current node to grow from.
 * @param data The training data for this node.
 * @param num_columns Number of features in the dataset (including the label).
 * @param num_classes Number of unique classes in the dataset.
 * @param max_depth Maximum allowed depth for the tree.
 * @param min_samples_split Minimum number of samples required to consider a split.
 * @param max_features Strategy for selecting features to consider for splitting.
 */
void grow_tree_1d(Node *parent, float *data, int num_columns, int num_classes, 
                 int max_depth, int min_samples_split, char* max_features, int num_threads);

/**
 * @brief Trains a decision tree on the provided dataset.
 * 
 * This function initializes a tree and trains it on the given data by
 * creating a root node and growing the tree from there.
 * 
 * @param tree Pointer to the tree structure to be trained.
 * @param data The training dataset as a float array.
 * @param num_rows Number of samples in the dataset.
 * @param num_columns Number of features in the dataset (including the label).
 * @param num_classes Number of unique classes in the dataset.
 * @param max_depth Maximum allowed depth for the tree.
 * @param min_samples_split Minimum number of samples required to consider a split.
 * @param max_features Strategy for selecting features to consider for splitting.
 */
void train_tree_1d(Tree *tree, float *data, int num_rows, int num_columns, 
                  int num_classes, int max_depth, int min_samples_split, char* max_features, int num_threads);

/**
 * @brief Uses a trained tree to make predictions on a dataset.
 * 
 * This function applies the decision rules in the tree to classify
 * each sample in the provided dataset.
 * 
 * @param tree Pointer to the trained tree structure.
 * @param data The dataset to make predictions on.
 * @param num_rows Number of samples in the dataset.
 * @param num_columns Number of features in the dataset.
 * @return An array of predicted class labels for each input sample.
 */
int* tree_inference_1d(Tree *tree, float *data, int num_rows, int num_columns);

/**
 * @brief Trains a decision tree using MPI for parallel processing.
 * 
 * This function is designed for distributed training of a decision tree
 * using the Message Passing Interface (MPI) for parallelization.
 * 
 * @param tree Pointer to the tree structure to be trained.
 * @param train_data The training dataset as a float array.
 * @param sample_size Number of samples in the training dataset.
 * @param num_columns Number of features in the dataset (including the label).
 * @param num_classes Number of unique classes in the dataset.
 * @param max_depth Maximum allowed depth for the tree.
 * @param min_samples_split Minimum number of samples required to consider a split.
 * @param max_features Strategy for selecting features to consider for splitting.
 */
void mpi_train_tree(Tree *tree, float *train_data, int sample_size, int num_columns, 
                   int num_classes, int max_depth, int min_samples_split, char* max_features);
void split_data_1d_safe(float *data, float *left_data, float *right_data, 
                       int num_rows, int num_columns, int feature_index, float threshold,
                       int *actual_left_size, int *actual_right_size);

#endif // TREE_H
