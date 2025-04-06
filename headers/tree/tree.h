/**
 * @file tree.h
 * @brief Header file for the decision tree structure and functions.
 *
 * This file defines the structures and function prototypes for building,
 * training, and performing inference with a decision tree.
 */

#ifndef TREE_H
#define TREE_H

// Maximum depth for the tree
#define MAX_DEPTH 100

// Minimum number of samples required to split a node
#define MIN_SAMPLES_SPLIT 15

/**
 * @struct Node
 * @brief Represents a node in the decision tree.
 *
 * This structure holds information about a single node in the decision tree,
 * including the feature used for splitting, the threshold value for the split,
 * the child nodes, and the prediction at the leaf node.
 */
typedef struct Node {
    int feature;          /**< The feature index used for splitting. */
    float threshold;      /**< The threshold for splitting the feature. */
    struct Node *left;    /**< Pointer to the left child node. */
    struct Node *right;   /**< Pointer to the right child node. */
    int pred;             /**< Prediction value for the leaf node. */
    float entropy;        /**< The entropy of the node. */
    int depth;            /**< The depth of the node in the tree. */
    int num_samples;      /**< The number of samples at this node. */
} Node;


/**
 * @struct Tree
 * @brief Represents the entire decision tree.
 *
 * This structure contains the root node of the tree and represents the complete
 * decision tree model, with functions to grow the tree and make predictions.
 */
typedef struct Tree {
    Node *root;           /**< Root node of the tree. */
} Tree;


/**
 * @brief Creates a new tree node with the given parameters.
 *
 * This function allocates memory for a new `Node` and initializes it with the
 * provided feature, threshold, and other attributes. It returns a pointer to
 * the created node.
 *
 * @param feature      The feature index used for splitting the data.
 * @param threshold    The threshold value for the split.
 * @param left         Pointer to the left child node.
 * @param right        Pointer to the right child node.
 * @param pred         The prediction value at the node (used at leaf nodes).
 * @param depth        The depth of the node in the tree.
 * @param entropy      The entropy at the node.
 * @param num_samples  The number of samples at this node.
 * @return Pointer to the newly created `Node`.
 */
Node *create_node(int feature, float threshold, Node *left, Node *right, 
                  int pred, int depth, float entropy, int num_samples);

/**
 * @brief Recursively grows the decision tree by splitting data.
 *
 * This function performs a recursive depth-first search to build the tree by
 * splitting the data (deep-copy) based on the best feature and threshold. It stops if
 * the number of samples is below a minimum threshold or if the maximum depth
 * is reached. The function continues splitting until leaf nodes are created.
 *
 * @param parent        The parent node to which the left and right child nodes are added.
 * @param data          The data used for growing the tree.
 * @param num_columns   The number of columns (features) in the dataset.
 * @param num_classes   The number of distinct classes in the dataset.
 */
void grow_tree(Node *parent, float **data, int num_columns, int num_classes);

/**
 * @brief Trains a decision tree using the provided data.
 *
 * This function initializes the tree's root node and starts the tree-growing
 * process using the `grow_tree` function. It outputs basic information about
 * the training process and the tree's parameters.
 *
 * @param tree          Pointer to the `Tree` structure to be trained.
 * @param data          The training data to use for training the tree.
 * @param num_rows      The number of data samples.
 * @param num_columns   The number of features per data sample.
 * @param num_classes   The number of possible output classes.
 */
void train_tree(Tree *tree, float **data, int num_rows, int num_columns, int num_classes);

/**
 * @brief Performs inference on a set of data using the trained decision tree.
 *
 * This function traverses the decision tree for each data sample and returns
 * the predicted class label based on the leaf node's prediction. It outputs
 * an array of predictions for all input samples.
 *
 * @param tree          The trained decision tree used for inference.
 * @param data          The data samples to be predicted.
 * @param num_rows      The number of data samples.
 * @return Array of predicted class labels, one for each input sample.
 */
int* tree_inference(Tree *tree, float **data, int num_rows);

#endif 