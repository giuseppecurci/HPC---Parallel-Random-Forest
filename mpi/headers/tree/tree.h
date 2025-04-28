#ifndef TREE_H
#define TREE_H

// Forward declaration of Tree struct
typedef struct Tree Tree;

typedef struct Node {
    int feature;
    float threshold;
    struct Node *left;
    struct Node *right;
    int pred;
    float entropy;
    int depth;
    int num_samples;
} Node;

typedef struct {
    float entropy;
    float threshold;
    int feature_index;
} BestSplit;

// Define Tree struct
struct Tree {
    Node *root;
};

// Function declarations
Node* create_node(int feature, float threshold, Node *left, Node *right, int pred, int depth, float entropy, int num_samples);
BestSplit find_best_split_1d(float *data, int num_rows, int num_columns, int num_classes, int *class_pred_left, int *class_pred_right, int *best_size_left, int *best_size_right, char *max_features);
void split_data_1d(float *data, float *left_data, float *right_data, int num_rows, int num_columns, int feature_index, float threshold);
void grow_tree_1d(Node *parent, float *data, int num_columns, int num_classes, int max_depth, int min_samples_split, char* max_features);
void train_tree_1d(Tree *tree, float *data, int num_rows, int num_columns, int num_classes, int max_depth, int min_samples_split, char* max_features);
int* tree_inference_1d(Tree *tree, float *data, int num_rows, int num_columns);
void mpi_train_tree(Tree *tree, float *train_data, int sample_size, int num_columns, int num_classes, int max_depth, int min_samples_split, char* max_features);

#endif // TREE_H
