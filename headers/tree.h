#ifndef TREE_H  
#define TREE_H  

#include "headers/utils.h"

#define MAX_DEPTH 100
#define MIN_SAMPLES_SPLIT 10

typedef struct Node {
    int feature;
    float threshold;
    struct Node *left;
    struct Node *right;
    float value;
} Node;

typedef struct Tree {
    Node *root;
    int max_depth;
    int min_samples_split;
} Tree;

Node *create_node(int feature, float threshold, Node *left, Node *right, float value);
Tree *create_tree(int max_depth, int min_samples_split);
void destroy_tree(Tree *tree);
void destroy_node(Node *node);
void print_tree(Tree *tree);
void print_node(Node *node);
int predict(Tree *tree, float *sample);

#endif 
