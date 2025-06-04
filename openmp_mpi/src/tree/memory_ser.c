#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include "../../headers/tree/tree.h" 

// Recursively count number of nodes
int count_nodes(const Node* node) {
    if (!node) return 0;
    return 1 + count_nodes(node->left) + count_nodes(node->right);
}

// Recursively serialize a node into the buffer
void flatten_node(const Node* node, uint8_t* buffer, int* offset) {
    memcpy(buffer + *offset, &node->feature, sizeof(int));
    *offset += sizeof(int);

    memcpy(buffer + *offset, &node->threshold, sizeof(float));
    *offset += sizeof(float);

    memcpy(buffer + *offset, &node->pred, sizeof(int));
    *offset += sizeof(int);

    memcpy(buffer + *offset, &node->entropy, sizeof(float));
    *offset += sizeof(float);

    memcpy(buffer + *offset, &node->depth, sizeof(int));
    *offset += sizeof(int);

    memcpy(buffer + *offset, &node->num_samples, sizeof(int));
    *offset += sizeof(int);

    int has_left = (node->left != NULL);
    int has_right = (node->right != NULL);
    memcpy(buffer + *offset, &has_left, sizeof(int));
    *offset += sizeof(int);
    memcpy(buffer + *offset, &has_right, sizeof(int));
    *offset += sizeof(int);

    if (has_left) flatten_node(node->left, buffer, offset);
    if (has_right) flatten_node(node->right, buffer, offset);
}

void serialize_tree_to_buffer(const struct Tree* tree, void** out_buffer, int* out_size) {
    int total_nodes = count_nodes(tree->root);
    int size_per_node = sizeof(int) * 6 + sizeof(float) * 2;  // 6 ints + 2 floats per node
    int total_size = total_nodes * size_per_node;

    uint8_t* buffer = (uint8_t*)malloc(total_size);
    int offset = 0;
    flatten_node(tree->root, buffer, &offset);

    *out_buffer = buffer;
    *out_size = offset;
}
// Recursively rebuild node from buffer
Node* rebuild_node(const uint8_t* buffer, int* offset) {
    Node* node = (Node*)malloc(sizeof(Node));

    memcpy(&node->feature, buffer + *offset, sizeof(int));
    *offset += sizeof(int);

    memcpy(&node->threshold, buffer + *offset, sizeof(float));
    *offset += sizeof(float);

    memcpy(&node->pred, buffer + *offset, sizeof(int));
    *offset += sizeof(int);

    memcpy(&node->entropy, buffer + *offset, sizeof(float));
    *offset += sizeof(float);

    memcpy(&node->depth, buffer + *offset, sizeof(int));
    *offset += sizeof(int);

    memcpy(&node->num_samples, buffer + *offset, sizeof(int));
    *offset += sizeof(int);

    int has_left, has_right;
    memcpy(&has_left, buffer + *offset, sizeof(int));
    *offset += sizeof(int);
    memcpy(&has_right, buffer + *offset, sizeof(int));
    *offset += sizeof(int);

    node->left = has_left ? rebuild_node(buffer, offset) : NULL;
    node->right = has_right ? rebuild_node(buffer, offset) : NULL;

    return node;
}

void deserialize_tree_from_buffer(const void* buffer, struct Tree* tree) {
    int offset = 0;
    tree->root = rebuild_node((const uint8_t*)buffer, &offset);
}
