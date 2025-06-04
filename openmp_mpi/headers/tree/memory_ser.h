/**
 * @file tree_serialization.h
 * @brief Functions for serializing and deserializing tree structures for MPI communication
 */

#ifndef TREE_SERIALIZATION_H
#define TREE_SERIALIZATION_H
#include <string.h>

#include "tree.h"
#include <mpi.h>
#include <stdint.h>

/**
 * @brief Recursively counts the number of nodes in a tree
 * 
 * @param node The root node to start counting from
 * @return int The total number of nodes in the tree
 */
int count_nodes(const Node* node);

/**
 * @brief Recursively serializes a node and its children into a buffer
 * 
 * @param node The node to serialize
 * @param buffer The buffer to write serialized data into
 * @param offset Pointer to the current offset in the buffer (will be updated)
 */
void flatten_node(const Node* node, uint8_t* buffer, int* offset);

/**
 * @brief Serializes an entire tree structure into a buffer
 * 
 * @param tree The tree to serialize
 * @param out_buffer Pointer to where the allocated buffer should be stored
 * @param out_size Pointer to where the size of the buffer should be stored
 * 
 * @note The caller is responsible for freeing the allocated buffer
 */
void serialize_tree_to_buffer(const struct Tree* tree, void** out_buffer, int* out_size);

/**
 * @brief Recursively rebuilds a node from a serialized buffer
 * 
 * @param buffer The buffer containing serialized data
 * @param offset Pointer to the current offset in the buffer (will be updated)
 * @return Node* The reconstructed node
 */
Node* rebuild_node(const uint8_t* buffer, int* offset);

/**
 * @brief Deserializes a buffer into a tree structure
 * 
 * @param buffer The buffer containing serialized tree data
 * @param tree Pointer to the tree structure to populate
 */
void deserialize_tree_from_buffer(const void* buffer, struct Tree* tree);

#endif /* TREE_SERIALIZATION_H */
