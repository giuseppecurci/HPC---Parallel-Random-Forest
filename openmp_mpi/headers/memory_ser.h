#ifndef FOREST_BUFFER_SERIALIZATION_H
#define FOREST_BUFFER_SERIALIZATION_H

#include <string.h>
#include "forest.h"

/**
 * Serializes a forest structure to a memory buffer
 * 
 * @param forest The forest to serialize
 * @param out_buffer Pointer to buffer where serialized data will be stored (will be malloc'd)
 * @param out_size Pointer to store the total size of the serialized data
 */
void serialize_forest_to_buffer(const Forest* forest, void** out_buffer, int* out_size);

/**
 * Deserializes a forest structure from a memory buffer
 * 
 * @param buffer The buffer containing serialized forest data
 * @param forest Pointer to forest structure where data will be loaded
 */
void deserialize_forest_from_buffer(const void* buffer, Forest* forest);

/**
 * Deserializes a forest structure from a memory buffer and returns a new forest
 * 
 * @param buffer The buffer containing serialized forest data
 * @return A newly allocated Forest structure initialized from the buffer
 */
Forest* deserialize_forest_from_buffer_new(const void* buffer);

/**
 * @brief Receives forest data from other processes in a distributed environment.
 * 
 * This function collects serialized tree data from multiple processes and 
 * deserializes them into a complete forest structure.
 * 
 * @param num_trees_received Pointer to store the total number of trees received.
 * @return A dynamically allocated array of Tree pointers forming the complete forest.
 */
Tree **receive_forest(int *num_trees_received);

/**
 * @brief Distributes forest data across multiple processes.
 * 
 * This function splits a forest among multiple processes for distributed 
 * inference or evaluation. Each process receives a portion of the forest 
 * appropriate to its computational resources.
 * 
 * @param forest Array of Tree pointers representing the complete random forest.
 * @param num_trees Total number of trees in the forest to distribute.
 * @param process_number Number of processes to distribute the forest across.
 */
void distribute_forest(Tree **forest, int num_trees, int process_number);
#endif /* FOREST_BUFFER_SERIALIZATION_H */
