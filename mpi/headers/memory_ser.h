#ifndef FOREST_BUFFER_SERIALIZATION_H
#define FOREST_BUFFER_SERIALIZATION_H

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

Tree **receive_forest(int *num_trees_received);
void distribute_forest(Tree **forest, int num_trees, int process_number);
#endif /* FOREST_BUFFER_SERIALIZATION_H */
