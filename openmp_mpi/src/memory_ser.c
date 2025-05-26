#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "../headers/tree/tree.h"
#include "../headers/forest.h"

// Import declarations from tree serialization
void serialize_tree_to_buffer(const struct Tree* tree, void** out_buffer, int* out_size);
void deserialize_tree_from_buffer(const void* buffer, struct Tree* tree);

/**
 * Serializes a forest structure to a memory buffer
 * 
 * @param forest The forest to serialize
 * @param out_buffer Pointer to buffer where serialized data will be stored (will be malloc'd)
 * @param out_size Pointer to store the total size of the serialized data
 */
void serialize_forest_to_buffer(const Forest* forest, void** out_buffer, int* out_size) {
    // First calculate the total size needed
    int header_size = sizeof(int) * 3 + sizeof(char) * (strlen(forest->max_features) + 1);
    
    // Reserve space for tree offsets
    int tree_offsets_size = sizeof(int) * forest->num_trees;
    
    // Calculate total size and create buffer
    int total_size = header_size + tree_offsets_size;
    
    // First, serialize all trees to separate buffers to calculate sizes
    void** tree_buffers = (void**)malloc(forest->num_trees * sizeof(void*));
    int* tree_sizes = (int*)malloc(forest->num_trees * sizeof(int));
    
    for (int i = 0; i < forest->num_trees; i++) {
        serialize_tree_to_buffer(&forest->trees[i], &tree_buffers[i], &tree_sizes[i]);
        total_size += tree_sizes[i];
    }
    
    // Allocate the complete buffer
    uint8_t* buffer = (uint8_t*)malloc(total_size);
    int offset = 0;
    
    // Write forest metadata
    memcpy(buffer + offset, &forest->num_trees, sizeof(int));
    offset += sizeof(int);
    
    memcpy(buffer + offset, &forest->max_depth, sizeof(int));
    offset += sizeof(int);
    
    memcpy(buffer + offset, &forest->min_samples_split, sizeof(int));
    offset += sizeof(int);
    
    // Write max_features string including null terminator
    int max_features_len = strlen(forest->max_features) + 1;
    memcpy(buffer + offset, forest->max_features, max_features_len);
    offset += max_features_len;
    
    // Write tree offsets (where each tree's data begins in the buffer)
    int current_tree_offset = offset + tree_offsets_size;
    for (int i = 0; i < forest->num_trees; i++) {
        memcpy(buffer + offset, &current_tree_offset, sizeof(int));
        offset += sizeof(int);
        current_tree_offset += tree_sizes[i];
    }
    
    // Write tree data
    for (int i = 0; i < forest->num_trees; i++) {
        memcpy(buffer + offset, tree_buffers[i], tree_sizes[i]);
        offset += tree_sizes[i];
        
        // Free the individual tree buffer
        free(tree_buffers[i]);
    }
    
    // Free temporary arrays
    free(tree_buffers);
    free(tree_sizes);
    
    // Set output parameters
    *out_buffer = buffer;
    *out_size = total_size;
}

/**
 * Deserializes a forest structure from a memory buffer
 * 
 * @param buffer The buffer containing serialized forest data
 * @param forest Pointer to forest structure where data will be loaded
 */
void deserialize_forest_from_buffer(const void* buffer, Forest* forest) {
    const uint8_t* buf = (const uint8_t*)buffer;
    int offset = 0;
    
    // Read forest metadata
    memcpy(&forest->num_trees, buf + offset, sizeof(int));
    offset += sizeof(int);
    
    memcpy(&forest->max_depth, buf + offset, sizeof(int));
    offset += sizeof(int);
    
    memcpy(&forest->min_samples_split, buf + offset, sizeof(int));
    offset += sizeof(int);
    
    // Read max_features string
    forest->max_features = strdup((const char*)(buf + offset));
    offset += strlen(forest->max_features) + 1;
    
    // Allocate memory for trees
    forest->trees = (Tree*)malloc(forest->num_trees * sizeof(Tree));
    
    // Read tree offsets
    int* tree_offsets = (int*)malloc(forest->num_trees * sizeof(int));
    for (int i = 0; i < forest->num_trees; i++) {
        memcpy(&tree_offsets[i], buf + offset, sizeof(int));
        offset += sizeof(int);
    }
    
    // Deserialize each tree
    for (int i = 0; i < forest->num_trees; i++) {
        deserialize_tree_from_buffer(buf + tree_offsets[i], &forest->trees[i]);
    }
    
    // Free temporary array
    free(tree_offsets);
}

/**
 * Deserializes a forest structure from a memory buffer and returns a new forest
 * 
 * @param buffer The buffer containing serialized forest data
 * @return A newly allocated Forest structure initialized from the buffer
 */
Forest* deserialize_forest_from_buffer_new(const void* buffer) {
    Forest* forest = (Forest*)malloc(sizeof(Forest));
    deserialize_forest_from_buffer(buffer, forest);
    return forest;
}
void distribute_forest(Tree **forest, int num_trees, int process_number) {
    int base_trees = num_trees / (process_number - 1);
    int extra_trees = num_trees % (process_number - 1);
    int offset = 0;

    for (int p = 1; p < process_number; p++) {
        int trees_for_process = base_trees + (p <= extra_trees ? 1 : 0);

        // Send number of trees
        MPI_Send(&trees_for_process, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

        for (int i = 0; i < trees_for_process; i++) {
            void *buffer;
            int buffer_size;
            serialize_tree_to_buffer(forest[offset + i], &buffer, &buffer_size);

            // Send buffer size and buffer
            MPI_Send(&buffer_size, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
            MPI_Send(buffer, buffer_size, MPI_BYTE, p, 2, MPI_COMM_WORLD);

            free(buffer);
        }

        offset += trees_for_process;
    }
}

Tree **receive_forest(int *num_trees_received) {
    MPI_Recv(num_trees_received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    Tree **forest = (Tree **)malloc(*num_trees_received * sizeof(Tree *));
    if (!forest) {
        perror("malloc failed for forest");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < *num_trees_received; i++) {
        int buffer_size;
        void *buffer;

        MPI_Recv(&buffer_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        buffer = malloc(buffer_size);
        if (!buffer) {
            perror("malloc failed for buffer");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Recv(buffer, buffer_size, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        forest[i] = (Tree *)malloc(sizeof(Tree));
        if (!forest[i]) {
            perror("malloc failed for tree");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        deserialize_tree_from_buffer(buffer, forest[i]);
        free(buffer);
    }

    return forest;
}

