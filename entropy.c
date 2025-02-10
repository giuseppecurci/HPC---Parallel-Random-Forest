#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Structure to hold the count of samples for each class node
// There is one per node, each one has to keep track of 
typedef struct {
    int* counts;         // A pointer to an array storing count of samples per class (10 classes, array[0] = 12 means that class 0 has been observed 12 times)
    int num_classes;     // Total number of possible classes, determines the size of counts
    int total_samples;   // Total number of samples in this node, needed bc we need to compute the probability per each class.
} ClassCounts;

// Function to calculate entropy for a set of class counts on a node
double calculate_local_entropy(ClassCounts* counts) {
    double entropy = 0.0;
     
	// We use -> operator because counts is a pointer to a structure
    for (int i = 0; i < counts->num_classes; i++) {
		// Per class, if we have more than 0 samples compute entropy
        if (counts->counts[i] > 0) {
		// counts->counts is accessing the counts array inside the structure
            double p = (double)counts->counts[i] / counts->total_samples;
            entropy -= p * log2(p);
        }
    }
    // the entropy returned is of a single node, it will be combined later
    return entropy;
}

// Function to calculate entropy for a potential split
double calculate_split_entropy(
	// describe data
    float* feature_values,     // Array of feature values for the current feature (eg split on height)
    int* labels,               // Array of class labels 
    int n_samples,             // Number of samples in this node
    int n_classes,             // Total number of classes
    // parallel setup 
	float threshold,           // Value to split on
    int rank,                  // MPI rank
    int size,                  // Number of MPI processes
    MPI_Comm comm              // MPI communicator
) {
    // Calculate local range for this process
    int samples_per_proc = n_samples / size;
    int start_idx = rank * samples_per_proc;
	// If data cannot be evenly divided (ternary operator, can be written using if else)
    int end_idx = (rank == size - 1) ? n_samples : start_idx + samples_per_proc;
    
    // Initialize count structures for left and right splits
	// calloc initializes values to 0, malloc does not, we count up from 0 
    ClassCounts left = {calloc(n_classes, sizeof(int)), n_classes, 0};
    ClassCounts right = {calloc(n_classes, sizeof(int)), n_classes, 0};
    
    // Count samples in left and right splits for this process's portion
	// Based on the threshold, checks that feature and separates the sample in 2
    for (int i = start_idx; i < end_idx; i++) {
        if (feature_values[i] <= threshold) {
            left.counts[labels[i]]++;
            left.total_samples++;
        } else {
            right.counts[labels[i]]++;
            right.total_samples++;
        }
    }
    
    // Allocate buffers for receiving global counts
    int* global_left_counts = (rank == 0) ? malloc(n_classes * sizeof(int)) : NULL;
    int* global_right_counts = (rank == 0) ? malloc(n_classes * sizeof(int)) : NULL;
    int global_left_total = 0;
    int global_right_total = 0;
    
    // Reduce class counts to root process
	// Sum up all the left and right across the nodes and send them to process 0
    MPI_Reduce(left.counts, global_left_counts, n_classes, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(right.counts, global_right_counts, n_classes, MPI_INT, MPI_SUM, 0, comm);

	// qui
	// Receive the final result
    MPI_Reduce(&left.total_samples, &global_left_total, 1, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(&right.total_samples, &global_right_total, 1, MPI_INT, MPI_SUM, 0, comm);
    
    double split_entropy = 0.0;
    
    if (rank == 0) {
        // Calculate weighted average entropy
        if (global_left_total > 0) {
            ClassCounts global_left = {
                global_left_counts,
                n_classes,
                global_left_total
            };
            split_entropy += (double)global_left_total / n_samples * 
                           calculate_local_entropy(&global_left);
        }
        
        if (global_right_total > 0) {
            ClassCounts global_right = {
                global_right_counts,
                n_classes,
                global_right_total
            };
            split_entropy += (double)global_right_total / n_samples * 
                           calculate_local_entropy(&global_right);
        }
    }
    
    // Broadcast the final entropy value to all processes
	//
    MPI_Bcast(&split_entropy, 1, MPI_DOUBLE, 0, comm);
    
    // Clean up
    free(left.counts);
    free(right.counts);
    if (rank == 0) {
        free(global_left_counts);
        free(global_right_counts);
    }
    
    return split_entropy;
}
// Prompt: what is the order of functions that will be called?
// Function to find the best split for a feature
double find_best_split(
    float* feature_values,
    int* labels,
    int n_samples,
    int n_classes,
    float* unique_values,
    int n_unique,
    float* best_threshold,
    int rank,
    int size,
    MPI_Comm comm){
    double best_entropy = INFINITY;
    *best_threshold = 0.0;
    
    // Each process evaluates a subset of potential thresholds
    int thresholds_per_proc = n_unique / size;
    int start_idx = rank * thresholds_per_proc;
    int end_idx = (rank == size - 1) ? n_unique - 1 : start_idx + thresholds_per_proc;
    
    // Local best values
    double local_best_entropy = INFINITY;
    float local_best_threshold = 0.0;
    
    // Evaluate each potential threshold in this process's range
    for (int i = start_idx; i < end_idx; i++) {
		// counts->counts is accessing the counts array inside the structure
        float threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;
		// Calculate probability of class i
        double entropy = calculate_split_entropy(
            feature_values, labels, n_samples, n_classes,
            threshold, rank, size, comm
        );
        
        if (entropy < local_best_entropy) {
            local_best_entropy = entropy;
            local_best_threshold = threshold;
        }
    }
    
    // Structure to hold both entropy and threshold for reduction
    struct {
        double entropy;
        float threshold;
    } local_best = {local_best_entropy, local_best_threshold},
      global_best;
    
    // Custom MPI datatype for the reduction
    MPI_Datatype entropy_threshold_type;
    int blocklengths[2] = {1, 1};
    MPI_Aint offsets[2];
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_FLOAT};
    
    offsets[0] = offsetof(struct { double entropy; float threshold; }, entropy);
    offsets[1] = offsetof(struct { double entropy; float threshold; }, threshold);
    
    MPI_Type_create_struct(2, blocklengths, offsets, types, &entropy_threshold_type);
    MPI_Type_commit(&entropy_threshold_type);
    
    // Find global best threshold
    MPI_Allreduce(&local_best, &global_best, 1, entropy_threshold_type, 
                  MPI_MINLOC, comm);
    
    MPI_Type_free(&entropy_threshold_type);
    
    *best_threshold = global_best.threshold;
    return global_best.entropy;
}
