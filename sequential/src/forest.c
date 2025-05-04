#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "../headers/tree/train_utils.h"
#include "../headers/tree/tree.h"
#include "../headers/tree/utils.h"
#include "../headers/forest.h"
#include "../headers/utils.h"


void create_forest(Forest *forest, int num_trees, int max_depth, int min_samples_split, char* max_features) {
    forest->num_trees = num_trees;
    forest->max_depth = max_depth;
    forest->min_samples_split = min_samples_split;
    forest->max_features = max_features;
    forest->trees = (Tree *)malloc(num_trees * sizeof(Tree));
    
    for (int i = 0; i < num_trees; i++) {
        forest->trees[i].root = NULL;
    }
}

void train_forest(Forest *forest, float **data, int num_rows, int num_columns, int train_tree_size, int num_classes) {
    for (int i = 0; i < forest->num_trees; i++) {
        printf("\rTraining tree %d/%d... (%d%%)", i + 1, forest->num_trees, (i + 1) * 100 / forest->num_trees);
        fflush(stdout);
        if (train_tree_size != num_rows) {
            float ** sampled_data = (float **)malloc(train_tree_size * sizeof(float *));
            for (int j = 0; j < train_tree_size; j++) {
                sampled_data[j] = (float *)malloc(num_columns * sizeof(float));
            }
            sample_data_without_replacement(data, num_rows, num_columns, train_tree_size, sampled_data);
            train_tree(&forest->trees[i], sampled_data, train_tree_size, num_columns, num_classes, 
                   forest->max_depth, forest->min_samples_split, forest->max_features);  
            for (int j = 0; j < train_tree_size; j++) {
                free(sampled_data[j]);
            }
            free(sampled_data);  
        }
        else {
            train_tree(&forest->trees[i], data, num_rows, num_columns, num_classes, 
                   forest->max_depth, forest->min_samples_split, forest->max_features);
        }
    }
    printf("\n");
}

int* forest_inference(Forest *forest, float **data, int num_rows, int num_classes) {
    int *predictions = (int *)malloc(num_rows * sizeof(int));
    
    int **predictions_per_tree = (int **)malloc(forest->num_trees * sizeof(int *));
    for (int i = 0; i < forest->num_trees; i++) {
        printf("\rInference tree %d/%d... (%d%%)", i + 1, forest->num_trees, (i + 1) * 100 / forest->num_trees);
        fflush(stdout);
        predictions_per_tree[i] = tree_inference(&forest->trees[i], data, num_rows);
    }
    printf("\n");

    for (int i = 0; i < num_rows; i++) {
        int *class_counts = (int *)calloc(num_classes, sizeof(int));
        for (int j = 0; j < forest->num_trees; j++) {
            class_counts[predictions_per_tree[j][i]]++;
        }
        predictions[i] = argmax(class_counts, num_classes);
        free(class_counts);
    }

    for (int i = 0; i < forest->num_trees; i++) {
        free(predictions_per_tree[i]);
    }
    free(predictions_per_tree);

    return predictions;
}

void free_forest(Forest *forest) {
    for (int i = 0; i < forest->num_trees; i++) {
        destroy_tree(&forest->trees[i]);
    }
    free(forest->trees);
    free(forest);
}

void serialize_forest(Forest *forest, const char *out_dir) {
    // Build the path for the config file (now properly text)
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/forest_config.txt", out_dir);

    FILE *config_file = fopen(config_path, "w");  // open in text mode
    if (config_file == NULL) {
        perror("Failed to open config file for serialization of random forest");
        return;
    }

    // Write forest configuration as text
    fprintf(config_file, "num_trees: %d\n", forest->num_trees);
    fprintf(config_file, "max_depth: %d\n", forest->max_depth);
    fprintf(config_file, "min_samples_split: %d\n", forest->min_samples_split);
    fprintf(config_file, "max_features: %s\n", forest->max_features);

    fclose(config_file);

    // Serialize each tree into a separate binary file
    for (int i = 0; i < forest->num_trees; i++) {
        char tree_path[512];
        snprintf(tree_path, sizeof(tree_path), "%s/random_tree_%d.bin", out_dir, i);
        serialize_tree(&forest->trees[i], tree_path);
    }
}

void deserialize_forest(Forest *forest, const char *dir_path) {
    // Construct path to config file
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/forest_config.txt", dir_path);

    FILE *config_file = fopen(config_path, "rb");
    if (config_file == NULL) {
        perror("Failed to open forest config file");
        return;
    }
    
    // Read forest configuration
    fscanf(config_file, "num_trees: %d\n", &forest->num_trees);
    fscanf(config_file, "max_depth: %d\n", &forest->max_depth);
    fscanf(config_file, "min_samples_split: %d\n", &forest->min_samples_split);
    // Read null-terminated string for max_features
    char buffer[4];
    fscanf(config_file, "max_features: %s\n", buffer);
    forest->max_features = strdup(buffer);

    fclose(config_file);

    // Deserialize each tree from its file
    for (int i = 0; i < forest->num_trees; ++i) {
        char tree_path[512];
        snprintf(tree_path, sizeof(tree_path), "%s/random_tree_%d.bin", dir_path, i);
        forest->trees[i] = *deserialize_tree(tree_path);
    }
}
