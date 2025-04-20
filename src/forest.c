#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "forest.h"
#include "utils.h"
#include "tree/train_utils.h"
#include "tree/tree.h"
#include "tree/utils.h"

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

void train_forest(Forest *forest, float **data, int num_rows, int num_columns, int num_classes) {
    for (int i = 0; i < forest->num_trees; i++) {
        printf("\rTraining tree %d/%d... (%d%%)", i + 1, forest->num_trees, (i + 1) * 100 / forest->num_trees);
        fflush(stdout);
        train_tree(&forest->trees[i], data, num_rows, num_columns, num_classes);
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

Forest* deserialize_forest(const char *dir_path) {
    // Construct path to config file
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/forest_config.txt", dir_path);

    FILE *config_file = fopen(config_path, "rb");
    if (config_file == NULL) {
        perror("Failed to open forest config file");
        return NULL;
    }

    Forest *forest = (Forest *)malloc(sizeof(Forest));
    if (!forest) {
        perror("Failed to allocate memory for forest");
        fclose(config_file);
        return NULL;
    }

    // Read forest configuration
    fread(&forest->num_trees, sizeof(int), 1, config_file);
    fread(&forest->max_depth, sizeof(int), 1, config_file);
    fread(&forest->min_samples_split, sizeof(int), 1, config_file);

    // Read null-terminated string for max_features
    char buffer[256];
    fread(buffer, sizeof(char), sizeof(buffer), config_file);
    forest->max_features = strdup(buffer);  // malloc + copy

    fclose(config_file);

    // Allocate array for trees
    forest->trees = (Tree *)malloc(forest->num_trees * sizeof(Tree));
    if (!forest->trees) {
        perror("Failed to allocate memory for trees");
        free(forest->max_features);
        free(forest);
        return NULL;
    }

    // Deserialize each tree from its file
    for (int i = 0; i < forest->num_trees; ++i) {
        printf("%d", forest->num_trees);
        char tree_path[512];
        snprintf(tree_path, sizeof(tree_path), "%s/random_tree_%d.bin", dir_path, i);
        forest->trees[i] = *deserialize_tree(tree_path);
    }

    return forest;
}