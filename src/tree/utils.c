#include <stdlib.h>
#include <stdio.h>
#include "../headers/tree/utils.h"
#include "../headers/tree/tree.h"
#include "../headers/tree/train_utils.h"

void get_class_pred(float** data, int num_rows, int num_columns, int num_classes, Node *node) {
    int classes_count[num_classes];
    for (int i = 0; i < num_rows; i++) {
        classes_count[(int)data[i][num_columns - 1]]++;
    }
    node->pred = argmax(classes_count, 3);
}

void destroy_tree(Tree *tree) {
    destroy_node(tree->root);
    free(tree);
};

void destroy_node(Node *node) {
    if (node == NULL) return;
    destroy_node(node->left);
    destroy_node(node->right);
    free(node);
};

void print_tree(Tree *tree) {
    printf("Printing tree\n");
    print_node(tree->root);
};

void print_node(Node *node) {
    if (node == NULL) return;
    printf("Feature: %d, Threshold: %.6f, Value: %d, Num samples: %d\n", node->feature, node->threshold, node->pred, node->num_samples);
    print_node(node->left);
    print_node(node->right);
};

void save_predictions(const int *predictions, int num_rows, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Errore nell'apertura del file per scrivere le predizioni");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "Predictions\n");
    for (int i = 0; i < num_rows; i++) {
        fprintf(fp, "%d\n", predictions[i]);
    }

    fclose(fp);
}

void serialize_node(Node *node, FILE *fp) {
    if (node == NULL) {
        int marker = -1;
        fwrite(&marker, sizeof(int), 1, fp);
        return;
    }

    int marker = 1;
    fwrite(&marker, sizeof(int), 1, fp);
    fwrite(&node->feature, sizeof(int), 1, fp);
    fwrite(&node->threshold, sizeof(float), 1, fp);
    fwrite(&node->pred, sizeof(int), 1, fp);
    fwrite(&node->entropy, sizeof(float), 1, fp);
    fwrite(&node->depth, sizeof(int), 1, fp);
    fwrite(&node->num_samples, sizeof(int), 1, fp);

    serialize_node(node->left, fp);
    serialize_node(node->right, fp);
}

void serialize_tree(Tree *tree, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Errore nell'apertura del file per la scrittura");
        exit(EXIT_FAILURE);
    }

    serialize_node(tree->root, fp);
    fclose(fp);
}

Node *deserialize_node(FILE *fp) {
    int marker;
    if (fread(&marker, sizeof(int), 1, fp) != 1)
        return NULL;

    if (marker == -1)
        return NULL;

    Node *node = malloc(sizeof(Node));
    if (!node) {
        perror("Errore nell'allocazione di memoria per il nodo");
        exit(EXIT_FAILURE);
    }

    fread(&node->feature, sizeof(int), 1, fp);
    fread(&node->threshold, sizeof(float), 1, fp);
    fread(&node->pred, sizeof(int), 1, fp);
    fread(&node->entropy, sizeof(float), 1, fp);
    fread(&node->depth, sizeof(int), 1, fp);
    fread(&node->num_samples, sizeof(int), 1, fp);

    node->left = deserialize_node(fp);
    node->right = deserialize_node(fp);

    return node;
}

Tree *deserialize_tree(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Errore nell'apertura del file per la lettura");
        exit(EXIT_FAILURE);
    }

    Tree *tree = malloc(sizeof(Tree));
    if (!tree) {
        perror("Errore nell'allocazione di memoria per l'albero");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    tree->root = deserialize_node(fp);
    fclose(fp);
    return tree;
}