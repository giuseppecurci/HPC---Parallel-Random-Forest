#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "../headers/metrics.h"

float* accuracy(int *predictions, int *targets, int size, int num_classes)
{
    float *accuracy = (float *)calloc(num_classes, sizeof(float));
    int *correct = (int *)calloc(num_classes, sizeof(int));
    int *total = (int *)calloc(num_classes, sizeof(int));
    for (int i = 0; i < size; i++)
    {
        total[targets[i]]++;
        if (predictions[i] == targets[i])
        {
            correct[predictions[i]]++;
        }
    }
    for (int i = 0; i < num_classes; i++)
    {
        accuracy[i] = (float)correct[i] / total[i];
    }
    free(correct);
    free(total);
    return accuracy;
}


float** precision_recall(int *predictions, int *targets, int size, int num_classes)
{
    float **metrics = (float **)malloc(2 * sizeof(float *));  // Allocate space for two float arrays
    metrics[0] = (float *)malloc(num_classes * sizeof(float)); // Precision array
    metrics[1] = (float *)malloc(num_classes * sizeof(float)); // Recall array

    int *true_positives = (int *)calloc(num_classes, sizeof(int));
    int *false_positives = (int *)calloc(num_classes, sizeof(int));
    int *positives = (int *)calloc(num_classes, sizeof(int));

    for (int i = 0; i < size; i++)
    {
        positives[targets[i]]++;
        if (predictions[i] == targets[i])
        {
            true_positives[predictions[i]]++;
        }
        else
        {
            false_positives[predictions[i]]++;
        }
    }

    for (int i = 0; i < num_classes; i++)
    {
        metrics[0][i] = (float)true_positives[i] / (true_positives[i] + false_positives[i]); // Precision
        metrics[1][i] = (float)true_positives[i] / (true_positives[i] + (positives[i] - true_positives[i])); // Recall
    }

    free(true_positives);
    free(false_positives);
    free(positives);

    return metrics;
}

void compute_metrics(int *predictions, int *targets, int size, int num_classes, const char* metrics_path, int rank)
{
    float *acc = accuracy(predictions, targets, size, num_classes);
    float **pr = precision_recall(predictions, targets, size, num_classes);

    FILE *metrics_doc = fopen(metrics_path, "w");
    if (metrics_doc == NULL) {
        printf("Error opening file for writing\n");
        return;
    }

    for (int i = 0; i < num_classes; i++)
    {
        fprintf(metrics_doc, "Accuracy for class %d: %.6f\n", i, acc[i]);
        fprintf(metrics_doc, "Precision for class %d: %.6f\n", i, pr[0][i]);
        fprintf(metrics_doc, "Recall for class %d: %.6f\n", i, pr[1][i]);
        fprintf(metrics_doc, "*********************\n");
    }
	// Get and print timestamp
    time_t now = time(NULL);
    char *timestamp = ctime(&now);
    if (timestamp != NULL) {
        // Remove newline from timestamp
        timestamp[strcspn(timestamp, "\n")] = '\0';
        fprintf(metrics_doc, "Timestamp: %s\n Process that wrote the file: %d", timestamp, rank);
    }

    free(acc);
    for (int i = 0; i < 2; i++) free(pr[i]);
    free(pr);
    fclose(metrics_doc);
}

void aggregate_and_save_predictions(int process_number, int test_size, int num_classes,
                                   int **all_predictions, int *tree_counts, int *targets,
                                   const char *store_predictions_path, const char *store_metrics_path, int rank) {
    // Calculate total number of trees
    int total_trees = 0;
    for (int p = 0; p < process_number - 1; p++) {
        total_trees += tree_counts[p];
    }

    int *aggregated_predictions = (int *)malloc(test_size * sizeof(int));
    if (!aggregated_predictions) {
        printf("Memory allocation failed for aggregated_predictions\n");
        return;
    }

    for (int i = 0; i < test_size; i++) {
        // Vote count array for each class
        int *votes = (int *)calloc(num_classes, sizeof(int));
        if (!votes) {
            printf("Memory allocation failed for votes\n");
            free(aggregated_predictions);
            return;
        }

        // Aggregate votes from all trees (all processes)
        for (int p = 0; p < process_number - 1; p++) {
            int trees_p = tree_counts[p];
            for (int t = 0; t < trees_p; t++) {
                int pred_class = all_predictions[p][t * test_size + i];
                if (p != process_number - 2 || t != trees_p - 1) {
                }
                if (pred_class >= 0 && pred_class < num_classes) {
                    votes[pred_class]++;
                }
            }
        }

        // Find class with maximum votes
        int max_votes = -1;
        int max_class = 0;
        for (int c = 0; c < num_classes; c++) {
            if (votes[c] > max_votes) {
                max_votes = votes[c];
                max_class = c;
            }
        }
        aggregated_predictions[i] = max_class;
        free(votes);
    }

    // Save predictions if path is given
    if (store_predictions_path != NULL) {
        FILE *pred_file = fopen(store_predictions_path, "w");
        if (pred_file == NULL) {
            printf("Error opening predictions file for writing\n");
        } else {
            fprintf(pred_file, "true_label,predicted_label\n");
            for (int i = 0; i < test_size; i++) {
                fprintf(pred_file, "%d,%d\n", targets[i], aggregated_predictions[i]);
            }
            fclose(pred_file);
        }
    }

    // Compute and save metrics if path is given
    if (store_metrics_path != NULL) {
        compute_metrics(aggregated_predictions, targets, test_size, num_classes, store_metrics_path, rank);
    }

    free(aggregated_predictions);
}

