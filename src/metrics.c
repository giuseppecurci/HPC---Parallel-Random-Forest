#include <stdlib.h>
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