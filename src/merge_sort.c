#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "../headers/merge_sort.h"

void merge(float *features, float *targets, float *temp_features, float *temp_targets, int left, int mid, int right) {
    int i = left;      // index left subarray
    int j = mid + 1;   // index right subarray
    int k = left;      // index temp array
    
    // merge the subarrays
    while (i <= mid && j <= right) {
        if (features[i] <= features[j]) {
            temp_features[k] = features[i];
            temp_targets[k] = targets[i];  // Maintain alignment
            i++;
        } else {
            temp_features[k] = features[j];
            temp_targets[k] = targets[j];  // Maintain alignment
            j++;
        }
        k++;
    }
    
    // copy the remaining elements from left subarray
    while (i <= mid) {
        temp_features[k] = features[i];
        temp_targets[k] = targets[i];
        i++;
        k++;
    }
    
    // copy the remaining elements from right subarray
    while (j <= right) {
        temp_features[k] = features[j];
        temp_targets[k] = targets[j];
        j++;
        k++;
    }
    
    // copy back the sorted elements to original array
    for (i = left; i <= right; i++) {
        features[i] = temp_features[i];
        targets[i] = temp_targets[i];
    }
}

void merge_sort_helper(float *features, float *targets, float *temp_features, float *temp_targets, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;  // Avoid integer overflow

        // Recursively sort the subarrays
        merge_sort_helper(features, targets, temp_features, temp_targets, left, mid);
        merge_sort_helper(features, targets, temp_features, temp_targets, mid + 1, right);

        // Merge the sorted subarrays
        merge(features, targets, temp_features, temp_targets, left, mid, right);
    }
}

void merge_sort(float *features, float *targets, int size) {
    float *temp_features = (float *)malloc(size * sizeof(float));
    float *temp_targets = (float *)malloc(size * sizeof(float));
    if (temp_features == NULL || temp_targets == NULL) {
        printf("Memory allocation failed\n");
        free(temp_features);
        free(temp_targets);
        return;
    }
    
    merge_sort_helper(features, targets, temp_features, temp_targets, 0, size - 1);
    
    free(temp_features);
    free(temp_targets);
}