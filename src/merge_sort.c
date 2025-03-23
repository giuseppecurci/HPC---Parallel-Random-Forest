#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "../headers/merge_sort.h"

void merge(float *arr, double *temp, int left, int mid, int right) {
    int i = left;      // index left subarray
    int j = mid + 1;   // index right subarray
    int k = left;      // index temp array
    
    // merge the subarrays
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    // copy the remaining elements from left subarray
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    
    // copy the remaining elements from right subarray
    while (j <= right) {
        temp[k++] = arr[j++];
    }
    
    // copy back the sorted elements to original array
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

void merge_sort_helper(float *arr, double *temp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
		// avoid integer overflow
        
        // Recursively sort the subarrays
        merge_sort_helper(arr, temp, left, mid);
        merge_sort_helper(arr, temp, mid + 1, right);
        
        // Merge the sorted subarrays
        merge(arr, temp, left, mid, right);
    }
}

void merge_sort(float *arr, int size) {
    double *temp = (double *)malloc(size * sizeof(double));
    if (temp == NULL) {
        printf("Memory allocation failed\n");
        return;
    }
    
    merge_sort_helper(arr, temp, 0, size - 1);
	// size-1 because we pass the last index of the array
    
    free(temp);
}