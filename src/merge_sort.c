#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

void merge(double *arr, double *temp, int left, int mid, int right) {
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

static void mergeSortHelper(double *arr, double *temp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
		// avoid integer overflow
        
        // Recursively sort the subarrays
        mergeSortHelper(arr, temp, left, mid);
        mergeSortHelper(arr, temp, mid + 1, right);
        
        // Merge the sorted subarrays
        merge(arr, temp, left, mid, right);
    }
}

static void mergeSort(double *arr, int size) {
    double *temp = (double *)malloc(size * sizeof(double));
    if (temp == NULL) {
        printf("Memory allocation failed\n");
        return;
    }
    
    mergeSortHelper(arr, temp, 0, size - 1);
	// size-1 because we pass the last index of the array
    
    free(temp);
}

void printArray(double *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}
