//prevent multiple inclusions of the same header file//
#ifndef UTILS_H 
#define UTILS_H

#include <stdio.h>

// Function to read and print CSV data
void print_matrix(float **data, int num_rows, int num_columns);
void print_array(float *arr, int size);

#endif