//prevent multiple inclusions of the same header file//
#ifndef UTILS_H 
#define UTILS_H

#include <stdio.h>
#include <math.h>

// Function to read and print CSV data
void print_matrix(float **data, int num_rows, int num_columns, int max_rows);
void print_array(float *arr, int size, int max_elements);
float get_entropy(float *left_split, float *right_split, int left_size, int right_size);

#endif