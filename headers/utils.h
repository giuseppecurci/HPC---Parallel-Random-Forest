//prevent multiple inclusions of the same header file//
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>

void print_matrix(float **data, int num_rows, int num_columns, int max_rows);
void print_array(float *arr, int size, int max_elements);

#endif