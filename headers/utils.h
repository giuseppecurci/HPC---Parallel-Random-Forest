/**
 * @file utils.h
 * @brief Utility functions for matrix printing, array printing, and CSV file reading.
 * 
 * This file provides various utility functions such as:
 * - Printing a matrix with specified formatting.
 * - Printing an array with a specified maximum number of elements.
 * - Reading a CSV file and storing the data into a matrix.
 * 
 * These functions can be used for debugging and data processing purposes.
 * 
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>

//Max number of characters that can be stored in the buffer line
#define MAX_LINE 1024
#define MAX_ROWS 100000

/**
 * @brief Prints a matrix with specified number of rows and columns.
 * 
 * This function prints a feature matrix with the option to limit the number of rows printed.
 * The target column is included in the output with a separator between features and the target.
 * 
 * @param data The matrix to be printed.
 * @param num_rows The number of rows in the matrix.
 * @param num_columns The number of columns in the matrix.
 * @param max_rows The maximum number of rows to print. If set to -1, all rows are printed.
 */
void print_matrix(float **data, int num_rows, int num_columns, int max_rows);

/**
 * @brief Prints the elements of a single-dimensional array.
 * 
 * This function prints the elements of an array up to a specified maximum number of elements.
 * 
 * @param arr The array to be printed.
 * @param size The total size of the array.
 * @param max_elements The maximum number of elements to print. If set to -1, all elements are printed.
 */
void print_array(float *arr, int size, int max_elements);

/**
 * @brief Reads data from a CSV file and returns it as a matrix.
 * 
 * This function reads data from a CSV file, allocates memory for a matrix, and stores the data.
 * The first row of the CSV is treated as a header, and the data rows are read into the matrix.
 * 
 * @param filename The name of the CSV file to be read.
 * @param num_rows Output parameter that will store the number of rows in the matrix.
 * @param num_columns Output parameter that will store the number of columns in the matrix.
 * @return A pointer to the matrix containing the data from the CSV file, or NULL if an error occurs.
 */
float** read_csv(const char *filename, int *num_rows, int *num_columns);

#endif