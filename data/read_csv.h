//prevent multiple inclusions of the same header file//
#ifndef READ_CSV_H 
#define READ_CSV_H

//Max number of characters that can be stored in the buffer line
#define MAX_LINE 1024
#define MAX_ROWS 100000

// Function to read and print CSV data
float** read_csv(const char *filename, int *num_rows, int *num_columns);

#endif
