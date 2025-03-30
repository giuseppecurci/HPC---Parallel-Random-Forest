#ifndef METRICS_H  
#define METRICS_H 

float* accuracy(int *predictions, int *targets, int size, int num_classes);
float** precision_recall(int *predictions, int *targets, int size, int num_classes);

#endif