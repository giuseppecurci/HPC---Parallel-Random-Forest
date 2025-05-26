/**
 * @file metrics.h
 * @brief Header file defining metrics calculation functions.
 * 
 * This file provides the function for computing common classification metrics, 
 * such as accuracy, precision, and recall. The metrics are computed for each class and 
 * can be used for model evaluation.
 * 
 * The metrics are:
 * - Accuracy: Proportion of correctly predicted instances for each class.
 * - Precision: Proportion of correctly predicted positive instances for each class.
 * - Recall: Proportion of actual positive instances correctly identified for each class.
 * 
 */
#ifndef METRICS_H  
#define METRICS_H 

/**
 * @brief Computes accuracy for each class.
 * 
 * This function calculates the accuracy for each class.
 * Accuracy is defined as the ratio of correct predictions to the total number of predictions 
 * for each class.
 * 
 * @param predictions Array of predicted class labels.
 * @param targets Array of true class labels.
 * @param size The total number of predictions/targets.
 * @param num_classes The number of unique classes.
 * @return A dynamically allocated array containing the accuracy for each class.
 */
float* accuracy(int *predictions, int *targets, int size, int num_classes);

/**
 * @brief Computes precision and recall for each class.
 * 
 * This function computes both precision and recall for each class.
 * Precision is the ratio of true positives to predicted positives, while recall is the 
 * ratio of true positives to actual positives. They are computed together for each class 
 * for efficiency.
 * 
 * @param predictions Array of predicted class labels.
 * @param targets Array of true class labels.
 * @param size The total number of predictions/targets.
 * @param num_classes The number of unique classes.
 * @return A dynamically allocated 2D array where:
 *         - metrics[0] contains the precision values for each class.
 *         - metrics[1] contains the recall values for each class.
 */
float** precision_recall(int *predictions, int *targets, int size, int num_classes);

/**
 * @brief Computes accuracy, precision, and recall and writes the results to a file.
 * 
 * This function calculates accuracy, precision, and recall for each class, and writes the results 
 * to a text file (`metrics_output.txt`).
 * 
 * @param predictions Array of predicted class labels.
 * @param targets Array of true class labels.
 * @param size The total number of predictions/targets.
 * @param num_classes The number of unique classes.
 * @param metrics_path The path to the .txt file where the metrics will be saved.
 * @param rank The rank of the process in a distributed environment (for MPI).
 */
void compute_metrics(int *predictions, int *targets, int size, int num_classes, const char* metrics_path, int rank);

/**
 * @brief Aggregates predictions from multiple processes and computes/saves performance metrics.
 * 
 * This function collects predictions from all processes in a distributed environment,
 * aggregates them, computes the performance metrics (accuracy, precision, recall),
 * and saves both the predictions and metrics to specified files.
 * 
 * @param process_number The total number of processes in the distributed environment.
 * @param test_size The number of samples in the test dataset.
 * @param num_classes The number of unique classes in the dataset.
 * @param all_predictions A 2D array containing predictions from all processes.
 * @param targets Array of true class labels.
 * @param store_predictions_path The path to the file where predictions will be saved.
 * @param store_metrics_path The path to the file where metrics will be saved.
 * @param rank The rank of the current process in the distributed environment.
 */
void aggregate_and_save_predictions(int process_number, int test_size, int num_classes,
                                     int **all_predictions, int *tree_count, int *targets,
                                     const char *store_predictions_path, const char *store_metrics_path, int rank);

#endif
