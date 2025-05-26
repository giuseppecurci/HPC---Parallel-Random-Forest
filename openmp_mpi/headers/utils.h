//Max number of characters that can be stored in the buffer line
#define MAX_LINE 1024

//Maximum number of rows that can be processed
#define MAX_ROWS 20000000

/**
 * @brief Parses command-line arguments for various options.
 * 
 * This function parses command-line arguments to set various parameters. It returns 0 on success
 * and 1 on error. The function checks for the following arguments:
 * @param max_matrix_rows_print Maximum number of rows to print from the matrix (--print_matrix).
 * @param num_classes Number of classes in the dataset (--num_classes int).
 * @param num_trees Number of trees in the forest. (--num_trees int).
 * @param max_depth Maximum depth of the trees. (--max_depth int).
 * @param min_samples_split Minimum number of samples required to split a node. (--min_samples_split int).
 * @param max_features Number of features to consider when looking for the best split. (--max_features char*).
 * @param trained_tree_path Path for the trained tree to deserialize (--trained_tree_path).
 * @param store_predictions_path Path for storing predictions (--store_predictions_path).
 * @param store_metrics_path Path for storing performance metrics (--store_metrics_path).
 * @param new_tree_path Path for the new tree to train and then serialize (--new_tree_path).
 * @param dataset_path Path for the dataset to be used (--dataset_path).
 * @param train_proportion Proportion of data to be used for training (--train_proportion).
 * @param num_trees Number of trees to be used in the forest (--num_trees).
 * @param seed Random seed for reproducibility (--seed).
 */
int parse_arguments(int argc, char *argv[], int *max_matrix_rows_print, int *num_classes, int *num_trees,
                    int *max_depth, int *min_samples_split, char **max_features, char **trained_tree_path, 
                    char **store_predictions_path, char **store_metrics_path, char **new_tree_path, 
                    char **dataset_path, float *train_proportion, float *train_tree_proportion, int *seed, int *thread_count);

/**
 * @brief Reads data from a CSV file into a float array.
 * 
 * This function reads numerical data from a CSV file and stores it in a dynamically allocated
 * float array. It also determines the number of rows and columns in the dataset.
 * 
 * @param filename Path to the CSV file to read.
 * @param num_rows Pointer to store the number of rows read from the file.
 * @param num_columns Pointer to store the number of columns detected in the file.
 * @return A pointer to the allocated float array containing the data, or NULL on failure.
 */
float* read_csv(const char *filename, int *num_rows, int *num_columns);

/**
 * @brief Performs a stratified split of the dataset into training and testing sets.
 * 
 * This function divides the dataset into training and testing subsets while maintaining
 * the same class distribution in both sets.
 * 
 * @param data The input dataset as a float array.
 * @param num_rows Total number of rows in the dataset.
 * @param num_columns Number of columns in the dataset (including the class label).
 * @param num_classes Number of different classes in the dataset.
 * @param train_proportion Proportion of data to be used for training (between 0 and 1).
 * @param train_data Pointer to store the allocated training dataset.
 * @param train_size Pointer to store the number of samples in the training set.
 * @param test_data Pointer to store the allocated testing dataset.
 * @param test_size Pointer to store the number of samples in the testing set.
 * @param seed Random seed for reproducible splitting.
 */
void stratified_split(float *data, int num_rows, int num_columns, int num_classes, float train_proportion,
                     float **train_data, int *train_size, float **test_data, int *test_size, int seed);

/**
 * @brief Displays a summary of the random forest configuration and dataset information.
 * 
 * This function prints a summary of all the parameters used for training and evaluation
 * of the random forest model.
 * 
 * @param dataset_path Path to the dataset file.
 * @param train_proportion Proportion of data used for training.
 * @param train_size Number of samples in the training set.
 * @param num_columns Number of columns in the dataset (including the class label).
 * @param num_classes Number of different classes in the dataset.
 * @param num_trees Number of trees in the random forest.
 * @param max_depth Maximum depth of trees in the forest.
 * @param min_samples_split Minimum number of samples required to split a node.
 * @param max_features Strategy for selecting features at each split.
 * @param store_predictions_path Path for storing model predictions.
 * @param store_metrics_path Path for storing model performance metrics.
 * @param new_tree_path Path for saving the newly trained forest model.
 * @param trained_tree_path Path to a pre-trained forest model (if used).
 * @param seed Random seed used for reproducibility.
 */
void summary(char* dataset_path, float train_proportion, float train_tree_proportion, int train_size, int num_columns,
             int num_classes, int num_trees, int max_depth, int min_samples_split, char* max_features, 
             char* store_predictions_path, char* store_metrics_path, char* new_tree_path,
             char* trained_tree_path, int seed);

/**
 * Samples data without replacement from the training dataset
 *
 * @param train_data The full training dataset
 * @param train_size Number of samples in the training dataset
 * @param num_columns Number of features per sample
 * @param sample_proportion Proportion of data to sample (e.g., 0.75 for 75%)
 * @param sampled_data Output buffer for the sampled data (must be pre-allocated)
 * @return Returns 0 on success, non-zero on failure
 */
int sample_data_without_replacement(float *train_data, int train_size, int num_columns,
                                    float sample_proportion, float *sampled_data, int seed);

/**
 * @brief Distributes trees among processes for parallel random forest training.
 * 
 * This function calculates how many trees each process should train in a distributed
 * computing environment. It populates the counts array with the number of trees each
 * process should train, and the displs array with the starting index for each process.
 * 
 * @param num_trees Total number of trees to be trained in the forest.
 * @param size Number of available processes for parallel computation.
 * @param counts Array to store the number of trees each process should train.
 * @param displs Array to store the displacement (starting index) for each process.
 */
void distribute_trees(int num_trees, int size, int *counts, int *displs);
