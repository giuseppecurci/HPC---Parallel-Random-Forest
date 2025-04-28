//Max number of characters that can be stored in the buffer line
#define MAX_LINE 1024
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
                    char **dataset_path, float *train_proportion, int *seed);
float* read_csv(const char *filename, int *num_rows, int *num_columns);

void stratified_split(float *data, int num_rows, int num_columns, int num_classes, float train_proportion,
                     float **train_data, int *train_size, float **test_data, int *test_size, int seed);


void summary(char* dataset_path, float train_proportion, int train_size, int num_columns,
             int num_classes, int num_trees, int max_depth, int min_samples_split, char* max_features, 
             char* store_predictions_path, char* store_metrics_path, char* new_tree_path, 
             char* trained_tree_path, int seed);
