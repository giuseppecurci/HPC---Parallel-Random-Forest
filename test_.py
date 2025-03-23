import math 
import numpy as np 
import pandas as pd 

# Function to compute entropy for multiple classes
def compute_entropy(split):
    size = len(split)
    if size == 0:
        return 0
    class_counts = split.value_counts().values
    probabilities = class_counts / size
    return -np.sum(probabilities * np.log2(probabilities))

# Function to find the best split based on entropy for multiple classes
def find_best_split(data, target_column):
    best_entropy = float('inf')
    best_split_feature = None
    best_threshold = None

    features = data.drop(columns=[target_column])
    target = data[target_column]

    for feature in features.columns:
        values = data[feature].unique()
        values.sort()

        # Try possible split points
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            
            left_split = target[data[feature] <= threshold]
            right_split = target[data[feature] > threshold]
            
            left_entropy = compute_entropy(left_split)
            right_entropy = compute_entropy(right_split)
            
            left_size, right_size = len(left_split), len(right_split)
            total_size = left_size + right_size
            
            entropy = (left_entropy * left_size + right_entropy * right_size) / total_size

            if entropy < best_entropy:
                best_entropy = entropy
                best_split_feature = feature
                best_threshold = threshold

    return best_split_feature, best_threshold, best_entropy

path_data = "/home/peppe-pooper/01_Studio/01_Unitn/3rd_semester/HPC/project/HPC---Parallel-Random-Forest/data/classification_dataset.csv"
data = pd.read_csv(path_data)


data = pd.read_csv(path_data)
best_feature, best_threshold, min_entropy = find_best_split(data, target_column="target")
print("Best feature:", best_feature)
print("Best threshold:", best_threshold)
print("Min entropy:", min_entropy)