import argparse
import pandas as pd
from sklearn.datasets import make_classification
import os

def generate_dataset(n_samples, n_features):
    """
    Generate a classification dataset with specified number of samples and numerical 
    features following a Gaussian distribution.
    
    Parameters:
        n_samples (int): Number of samples in the dataset.
        n_features (int): Number of features in the dataset.
    
    Saves:
        CSV file containing the dataset in the current working directory.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    filename = os.path.join(os.getcwd(), 'classification_dataset.csv')
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    """
    Parse command-line arguments and generate a classification dataset.
    """
    parser = argparse.ArgumentParser(description="Generate a classification dataset and save it as CSV.")
    parser.add_argument("--samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--features", type=int, required=True, help="Number of features")
    
    args = parser.parse_args()
    generate_dataset(args.samples, args.features)