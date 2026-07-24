import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Load the raw dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Perform data preprocessing."""
    # Check for missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Combine back
    X_scaled['target'] = y.values
    
    return X_scaled

def main():
    # Define paths
    input_path = '../breast_cancer_raw/breast_cancer.csv'
    output_dir = 'namadataset_preprocessing'
    output_path = os.path.join(output_dir, 'breast_cancer_preprocessing.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute pipeline
    print("Loading data...")
    df = load_data(input_path)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    print(f"Saving preprocessed data to {output_path}...")
    df_processed.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
