import pandas as pd

def load_data(txt_file_path, csv_file_path):
    """
    Load data from a .txt file and save it as .csv.

    Parameters:
        txt_file_path (str): Path to the input .txt file.
        csv_file_path (str): Path to save the converted .csv file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    data = pd.read_csv(txt_file_path, delimiter='|')
    data.to_csv(csv_file_path, index=False)
    return data

def handle_missing_values(data):
    """
    Handle missing values by filling numerical columns with median
    and categorical columns with mode.

    Parameters:
        data (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    return data

def clean_non_numeric_columns(data):
    """
    Convert numeric-like columns to numeric and drop invalid ones.
    
    Parameters:
        data (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for col in data.columns:
        if data[col].dtype == 'object':  # Check if column is object type
            try:
                # Attempt conversion to numeric
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except ValueError:
                print(f"Skipping column '{col}' as it contains non-numeric data.")
    return data
