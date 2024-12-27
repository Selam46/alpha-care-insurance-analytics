import pandas as pd

def data_summary(data):
    """
    Summarize the dataset, including data types and missing values.

    Parameters:
        data (pd.DataFrame): DataFrame to summarize.

    Returns:
        dict: Summary of data information.
    """
    info = data.info()
    missing_values = data.isnull().sum()
    return {"info": info, "missing_values": missing_values}


def correlation_matrix(data):
    """
    Calculate correlation matrix for numerical features.

    Parameters:
        data (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    # Convert numeric-like columns and drop non-numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    return numeric_data.corr()


