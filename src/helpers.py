import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

def load_data(csv_file_path):
    """
    Load data from the CSV file.

    Parameters:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(csv_file_path)

def create_groups(data, feature, group_a, group_b):
    """
    Create control (Group A) and test (Group B) groups based on a feature.

    Parameters:
        data (pd.DataFrame): The dataset.
        feature (str): Feature for grouping.
        group_a (any): Value for Group A.
        group_b (any): Value for Group B.

    Returns:
        pd.DataFrame, pd.DataFrame: Group A and Group B DataFrames.
    """
    group_a_data = data[data[feature] == group_a]
    group_b_data = data[data[feature] == group_b]
    return group_a_data, group_b_data

def perform_chi2_test(data, feature, target):
    """
    Perform chi-squared test for categorical data.

    Parameters:
        data (pd.DataFrame): The dataset.
        feature (str): Feature to test.
        target (str): Target feature for chi-squared.

    Returns:
        float: p-value from the test.
    """
    contingency_table = pd.crosstab(data[feature], data[target])
    _, p, _, _ = chi2_contingency(contingency_table)
    return p

def perform_t_test(group_a, group_b, target):
    """
    Perform t-test for numerical data.

    Parameters:
        group_a (pd.DataFrame): Group A data.
        group_b (pd.DataFrame): Group B data.
        target (str): Target feature for t-test.

    Returns:
        float: p-value from the test.
    """
    t_stat, p = ttest_ind(group_a[target], group_b[target], nan_policy='omit')
    return p
