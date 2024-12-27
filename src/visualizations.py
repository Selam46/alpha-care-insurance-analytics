import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(data, numerical_cols):
    """
    Plot histograms for numerical features.

    Parameters:
        data (pd.DataFrame): DataFrame with data.
        numerical_cols (list): List of numerical column names.
    """
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

def plot_scatter(data, x, y, hue=None):
    """
    Plot scatter plot for two numerical columns.

    Parameters:
        data (pd.DataFrame): DataFrame with data.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis.
        hue (str, optional): Column name for hue.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=hue, data=data)
    plt.title(f"{x} vs {y}")
    plt.show()
