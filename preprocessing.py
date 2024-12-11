import pandas as pd
import numpy as np

def preprocessing(df):
    """ 
    Analyze a DataFrame to calculate the available-to-missing value ratio for each column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to analyze.
    
    Returns:
    results (pd.DataFrame): DataFrame with Available_Count, Missing_Count, and Available_to_Missing_Ratio for each column.
    df1 (pd.DataFrame): Subset of columns with Available_to_Missing_Ratio >= 8 or with no missing values.
    """
    
    # Define a function to check if a value is not a number (NaN or any non-numeric type)
    def is_not_number(x):
        return not (isinstance(x, (int, float)) or pd.isna(x))
    
    # Apply the function across the DataFrame to identify non-numeric or missing values
    not_number = df.applymap(is_not_number)
    
    # Count 'True' values returned by the function (which indicate non-numeric values)
    not_number_count = not_number.sum()
    
    # Count the numeric and non-missing values in each column
    numeric_and_available_count = df.applymap(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).sum()
    
    # Calculate the ratio of available values to missing values for each column
    available_to_missing_ratio = numeric_and_available_count / not_number_count.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
    
    # Combine the counts and ratios into a single DataFrame
    results = pd.DataFrame({
        'Column': df.columns,
        'Available_Count': numeric_and_available_count,
        'Missing_Count': not_number_count,
        'Available_to_Missing_Ratio': available_to_missing_ratio
    })
    
    # Set the column names as the index
    results.set_index('Column', inplace=True)
    
    # Subset columns where the ratio of available to missing values is 8 or greater
    df1 = results[(results['Available_to_Missing_Ratio'] >= 8) | (results['Missing_Count'] == 0)]
    
    print("Overall Results:\n", results)
    print("\nColumns with 8 or more available values per missing value:\n", df1)
    
    return results, df1

# Example usage:
# df = pd.read_excel('path_to_file.xlsx', sheet_name=0)  # Replace with actual file path and sheet name
# results, filtered_df = preprocessing(df)  # Call the function with your DataFrame

