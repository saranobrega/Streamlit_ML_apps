import numpy as np
import pandas as pd

def split_dataframe_by_wt(input_df):
    # Group the input DataFrame by the "WT" column
    grouped = input_df.groupby('WT')

    # Create a dictionary of DataFrames, one for each unique "WT" value
    dataframes_dict = {name: group for name, group in grouped}

    return dataframes_dict


def delete_ap_nan(dataframes_dict):
    imputed_dataframes_dict = {}

    for key, df in dataframes_dict.items():
        df = df.reset_index(drop=True)
        wt1_df_imputation2 = df.copy()
        
        # Code to delete "ActivePower" column
        if "ActivePower" in wt1_df_imputation2.columns:
            wt1_df_imputation2.drop(columns=["ActivePower"], inplace=True)
            print(f"Deleted 'ActivePower' column for key={key}")
        else:
            print(f"'ActivePower' column not found for key={key}")

        # Code to delete rows with NaN in "WindSpeed" column
        rows_before_drop = len(wt1_df_imputation2)
        wt1_df_imputation2 = wt1_df_imputation2.dropna(subset=["WindSpeed"])
        rows_after_drop = len(wt1_df_imputation2)
        
        rows_deleted = rows_before_drop - rows_after_drop
        print(f"Deleted {rows_deleted} rows with NaN in 'WindSpeed' for key={key}")

        imputed_dataframes_dict[key] = wt1_df_imputation2

    return imputed_dataframes_dict



def remove_outliers_from_dataframes(interpolated_dataframes, column_of_interest="WindSpeed", z_threshold=3):
    cleaned_dataframes = {}  # Dictionary to store cleaned DataFrames

    # Check if input is a single DataFrame
    if not isinstance(interpolated_dataframes, dict):
        # If it's a single DataFrame, create a dictionary with a default key
        interpolated_dataframes = {'default_key': interpolated_dataframes}

    for wt, df in interpolated_dataframes.items():
        # Extract the DataFrame with interpolated data
        if isinstance(df, pd.DataFrame):
            df = pd.DataFrame.from_dict(df)

            # Calculate Z-scores for the specified column
            z_scores = np.abs((df[column_of_interest] - df[column_of_interest].mean()) / df[column_of_interest].std())

            # Keep only the rows where the Z-score is below the specified threshold
            df_no_outliers = df[z_scores < z_threshold]

            # Print the number of rows deleted due to outliers
            rows_deleted = len(df) - len(df_no_outliers)
            percentage_deleted = (rows_deleted / len(df)) * 100
            print(f"For WT {wt}, {percentage_deleted:.2f}% rows were deleted due to outliers.")

            cleaned_dataframes[wt] = df_no_outliers  # Store the DataFrame without outliers

    return cleaned_dataframes
