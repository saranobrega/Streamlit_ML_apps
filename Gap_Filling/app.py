import streamlit as st
import pandas as pd
from plot import plot_results, plot_wind_speed_vs_active_power
from preprocessing import split_dataframe_by_wt, delete_ap_nan, remove_outliers_from_dataframes
from validate_data import validate_data, gap_validation
from train import perform_multistep_forecasting, detect_index_horizon


# Page layout
st.set_page_config(page_title='The Gap Filling Machine Learning App', layout='wide')

def train_model(X_train, y_train):
    model = LGBMRegressor(
        random_state=42,
        learning_rate=0.05,
        max_depth=5,
        n_estimators=300,
        num_leaves=31,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


# Main Streamlit App
def main():
    # Page title and description
    st.write("# Gap Filling Streamlit App")
    st.write("This app performs gap filling using LightGBM.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])

    # Parameters sidebar
    st.sidebar.header('Parameters')
    learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.05, 0.01)
    max_depth = st.sidebar.slider('Max Depth', 1, 20, 5, 1)
    n_estimators = st.sidebar.slider('Number of Estimators', 1, 1000, 300, 1)
    num_leaves = st.sidebar.slider('Number of Leaves', 1, 100, 31, 1)
    num_lags = st.sidebar.slider('Number of Lags', 1, 10, 3, 1)
   
    

    # Display uploaded data
    if uploaded_file is not None:
        st.write("## Uploaded Data")
        df = pd.read_csv(uploaded_file)
        df=df[:11000]
        df = df.reset_index(drop=True)
        #st.write(df)
    


        # Validate Data
        validated_data = validate_data(df)
        # Display validation result
        #st.write("## Validation Result")
        #st.write(validated_data)
       

        split_data = split_dataframe_by_wt(validated_data)
        # Plot WindSpeed versus ActivePower

        #plot_wind_speed_vs_active_power(split_data)
        #Plot
        turbines_to_plot = list(split_data.keys())[:3]  # Choose the first three turbines for plotting
        
        for wt in turbines_to_plot:
            plot_wind_speed_vs_active_power(split_data[wt], f'Turbine {wt}')

         # Interpolate NaN values in each split DataFrame
        interpolated_data = delete_ap_nan(split_data)

        # Remove outliers from the interpolated data
        data_without_outliers = remove_outliers_from_dataframes(interpolated_data)
        
        # Print how many gaps there are 

        st.write("## How many gaps in each Turbine dataframe?")
        num_gaps_list = []

        for key, df in data_without_outliers.items():
            pairs = detect_index_horizon(df)
            num_gaps = len(pairs)
            num_gaps_list.append(num_gaps)
            st.write(f"Dataframe of Turbine {key} has {num_gaps} gaps")






        st.write("## Filling in Gaps")


        # Train Model and Perform Forecasting
        synthetic_data_frames = {}
        for wt, df in data_without_outliers.items():
            st.write(f"## Filling in Gaps for Turbine {wt}")
            synthetic_data_dict = perform_multistep_forecasting({wt: df}, num_lags=3)


            print(synthetic_data_dict)
        synthetic_data_frames[wt] = synthetic_data_dict

        #Validate new data
        gap_validation(synthetic_data_frames)

        
        list_of_dfs = []

        for turbine_data in synthetic_data_frames.values():
            # Extract DataFrames from the inner dictionary
            list_of_dfs.extend(turbine_data.values())

        # Concatenate the list of DataFrames
        final_synth_df = pd.concat(list_of_dfs, ignore_index=True)
        final_synth_df =final_synth_df.sort_values(by=['DATE', 'WT'])
        final_synth_df = final_synth_df.reset_index(drop=True)
       



    else:
        st.info('Please upload a CSV file.')

# Run the Streamlit app
if __name__ == '__main__':
    main()









