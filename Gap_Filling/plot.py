import matplotlib.pyplot as plt
import streamlit as st

def plot_results(result_df, merged_df):
    st.write("### Plotting Results")
    # Scatter plot for merged_df
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['DATE'], merged_df['WindSpeed'], color='blue', label='WindSpeed')
    
    # Highlight specific data points in red
    specific_data_points = merged_df[merged_df['DATE'].isin(result_df['DATE'])]
    plt.scatter(specific_data_points['DATE'], specific_data_points['WindSpeed'], color='red', label='Synthetic Data Points')
    
    plt.title('Scatter Plot of WindSpeed over Time')
    plt.xlabel('Date')
    plt.ylabel('WindSpeed')
    plt.legend()
    st.pyplot(plt)


def plot_wind_speed_vs_active_power(df, turbine_label):
    # Plot WindSpeed versus ActivePower for a single turbine
    st.write(f"## ActivePower vs WindSpeed Plot ({turbine_label})")
    #for key, df in df_original.items():
    plt.figure(figsize=(10, 6))
    #print("df", df)
    plt.scatter(df['WindSpeed'], df['ActivePower'], alpha=0.5)
    #plt.title(f'Scatter Plot of WindSpeed versus ActivePower (Turbine {turbine_label})')
    plt.xlabel('WindSpeed')
    plt.ylabel('ActivePower')
    st.pyplot(plt)


