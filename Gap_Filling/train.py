import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from plot import plot_results

def detect_index_horizon(df):
    df = df.reset_index(drop=True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    time_diff = df['DATE'].diff()
    missing_gaps = time_diff[time_diff > pd.Timedelta('10 minutes')]
    
    pairs = []
    
    for index_lightgbm, gap in missing_gaps.items():
   
        gap = pd.Timedelta(gap)
    
        horizon = gap.days * 144 + gap.components.hours * 6 + gap.components.minutes / 10
        horizon = int(horizon) - 1
       
        pairs.append((index_lightgbm, horizon))
    
    #print("pairs are", pairs)
    return pairs

def create_lag_features(df, num_lags=3):
    for i in range(1, num_lags + 1):
        df[f'WindSpeed_lag{i}'] = df['WindSpeed'].shift(i)
    return df

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



def perform_forecasting(data, pairs, num_lags=3):
    all_predictions_df = []

    if not pairs:
        print("this dataframe has no gaps")
        
        df = data
       

        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.reset_index(drop=True)
        columns_to_drop = ['Index', "ActivePower", "Unnamed:0", "INDEX"]
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        
        
        result_df = df
        result_df.sort_values(by='DATE', inplace=True)

        merged_df = result_df
        merged_df.sort_values(by='DATE', inplace=True)
        print("merged df",df)
        
        merged_df = merged_df[['WF', "DATE",'WT', 'WindSpeed']]
        merged_df.reset_index(drop=True, inplace=True)
        #plot_results(result_df, merged_df)

        print("data.shape, result_df.shape, merged_df.shape", data.shape, result_df.shape, merged_df.shape)

        return result_df, merged_df

    for index_lightgbm, forecast_horizon in pairs:
        df = data.copy()  # Make a copy to avoid modifying the original data
        wf_value = df['WF'].iloc[0]
        wt_value = int(df['WT'].iloc[0])

        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.reset_index(drop=True)
        columns_to_drop = ['WF', 'WT', 'Index', "ActivePower", "Unnamed:0", "INDEX"]
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        df = df.set_index('DATE')

        df = create_lag_features(df, num_lags)

        df = df[:index_lightgbm]

        train_size = int(index_lightgbm * 0.92)
        train = df.iloc[:train_size, :]
        test = df.iloc[train_size:, :]

        y_train = train['WindSpeed']
        X_train = train.drop('WindSpeed', axis=1)

        X_test = test.copy()

        ##print("test", test)
        for i in range(1, num_lags + 1):
            X_test[f'WindSpeed_lag{i}'] = test['WindSpeed'].iloc[-i]

        X_test = X_test.drop('WindSpeed', axis=1)

        model = train_model(X_train, y_train)

        for step in range(forecast_horizon):
            forecast = model.predict(X_test.iloc[-1:, :])

            for i in range(1, num_lags + 1):
                X_test[f'WindSpeed_lag{i}'].iloc[-1] = test['WindSpeed'].iloc[-i - 1]

            test = pd.concat([test, pd.DataFrame({'WindSpeed': forecast}, index=[test.index[-1] + pd.DateOffset(minutes=10)])])

        predictions = test['WindSpeed'][-forecast_horizon:]
        predictions_df = pd.DataFrame({'DATE': predictions.index, 'WindSpeed': predictions.values})
        predictions_df['DATE'] = pd.to_datetime(predictions_df['DATE'])

        predictions_df['WF'] = wf_value
        predictions_df['WT'] = wt_value

        all_predictions_df.append(predictions_df)

    result_df = pd.concat(all_predictions_df, ignore_index=True)
    result_df.sort_values(by='DATE', inplace=True)

    merged_df = pd.concat([data, result_df], ignore_index=True)
    merged_df.sort_values(by='DATE', inplace=True)

    merged_df = merged_df[['WF', 'DATE', 'WT', 'WindSpeed']]
    merged_df.reset_index(drop=True, inplace=True)
    plot_results(result_df, merged_df)

    print("data.shape, result_df.shape, merged_df.shape", data.shape, result_df.shape, merged_df.shape)

    return result_df, merged_df


    



def perform_multistep_forecasting(df_original, num_lags=3):
    all_merged_dataframes = {}
    
    for key, df in df_original.items():
        pairs = detect_index_horizon(df)
        
        result_df, merged_df = perform_forecasting(df, pairs, num_lags)
    
        all_merged_dataframes[key] = merged_df
    
    return all_merged_dataframes
