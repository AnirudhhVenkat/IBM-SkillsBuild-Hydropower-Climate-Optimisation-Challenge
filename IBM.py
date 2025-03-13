import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import gc
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')

print("Starting data loading and preprocessing...")

# Load data
all_data_df = pd.read_csv(r"C:\Users\aniru\Downloads\Data\Data.csv")
ss = pd.read_csv(r"C:\Users\aniru\Downloads\SampleSubmission.csv")
print(f"Loaded data: {all_data_df.shape} rows, {ss.shape} rows in sample submission")

# Split 'Source' into 'consumer_device' and 'data_user'
all_data_df[['consumer_device', 'data_user']] = all_data_df['Source'].str.extract(r'(consumer_device_\d+)_data_user_(\d+)')
print(f"Unique consumer devices: {all_data_df['consumer_device'].nunique()}")
print(f"Unique data users: {all_data_df['data_user'].nunique()}")

# Drop the consumer_device_9 column if it exists
if 'consumer_device_9' in all_data_df.columns:
    all_data_df = all_data_df.drop('consumer_device_9', axis=1)
    print("Dropped column: 'consumer_device_9'")

# Define devices to drop
devices_to_drop = ["consumer_device_3","consumer_device_5","consumer_device_11", "consumer_device_14",
                   "consumer_device_15", "consumer_device_17", "consumer_device_24",
                   "consumer_device_25","consumer_device_27","consumer_device_33","consumer_device_4","consumer_device_9"]
print(f"Dropping {len(devices_to_drop)} devices: {devices_to_drop}")

# Filter out specified devices
filtered_df = all_data_df[~all_data_df['consumer_device'].isin(devices_to_drop)]
print(f"After filtering: {filtered_df.shape} rows")

# Now drop the consumer_device column from filtered_df
filtered_df = filtered_df.drop('consumer_device', axis=1)
print("Dropped column: 'consumer_device' from filtered dataframe")

# Aggregate data by date and source
aggregated_data = filtered_df.groupby(['date_time', 'Source'])['kwh'].sum().reset_index()
aggregated_data['Date'] = pd.to_datetime(aggregated_data['date_time']).dt.date
print(f"After aggregation: {aggregated_data.shape} rows")

# Find min and max dates
min_date = aggregated_data['Date'].min()
max_date = aggregated_data['Date'].max()
print(f"Date range: {min_date} to {max_date}")

# Create complete date range and fill missing values
date_rng = pd.date_range(start=min_date, end=max_date, freq='D')
complete_data = pd.DataFrame()

print("Filling missing dates...")
# Fill missing dates with 0 kwh
for source in aggregated_data['Source'].unique():
    source_data = aggregated_data[aggregated_data['Source'] == source].copy()
    source_data['Date'] = pd.to_datetime(source_data['Date'])
    
    source_date_rng = pd.DataFrame({'Date': date_rng})
    source_date_rng['Source'] = source
    
    source_data = pd.merge(source_date_rng, source_data, on=['Date', 'Source'], how='left')
    source_data['kwh'] = source_data['kwh'].fillna(0)
    
    complete_data = pd.concat([complete_data, source_data], ignore_index=True)

print(f"Complete data shape after filling missing dates: {complete_data.shape}")
print(f"Number of unique sources in complete data: {complete_data['Source'].nunique()}")

def forecast_arima(all_data, forecast_horizon=30, output_template=None):
    print(f"Starting ARIMA forecasting for {forecast_horizon} days...")
    all_data['Date'] = pd.to_datetime(all_data['Date'])
    all_data[['consumer_device', 'data_user']] = all_data['Source'].str.extract(r'consumer_device_(\d+)_data_user_(\d+)')
    all_data = all_data.sort_values(by=['consumer_device', 'data_user', 'Date'])
    
    forecast_results = []
    total_combinations = all_data.groupby(["consumer_device", "data_user"]).ngroups
    print(f"Total device-user combinations to process: {total_combinations}")
    
    processed = 0
    successful = 0
    failed = 0
    
    for (consumer_device, data_user), group in all_data.groupby(["consumer_device", "data_user"]):
        processed += 1
        if processed % 10 == 0:
            print(f"Processing combination {processed}/{total_combinations}: device_{consumer_device}, user_{data_user}")
        
        # Set the Date as index
        group = group.set_index("Date")
        
        # Check for and handle duplicate index values
        if group.index.duplicated().any():
            print(f"Warning: Found duplicate dates in data for {source}. Aggregating by mean.")
            # Only apply mean to numeric columns
            numeric_cols = group.select_dtypes(include=['number']).columns
            if 'kwh' in numeric_cols:
                group = group[['kwh']].groupby(level=0).mean()
            else:
                # If kwh is not numeric, convert it first
                try:
                    group['kwh'] = pd.to_numeric(group['kwh'])
                    group = group[['kwh']].groupby(level=0).mean()
                except:
                    print(f"Error: Could not convert kwh to numeric for {source}")
                    continue
        
        group = group.asfreq('D').fillna(method='ffill')
        
        try:
            print(f"Fitting ARIMA(5,1,0) for device_{consumer_device}, user_{data_user} with {len(group)} observations")
            model = ARIMA(group["kwh"], order=(5, 1, 0))
            fitted_model = model.fit()
            
            forecast_dates = pd.date_range(start=group.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_horizon, freq='D')
            forecast_values = fitted_model.forecast(steps=forecast_horizon)
            
            forecast_df = pd.DataFrame({
                "ID": [f"{date.strftime('%Y-%m-%d')}_consumer_device_{consumer_device}_data_user_{data_user}"
                        for date in forecast_dates],
                "kwh": forecast_values
            })
            
            forecast_results.append(forecast_df)
            successful += 1
            
        except Exception as e:
            print(f"Error processing device_{consumer_device}, user_{data_user}: {e}")
            failed += 1
    
    print(f"Forecasting complete. Successful: {successful}, Failed: {failed}")
    forecast_df = pd.concat(forecast_results, ignore_index=True)
    print(f"Combined forecast shape: {forecast_df.shape}")
    
    if output_template is not None:
        output_template = output_template.drop(columns=['kwh'], errors='ignore')
        final_output = output_template.merge(forecast_df, on='ID', how='left').fillna(0)
        print(f"Final output shape after merging with template: {final_output.shape}")
    else:
        final_output = forecast_df
    
    return final_output

print("Starting forecasting process...")
# Generate forecasts
forecast = forecast_arima(all_data=complete_data, forecast_horizon=30, output_template=ss)

# Replace any NaN values with 0
forecast["kwh"] = forecast["kwh"].fillna(0)
print(f"Final forecast shape: {forecast.shape}")
print(f"Sample of forecasts:\n{forecast.head()}")

# Save forecasts
forecast.to_csv("forecast.csv", index=False)
print("Saved forecasts to forecast.csv")

# Calculate RMSE
merged_df = pd.merge(forecast, ss, on='ID', how='left', suffixes=('_forecast', '_actual'))
rmse = math.sqrt(mean_squared_error(merged_df['kwh_actual'], merged_df['kwh_forecast']))
print(f"RMSE: {rmse}")

print("Script execution complete.")