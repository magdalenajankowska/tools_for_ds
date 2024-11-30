import pandas as pd
import numpy as np
import holidays
from joblib import load

test_data = pd.read_parquet('data/final_test.parquet')
selected_columns2 = ['counter_id', 'counter_installation_date','latitude', 'longitude']
new_test_df = test_data[selected_columns2]

new_test_df['datetime'] = pd.to_datetime(test_data['date'])

# new_train_df['date'] = pd.to_datetime(new_train_df['datetime'])

new_test_df['day_of_month'] = new_test_df['datetime'].dt.day
new_test_df['day_of_week'] = new_test_df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
new_test_df['hour'] = new_test_df['datetime'].dt.hour
new_test_df['day_of_year'] = new_test_df['datetime'].dt.dayofyear
new_test_df['week_number'] = new_test_df['datetime'].dt.isocalendar().week  # week number
new_test_df['month'] = new_test_df['datetime'].dt.month
new_test_df['year'] = new_test_df['datetime'].dt.year

new_test_df['is_weekend'] = new_test_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) # 1: weekend, 0: weekday


holidays = holidays.CountryHoliday('France')

# 1: holiday, 0: not holiday


def is_holiday(date):

    return 1 if date in holidays else 0


new_test_df['is_holiday'] = new_test_df['datetime'].apply(is_holiday)


new_test_df.set_index('datetime', inplace=True)

test_data['date'] = pd.to_datetime(test_data['date'])
test_data.set_index('counter_id', inplace=True)

# weather data


weather_df = pd.read_csv('external_data/external_data.csv')

weather_df = weather_df.dropna(axis=1, how='all')

columns_without_2000_non_nulls = weather_df.notnull().sum()[weather_df.notnull().sum() <= 2000].index
weather_df.drop(columns=columns_without_2000_non_nulls, inplace=True)

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df.set_index('date', inplace=True)

# Check for duplicate dates in the index
duplicate_dates = weather_df.index[weather_df.index.duplicated()]
print("Duplicate dates found:", duplicate_dates)

# Optionally, drop duplicate dates
weather_df = weather_df[~weather_df.index.duplicated(keep='first')]

# Resample to 1-hour intervals and interpolate missing values
weather_df_res = weather_df.resample('1H').interpolate(method='linear')

# Perform inner join on index to keep only matching dates

merged_test_df = weather_df_res.join(new_test_df, how='inner', lsuffix='_df1', rsuffix='_df2')

merged_test_df['hour_sin'] = np.sin(2 * np.pi * merged_test_df['hour']/24)
merged_test_df['hour_cos'] = np.cos(2 * np.pi * merged_test_df['hour']/24)
merged_test_df['day_of_week_sin'] = np.sin(2 * np.pi * merged_test_df['day_of_week']/7)
merged_test_df['day_of_week_cos'] = np.cos(2 * np.pi * merged_test_df['day_of_week']/7)
merged_test_df['month_sin'] = np.sin(2 * np.pi * merged_test_df['month']/12)
merged_test_df['month_cos'] = np.cos(2 * np.pi * merged_test_df['month']/12)

# Create interaction features
merged_test_df['temp_hour'] = merged_test_df['t'] * merged_test_df['hour_sin']
merged_test_df['weekend_temp'] = merged_test_df['t'] * merged_test_df['is_weekend']


# Create comfort index (simplified version of feels-like temperature)
merged_test_df['comfort_index'] = merged_test_df['t'] - 0.55 * (1 - merged_test_df['u']/100) * (merged_test_df['t'] - 14)

# Rain intensity categories
merged_test_df['rain_intensity'] = (merged_test_df['rr1'] > 0).astype(int) + \
                        (merged_test_df['rr3'] > 0).astype(int) + \
                        (merged_test_df['rr6'] > 0).astype(int) + \
                        (merged_test_df['rr12'] > 0).astype(int)

# Wind categories
merged_test_df['high_wind'] = (merged_test_df['ff'] > merged_test_df['ff'].mean() + merged_test_df['ff'].std()).astype(int)

print(merged_test_df.columns)

columns_to_drop = ['log_bike_count', 'bike_count']
merged_test_df = merged_test_df.drop(
    columns=[col for col in columns_to_drop if col in merged_test_df.columns]
)


#  Load the trained pipeline

loaded_model = load("trained_pipeline.joblib")

predictions = loaded_model.predict(merged_test_df)

submission = pd.DataFrame({
    'Id': test_data.index,  # Use appropriate index or ID column from the test data
    'log_bike_count': predictions
})
submission.to_csv('submission.csv', index=False)