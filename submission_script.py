import pandas as pd
import numpy as np
import holidays
from joblib import load

train_df = pd.read_parquet('data/train.parquet')
selected_columns = ['counter_id', 'bike_count','counter_installation_date','latitude', 'longitude', 'log_bike_count']
new_train_df = train_df[selected_columns]

new_train_df['datetime'] = pd.to_datetime(train_df['date'])

#new_train_df['date'] = pd.to_datetime(new_train_df['datetime'])
new_train_df['day_of_month'] = new_train_df['datetime'].dt.day
new_train_df['day_of_week'] = new_train_df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
new_train_df['hour'] = new_train_df['datetime'].dt.hour
new_train_df['day_of_year'] = new_train_df['datetime'].dt.dayofyear
new_train_df['week_number'] = new_train_df['datetime'].dt.isocalendar().week  # week number
new_train_df['month'] = new_train_df['datetime'].dt.month
new_train_df['year'] = new_train_df['datetime'].dt.year

new_train_df['is_weekend'] = new_train_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) # 1: weekend, 0: weekday

f_holidays = holidays.CountryHoliday('France')


def is_holiday(date):
    return 1 if date in f_holidays else 0


new_train_df['is_holiday'] = new_train_df['datetime'].apply(is_holiday)


new_train_df.set_index('datetime', inplace=True)

train_df['date'] = pd.to_datetime(train_df['date'])
train_df.set_index('counter_id', inplace=True)



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


f_holidays = holidays.CountryHoliday('France')

# 1: holiday, 0: not holiday


def is_holiday(date):

    return 1 if date in f_holidays else 0


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
merged_df = weather_df_res.join(new_train_df, how='inner', lsuffix='_df1', rsuffix='_df2')

merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['hour']/24)
merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['hour']/24)
merged_df['day_of_week_sin'] = np.sin(2 * np.pi * merged_df['day_of_week']/7)
merged_df['day_of_week_cos'] = np.cos(2 * np.pi * merged_df['day_of_week']/7)
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month']/12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month']/12)

# Create interaction features
merged_df['temp_hour'] = merged_df['t'] * merged_df['hour_sin']
merged_df['weekend_temp'] = merged_df['t'] * merged_df['is_weekend']


# Create comfort index (simplified version of feels-like temperature)
merged_df['comfort_index'] = merged_df['t'] - 0.55 * (1 - merged_df['u']/100) * (merged_df['t'] - 14)

# Rain intensity categories
merged_df['rain_intensity'] = (merged_df['rr1'] > 0).astype(int) + \
                        (merged_df['rr3'] > 0).astype(int) + \
                        (merged_df['rr6'] > 0).astype(int) + \
                        (merged_df['rr12'] > 0).astype(int)

# Wind categories
merged_df['high_wind'] = (merged_df['ff'] > merged_df['ff'].mean() + merged_df['ff'].std()).astype(int)
# Perform inner join on index to keep only matching dates

merged_test_df = weather_df_res.join(new_test_df, how='inner', lsuffix='_df1', rsuffix='_df2')

numeric_df = merged_df.select_dtypes(include=[np.number])

# Step 2: Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Step 3: Extract correlation with 'log_bike_count'
log_bike_correlation = correlation_matrix['log_bike_count'].sort_values(ascending=False)

# Display the correlation with log_bike_count
print("Correlation with log_bike_count:")
print(log_bike_correlation)

# Step 4: Visualization using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

limit = 0.05

# Step 1: Select numeric columns
numeric_df = merged_df.select_dtypes(include=[np.number])

# Step 2: Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Step 3: Extract correlation with 'log_bike_count'
log_bike_correlation = correlation_matrix['log_bike_count'].sort_values(ascending=False)

# Filter features with correlation >= 0.01
filtered_features = log_bike_correlation[log_bike_correlation.abs() >= limit].index

# Filter the correlation matrix to include only the filtered features
filtered_correlation_matrix = correlation_matrix.loc[filtered_features, filtered_features]

# Display the filtered correlation with log_bike_count
print("Filtered Correlation with log_bike_count:")
print(log_bike_correlation[log_bike_correlation.abs() >= limit])

# Step 4: Visualization using a heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(filtered_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Filtered Correlation Matrix')
plt.show()

# Identify columns with correlation <= 0.01 (in absolute value)
low_correlation_features = log_bike_correlation[log_bike_correlation.abs() <= limit].index

# Display the names of these columns
print("Columns with correlation <= 0.01 with log_bike_count:")
print(low_correlation_features)

# Step 5: Drop these columns from merged_df
merged_df_dropped = merged_df.drop(columns=low_correlation_features)


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_regression

'''def create_time_features(df):
    """Create additional time-based features"""
    df = df.copy()
    # Create cyclical features for time components
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # Create interaction features
    df['temp_hour'] = df['t'] * df['hour_sin']
    df['weekend_temp'] = df['t'] * df['is_weekend']

    return df

def create_weather_features(df):
    """Create additional weather-related features"""
    df = df.copy()
    # Create comfort index (simplified version of feels-like temperature)
    df['comfort_index'] = df['t'] - 0.55 * (1 - df['u']/100) * (df['t'] - 14)

    # Rain intensity categories
    df['rain_intensity'] = (df['rr1'] > 0).astype(int) + \
                          (df['rr3'] > 0).astype(int) + \
                          (df['rr6'] > 0).astype(int) + \
                          (df['rr12'] > 0).astype(int)

    # Wind categories
    df['high_wind'] = (df['ff'] > df['ff'].mean() + df['ff'].std()).astype(int)

    return df'''

# Define feature groups
numeric_features = ['t', 'td', 'u', 'ff', 'vv', 'pres', 'raf10',
                   'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']
categorical_features = ['counter_id', 'ww', 'etat_sol']

def build_model_pipeline():
    # Create preprocessing steps
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create feature engineering steps
    '''feature_engineering = Pipeline([
        ('time_features', FunctionTransformer(create_time_features)),
        ('weather_features', FunctionTransformer(create_weather_features))
    ])'''

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create final pipeline
    pipeline = Pipeline([
        #('feature_engineering', feature_engineering),
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression, k=50)),
        ('regressor', xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

    return pipeline

def evaluate_model(pipeline, X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return pipeline

# Example usage:
pipeline = build_model_pipeline()
columns_to_exclude = ['bike_count', 'log_bike_count']
X = merged_df.drop(columns=columns_to_exclude)
y = merged_df['log_bike_count']
model = evaluate_model(pipeline, X, y)

from joblib import dump
dump(model, 'trained_pipeline.joblib')

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

required_columns = [col for col in merged_df.columns if (col != 'log_bike_count' and col != 'bike_count')]

merged_test_df = merged_test_df[required_columns]

#  Load the trained pipeline

loaded_model = load("trained_pipeline.joblib")

predictions = loaded_model.predict(merged_test_df)

submission = pd.DataFrame({
    'Id': test_data.index,  # Use appropriate index or ID column from the test data
    'log_bike_count': predictions
})
submission.to_csv('submission.csv', index=False)