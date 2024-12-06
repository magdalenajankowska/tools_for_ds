import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import holidays

# Load and preprocess data
#train_df = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")
train_df = pd.read_parquet("data/train.parquet")
train_df = train_df[['counter_name', 'date', 'latitude', 'longitude', 'log_bike_count']]

#test_df = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")
test_df = pd.read_parquet("data/final_test.parquet")

#weather_df = pd.read_csv("/kaggle/input/msdb-2024/external_data.csv")
weather_df = pd.read_csv("external_data/external_data.csv")
# Drop columns with many NaN values
threshold = len(weather_df) * 0.85
weather_df = weather_df.dropna(axis=1, thresh=threshold)
weather_df = weather_df[["date", "t", "ff", "pres", "rafper", "u", "vv", "rr1",
                         "rr3", "rr6", "rr12", 'td', 'ww', 'raf10', 'etat_sol']]
# Drop rows with any NaN values
weather_df = weather_df.dropna()

# Replace negative values in the 'rr1' column with 0
for col in ['rr1', 'rr3', 'rr6', 'rr12']:
    weather_df[col] = weather_df[col].apply(lambda x: max(x, 0))

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df.set_index('date', inplace=True)

# Resample to 1-hour intervals and interpolate missing data
weather_df = weather_df.resample('1H').mean().interpolate(method='linear')

# Reset the index to make 'date' a column again
weather_df.reset_index(inplace=True)

# Merge the DataFrames on the 'date' column
merged_df = pd.merge(train_df, weather_df, on='date', how='inner')
merged_df_test = pd.merge(test_df, weather_df, on='date', how='inner')

# Add holiday feature
holidays = holidays.CountryHoliday('France')

def is_holiday(date):
    return 1 if date in holidays else 0

# Encode date-related features
def _encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X['is_weekend'] = X['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    X['is_holiday'] = X['date'].apply(is_holiday)
    return X.drop(columns=["date"])

# Feature engineering
def _encode_features(X):
    X = X.copy()
    X['temp_hour'] = X['t'] * X['hour']
    X['weekend_temp'] = X['t'] * X['is_weekend']
    X['comfort_index'] = X['t'] - 0.55 * (1 - X['u']/100) * (X['t'] - 14)
    X['rain_intensity'] = (X['rr1'] > 0).astype(int) + \
                          (X['rr3'] > 0).astype(int) + \
                          (X['rr6'] > 0).astype(int) + \
                          (X['rr12'] > 0).astype(int)
    X['high_wind'] = (X['ff'] > X['ff'].mean() + X['ff'].std()).astype(int)
    return X.drop(columns=["rr3", "rr6", "rr12"])

merged_df = _encode_dates(merged_df)
merged_df = _encode_features(merged_df)

merged_df_test = _encode_dates(merged_df_test)
merged_df_test = _encode_features(merged_df_test)

# Define feature columns
categorical_columns = ['counter_name', 'is_weekend', 'is_holiday']
numerical_columns = ["latitude", "longitude", "t", "ff", "pres", "rafper",
                     "u", "vv", "rr1", "year", "month", "day", "weekday",
                     "hour", 'temp_hour', 'weekend_temp', 'comfort_index',
                     'rain_intensity', 'high_wind', 'td', 'ww', 'raf10', 'etat_sol']
target_column = "log_bike_count"

# Split data into features and target
X = merged_df[categorical_columns + numerical_columns]
y = merged_df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical and categorical columns
numerical_preprocessor = StandardScaler()
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_preprocessor, numerical_columns),
        ("cat", categorical_preprocessor, categorical_columns),
    ]
)

# Define XGBoost model for feature selection
xgb_feature_selector = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_feature_selector.fit(preprocessor.fit_transform(X_train), y_train)

# Select important features
feature_selector = SelectFromModel(xgb_feature_selector, threshold="median", prefit=True)

# Transform datasets
X_train_selected = feature_selector.transform(preprocessor.transform(X_train))
X_test_selected = feature_selector.transform(preprocessor.transform(X_test))

# Define XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Train and evaluate the model
xgb_model.fit(X_train_selected, y_train)
y_pred = xgb_model.predict(X_test_selected)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# GridSearchCV for tuning
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectFromModel(xgb.XGBRegressor(objective="reg:squarederror", random_state=42))),
    ("model", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
])

param_grid = {
    "feature_selection__threshold": ["mean", "median", 0.01, 0.02],
    "model__n_estimators": [250, 300, 350],
    "model__max_depth": [8, 9, 10],
    "model__learning_rate": [0.25, 0.3, 0.4]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(preprocessor.transform(merged_df_test))
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred_best.shape[0]),
        log_bike_count=y_pred_best,
    )
)
results.to_csv("submission.csv", index=False)

print(f"Best Model RMSE: {rmse_best}")
print(f'Feature importance: {best_model.feature_importances_}')
