import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import holidays

# All previous data loading and preprocessing remain exactly the same
#train_df = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")
train_df = pd.read_parquet("data/train.parquet")
train_df = train_df[['counter_name', 'date', 'latitude', 'longitude', 'log_bike_count']]

#test_df = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")
test_df = pd.read_parquet("data/final_test.parquet")

#weather_df = pd.read_csv("/kaggle/input/msdb-2024/external_data.csv")
weather_df = pd.read_csv("external_data/external_data.csv")
threshold = len(weather_df) * 0.8
weather_df = weather_df.dropna(axis=1, thresh=threshold)
weather_df = weather_df[["date", "t","ff", "pres", "rafper", "u", "vv", "rr1", "rr3", "rr6", "rr12"]]
weather_df['rr1'] = weather_df['rr1'].apply(lambda x: max(x, 0))
weather_df['rr3'] = weather_df['rr3'].apply(lambda x: max(x, 0))
weather_df['rr6'] = weather_df['rr6'].apply(lambda x: max(x, 0))
weather_df['rr12'] = weather_df['rr12'].apply(lambda x: max(x, 0))

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df.set_index('date', inplace=True)
weather_df = weather_df.resample('1H').mean().interpolate(method='linear')
weather_df.reset_index(inplace=True)

merged_df = pd.merge(train_df, weather_df, on='date', how='inner')

# Existing holidays and encoding functions remain the same
holidays = holidays.CountryHoliday('France')
def is_holiday(date):
    return 1 if date in holidays else 0

def _encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X['day_of_week'] = X['date'].dt.dayofweek

    X['is_weekend'] = X['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    X['is_holiday'] = X['date'].apply(is_holiday)

    X['hour_sin'] = np.sin(2 * np.pi * X['hour']/24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour']/24)
    X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week']/7)
    X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week']/7)
    X['month_sin'] = np.sin(2 * np.pi * X['month']/12)
    X['month_cos'] = np.cos(2 * np.pi * X['month']/12)

    return X.drop(columns=["date"])

def _encode_features(X):
    X = X.copy()
    X['temp_hour'] = X['t'] * X['hour_sin']
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

# Define feature columns (same as before)
categorical_columns = ['counter_name', 'is_weekend', 'is_holiday']
numerical_columns = ["latitude", "longitude", "t", "ff", "pres", "rafper",
                     "u", "vv", "rr1", "year", "month", "day", "weekday",
                     "hour", 'day_of_week', 'temp_hour', 'weekend_temp',
                     'comfort_index', 'rain_intensity', 'high_wind',
                     'hour_sin', 'hour_cos', 'day_of_week_sin',
                     'day_of_week_cos', 'month_sin', 'month_cos'
    ]
target_column = "log_bike_count"

# Prepare features and target
X = merged_df[categorical_columns + numerical_columns]
y = merged_df[target_column]

# New function for feature selection
def find_best_k_features(X, y, max_features=None):
    if max_features is None:
        max_features = len(X.columns)

    # Preprocessing for numerical and categorical columns
    numerical_preprocessor = StandardScaler()
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_preprocessor, numerical_columns),
            ("cat", categorical_preprocessor, categorical_columns),
        ]
    )

    # Create a list to store cross-validation scores
    cv_scores = []

    # Try different numbers of features
    for k in range(5, max_features + 1, 1):
        # Create pipeline with feature selection
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(f_regression, k=k)),
            ("model", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
        ])

        # Perform cross-validation
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=3,
            scoring="neg_mean_squared_error"
        )

        # Convert to RMSE and store
        rmse_scores = np.sqrt(-scores)
        cv_scores.append((k, np.mean(rmse_scores), np.std(rmse_scores)))

    # Find the best k
    best_k = min(cv_scores, key=lambda x: x[1])

    return cv_scores, best_k

# Find the best number of features
feature_selection_results, best_k = find_best_k_features(X, y)

# Print feature selection results
print("Feature Selection Results:")
for k, mean_rmse, std_rmse in feature_selection_results:
    print(f"k={k}: Mean RMSE = {mean_rmse:.4f} (±{std_rmse:.4f})")

print(f"\nBest number of features: {best_k[0]}")
print(f"Best Mean RMSE: {best_k[1]:.4f}")

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ]
)

# Create pipeline with feature selection
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(f_regression, k=best_k[0])),
    ("model", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
])

# Define the parameter grid (now including feature selection parameter)
param_grid = {
    "feature_selection__k": [best_k[0], best_k[0]+5, best_k[0]-5],  # Include best k and nearby values
    "model__n_estimators": [250, 300, 350],
    "model__max_depth": [8, 9, 10],
    "model__learning_rate": [0.25, 0.3, 0.4]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1,  # Use all available cores
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Evaluate the model with the best parameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
print(f"Best Model RMSE: {rmse_best}")

# Prediction on test data
merged_df_test = pd.merge(test_df, weather_df, on='date', how='inner')
merged_df_test = _encode_dates(merged_df_test)
merged_df_test = _encode_features(merged_df_test)
X_pred = merged_df_test[categorical_columns + numerical_columns]

# Predict and save results
y_pred = best_model.predict(X_pred)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
