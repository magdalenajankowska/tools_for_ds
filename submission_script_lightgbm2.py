import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train_df = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")
train_df = train_df[["counter_name", "date", "latitude", "longitude", "log_bike_count"]]

test_df = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")


def add_covid(data):
    url = "https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv"

    # Load the CSV file directly into a Pandas DataFrame
    df = pd.read_csv(url)

    # Filter the DataFrame for France
    covid_df = df[
        df["countriesAndTerritories"] == "France"
    ]  # Replace 'country' with the actual column name if different

    # Display the first few rows of the France DataFrame
    print(covid_df.head())

    covid_df["date"] = pd.to_datetime(covid_df["dateRep"], format="%d/%m/%Y")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # Extract the date (without time) from the bike data
    df["date_only"] = df["date"].dt.date

    # Ensure the same for COVID data (extract date without time)
    covid_df["date_only"] = covid_df["date"].dt.date
    covid_df["covid_cases"] = covid_df["cases"]

    # Merge the COVID data with your bike data based on the date (ignoring the time part)
    merged_df = pd.merge(
        df, covid_df[["date_only", "covid_cases"]], on="date_only", how="left"
    )

    # Drop the 'date_only' column as it's no longer needed
    merged_df.drop("date_only", axis=1, inplace=True)

    # Calculate the 7-day rolling average of new COVID cases
    merged_df["7_day_rolling_avg_covid"] = (
        merged_df["covid_cases"].rolling(window=7, min_periods=1).mean()
    )
    return merged_df


train_df = add_covid(train_df)
test_df = add_covid(test_df)

weather_df = pd.read_csv("/kaggle/input/msdb-2024/external_data.csv")
# Drop columns with many nan values
threshold = len(weather_df) * 0.8
weather_df = weather_df.dropna(axis=1, thresh=threshold)
weather_df = weather_df[
    [
        "date",
        "t",
        "ff",
        "pres",
        "rafper",
        "u",
        "vv",
        "rr1",
        "rr3",
        "rr6",
        "rr12",
        "td",
        "ww",
        "raf10",
        "etat_sol",
    ]
]

# Drop rows with any NaN values
weather_df = weather_df.dropna()

# Replace negative values in the 'rr1' column with 0
weather_df["rr1"] = weather_df["rr1"].apply(lambda x: max(x, 0))

weather_df["date"] = pd.to_datetime(weather_df["date"])
weather_df.set_index("date", inplace=True)

# Resample to 1-hour intervals and interpolate missing data
weather_df = weather_df.resample("1H").mean().interpolate(method="linear")

# Reset the index to make 'date' a column again
weather_df.reset_index(inplace=True)

# Merge the DataFrames on the 'date' column
merged_df = pd.merge(train_df, weather_df, on="date", how="inner")

# Merge the DataFrames on the 'date' column
merged_df_test = pd.merge(test_df, weather_df, on="date", how="inner")

holidays = holidays.CountryHoliday("France")


def is_holiday(date):  # 1: holiday, 0: not holiday
    return 1 if date in holidays else 0


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["day_of_week"] = X["date"].dt.dayofweek

    X["is_weekend"] = X["day_of_week"].apply(
        lambda x: 1 if x >= 5 else 0
    )  # 1: weekend, 0: weekday
    X["is_holiday"] = X["date"].apply(is_holiday)

    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["day_of_week_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["day_of_week_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _encode_features(X):
    X = X.copy()
    # X['temp_hour'] = X['t'] * X['hour_sin']
    X["temp_hour"] = X["t"] * X["hour"]
    X["weekend_temp"] = X["t"] * X["is_weekend"]

    # Create comfort index (simplified version of feels-like temperature)
    X["comfort_index"] = X["t"] - 0.55 * (1 - X["u"] / 100) * (X["t"] - 14)

    # Rain intensity categories
    X["rain_intensity"] = (
        (X["rr1"] > 0).astype(int)
        + (X["rr3"] > 0).astype(int)
        + (X["rr6"] > 0).astype(int)
        + (X["rr12"] > 0).astype(int)
    )

    # Wind categories
    X["high_wind"] = (X["ff"] > X["ff"].mean() + X["ff"].std()).astype(int)
    return X.drop(columns=["rr3", "rr6", "rr12"])


# Encode date features
merged_df = _encode_dates(merged_df)
merged_df = _encode_features(merged_df)

# Define feature columns
categorical_columns = ["counter_name", "is_weekend", "is_holiday"]
numerical_columns = [
    "latitude",
    "longitude",
    "t",
    "ff",
    "pres",
    "rafper",
    "u",
    "vv",
    "rr1",
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "day_of_week",
    "temp_hour",
    "weekend_temp",
    "comfort_index",
    "rain_intensity",
    "high_wind",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "7_day_rolling_avg_covid",
    "covid_cases",
    "td",
    "ww",
    "raf10",
    "etat_sol",
]
target_column = "log_bike_count"

# Split data into features and target
X = merged_df[categorical_columns + numerical_columns]
y = merged_df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing for numerical and categorical columns
numerical_preprocessor = StandardScaler()
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_preprocessor, numerical_columns),
        ("cat", categorical_preprocessor, categorical_columns),
    ]
)


# Define the LightGBM model
lgb_model = LGBMRegressor(objective="regression", random_state=42)

# Create the pipeline with LightGBM
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", lgb_model)])

# Train the LightGBM model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# Define the parameter grid for LightGBM
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__num_leaves": [31, 50, 70],  # Controls tree complexity
    "model__max_depth": [-1, 10, 20],  # -1 means no limit
    "model__min_child_samples": [10, 20, 30],  # Minimum data per leaf
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

# Predictions on test data
merged_df_test = _encode_dates(merged_df_test)
merged_df_test = _encode_features(merged_df_test)
X_pred = merged_df_test
y_pred = best_model.predict(X_pred)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
