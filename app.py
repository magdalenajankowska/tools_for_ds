import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk


# Load data
@st.cache_data
def load_data(start_date=None, end_date=None):
    df = pd.read_parquet("data/train.parquet")
    df["date"] = pd.to_datetime(df["date"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df


# Streamlit app layout
st.title("Bike Counter Analysis")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")
df = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache_data)")

# Display data

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(df)


# Map
st.subheader("Interactive Bike Locations Map")

map_df = df.sample(frac=0.1, random_state=42)

# st.map(map_df[['latitude', 'longitude']])


# Create a Pydeck layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[longitude, latitude]",
    get_color="[200, 30, 0, 160]",
    get_radius=50,
    pickable=True,
)

# Set the initial view
view_state = pdk.ViewState(
    latitude=map_df["latitude"].mean(),
    longitude=map_df["longitude"].mean(),
    zoom=12,
    pitch=0,
)

# Render the deck
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Lat: {latitude}\nLon: {longitude}"},
)
st.pydeck_chart(r)


# Histograms of numerical features
st.subheader("Histograms of Numerical Features")

numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns

fig, axes = plt.subplots(
    nrows=len(numerical_columns), ncols=1, figsize=(12, 4 * len(numerical_columns))
)

# If there's only one numerical column, axes is not an array
if len(numerical_columns) == 1:
    axes = [axes]

for ax, col in zip(axes, numerical_columns):
    df[col].hist(ax=ax, bins=20, edgecolor="black")
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

plt.tight_layout()
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap of Numerical Features")

numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
corr_matrix = df[numerical_columns].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap", fontsize=16)
st.pyplot(fig)

# Bike counts over time
st.subheader("Bike Count Over Time")

fig, ax = plt.subplots(figsize=(12, 6))
df.groupby("date")["bike_count"].sum().plot(ax=ax)
ax.set_title("Bike Count Over Time", fontsize=16)
ax.set_ylabel("Bike Count")
ax.set_xlabel("Date")
st.pyplot(fig)


@st.cache_data
def load_weather_data():
    df = pd.read_csv("external_data/external_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


weather_df = load_weather_data()

# Weather features w.r.t. time
st.subheader("Weather Features Over Time")

features = ["t", "pmer", "td", "u", "ff"]

fig, axes = plt.subplots(len(features), 1, figsize=(15, 3 * len(features)), sharex=True)

for ax, feature in zip(axes, features):
    sns.lineplot(data=weather_df, x="date", y=feature, ax=ax)
    ax.set_title(f"{feature} Over Time")
    ax.set_ylabel(feature)

axes[-1].set_xlabel("Date")
plt.tight_layout()
st.pyplot(fig)
