import pandas as pd

url = "https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv"

# Load the CSV file directly into a Pandas DataFrame
df = pd.read_csv(url)

# Filter the DataFrame for France
covid_df = df[df['countriesAndTerritories'] == 'France']  # Replace 'country' with the actual column name if different

# Display the first few rows of the France DataFrame
print(covid_df.head())

covid_df['date'] = pd.to_datetime(covid_df['dateRep'], format='%d/%m/%Y')

# Display the updated DataFrame
print(covid_df.head())


'''# Load the COVID-19 data from the URL
covid_url = "https://static.data.gouv.fr/resources/donnees-relatives-a-lepidemie-de-covid-19-en-france-vue-densemble/20220517-222620/synthese-fra.csv"
covid_df = pd.read_csv(covid_url, delimiter=",")

# Ensure the 'date' column is in datetime format
covid_df['date'] = pd.to_datetime(covid_df['date'])

# Calculate the daily new cases by subtracting the previous day's total confirmed cases
covid_df['new_covid_cases'] = covid_df['total_cas_confirmes'].diff().fillna(0)'''

# Step 3: Load your bike count data
# Assuming your data is already loaded in df (as per the example)
# For demonstration, let's create a similar dataframe
data = pd.read_parquet("data/train.parquet")

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Step 4: Extract the date (without time) from the bike data
df['date_only'] = df['date'].dt.date

# Ensure the same for COVID data (extract date without time)
covid_df['date_only'] = covid_df['date'].dt.date

# Merge the COVID data with your bike data based on the date (ignoring the time part)
merged_df = pd.merge(df, covid_df[['date_only', 'cases']], on='date_only', how='left')

# Drop the 'date_only' column as it's no longer needed
merged_df.drop('date_only', axis=1, inplace=True)

# Calculate the 7-day rolling average of new COVID cases
merged_df['7_day_rolling_avg_covid'] = merged_df['cases'].rolling(window=7, min_periods=1).mean()

# Display the final DataFrame
print(merged_df.head(30))
print(merged_df.columns)
