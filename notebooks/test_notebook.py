# Databricks notebook source
# MAGIC %md
# MAGIC # Hotel Reservation Prediction Exercise
# MAGIC
# MAGIC This notebook demonstrates how to predict Hotel Reservation Status using the Hotel Reservation dataset. We'll go through the process of loading data, preprocessing, model creation, and visualization of results.
# MAGIC
# MAGIC ## Importing Required Libraries
# MAGIC
# MAGIC First, let's import all the necessary libraries.

# COMMAND ----------
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# COMMAND ----------

# Only works in a Databricks environment if the data is there
filepath = "/Volumes/mlops_dev/subhadip/hotel_reservation_data/Hotel_Reservations.csv"
# Load the data
df = pd.read_csv(filepath)

# Works both locally and in a Databricks environment
filepath = "../data/Hotel_Reservations.csv"
# Load the data
df = pd.read_csv(filepath)
df.head(2)

# COMMAND ----------

# Load configuration
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print(config.get("catalog_name"))
num_features = config.get("num_features")
print(num_features)


# MAGIC ## Preprocessing

# COMMAND ----------

# Remove rows with missing target

# Handle missing values and convert data types as needed
df["no_of_adults"] = pd.to_numeric(df["no_of_adults"], errors="coerce")

df["no_of_children"] = pd.to_numeric(df["no_of_children"], errors="coerce")
df["avg_price_per_room"] = pd.to_numeric(df["avg_price_per_room"], errors="coerce")

median_no_of_previous_cancellations = df["no_of_previous_cancellations"].median()
df["no_of_previous_cancellations"].fillna(median_no_of_previous_cancellations, inplace=True)

# Handle numeric features
num_features = config.get("num_features")
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values with mean or default values
df.fillna(
    {
        "no_of_children": df["no_of_children"].mean(),
        "type_of_meal_plan": "None",
        "no_of_special_requests": 0,
    },
    inplace=True,
)

# Convert categorical features to the appropriate type
cat_features = config.get("cat_features")
for cat_col in cat_features:
    df[cat_col] = df[cat_col].astype("category")

# Extract target and relevant features
target = config.get("target")
# relevant_columns = cat_features + num_features + [target]

df["Id"] = range(1, len(df) + 1)
relevant_columns = cat_features + num_features + [target] + ["Id"]
print(relevant_columns)

df = df[relevant_columns]
df["Id"] = df["Id"].astype("str")
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

df.head(2)
