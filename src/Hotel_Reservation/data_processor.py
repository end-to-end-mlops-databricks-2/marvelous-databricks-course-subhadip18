import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from Hotel_Reservation.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # Handle missing values and convert data types as needed

        self.df["no_of_adults"] = pd.to_numeric(self.df["no_of_adults"], errors="coerce")

        self.df["no_of_children"] = pd.to_numeric(self.df["no_of_children"], errors="coerce")
        self.df["avg_price_per_room"] = pd.to_numeric(self.df["avg_price_per_room"], errors="coerce")

        median_no_of_previous_cancellations = self.df["no_of_previous_cancellations"].median()
        self.df["no_of_previous_cancellations"].fillna(median_no_of_previous_cancellations, inplace=True)

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Fill missing values with mean or default values
        self.df.fillna(
            {
                "no_of_children": self.df["no_of_children"].mean(),
                "type_of_meal_plan": "None",
                "no_of_special_requests": 0,
            },
            inplace=True,
        )

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Extract target and relevant features
        target = self.config.target
        self.df[target] = self.df[target].replace({"Not_Canceled": 0, "Canceled": 1})
        self.df[target] = pd.to_numeric(self.df[target], errors="coerce")

        self.df["Id"] = range(1, len(self.df) + 1)
        relevant_columns = cat_features + num_features + [target] + ["Id"]
        self.df = self.df[relevant_columns]
        self.df["Id"] = self.df["Id"].astype("str")

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(pdf, drift: False, num_rows=10):
    """
    Generates synthetic data based on the distribution of the input Spark DataFrame.

    Parameters:
        df (pyspark.sql.dataframe.DataFrame): Input DataFrame with the desired schema.
        num_rows (int): Number of synthetic records to generate.

    Returns:
        pyspark.sql.dataframe.DataFrame: A Spark DataFrame containing synthetic data.
    """
    # Replace "NA" strings and drop any rows with missing values
    pdf = pdf.replace("NA", np.nan).dropna()

    synthetic_data = pd.DataFrame()

    # Iterate over columns and generate synthetic data based on column type/distribution
    for column in pdf.columns:
        # For the primary key, we will generate new synthetic IDs later
        if column == "Booking_ID":
            continue

        # Numeric columns
        if pd.api.types.is_numeric_dtype(pdf[column]):
            # For a year column, generate random integers within the observed range
            if column in {"arrival_year"}:
                synthetic_data[column] = np.random.randint(pdf[column].min(), pdf[column].max() + 1, num_rows)
            else:
                # For numeric columns (assumed integer if original is integer)
                if np.issubdtype(pdf[column].dtype, np.integer):
                    synthetic_data[column] = np.random.normal(pdf[column].mean(), pdf[column].std(), num_rows)
                    synthetic_data[column] = synthetic_data[column].round().astype(int)
                else:
                    synthetic_data[column] = np.random.normal(pdf[column].mean(), pdf[column].std(), num_rows)
                    synthetic_data[column] = synthetic_data[column].round(2)

        # Categorical or object columns
        elif pd.api.types.is_categorical_dtype(pdf[column]) or pd.api.types.is_object_dtype(pdf[column]):
            # Get normalized value counts to use as probabilities
            counts = pdf[column].value_counts(normalize=True)
            synthetic_data[column] = np.random.choice(counts.index, num_rows, p=counts.values)

        # Datetime columns (if any)
        elif pd.api.types.is_datetime64_any_dtype(pdf[column]):
            min_date, max_date = pdf[column].min(), pdf[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(np.random.randint(min_date.value, max_date.value, num_rows))
            else:
                synthetic_data[column] = [min_date] * num_rows

        # Fallback: random choice from the column values
        else:
            synthetic_data[column] = np.random.choice(pdf[column], num_rows)

    # Generate new synthetic Booking_ID values using a timestamp base
    timestamp_base = int(time.time() * 1000)
    synthetic_data["Booking_ID"] = [f"BKG{str(timestamp_base + i).zfill(5)}" for i in range(num_rows)]

    if drift:
        # Skew the top features to introduce drift
        top_features = [
            "no_of_weekend_nights",
            "no_of_previous_bookings_not_canceled",
            "no_of_week_nights",
        ]  # Select top 3 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 2

        # Set arrival_year to within the last 2 years
        current_year = pd.Timestamp.now().year
        if "arrival_year" in synthetic_data.columns:
            synthetic_data["arrival_year"] = np.random.randint(current_year - 2, current_year + 1, num_rows)

    # Reorder columns to match the desired schema
    columns_order = [
        "Booking_ID",
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "market_segment_type",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "booking_status",
    ]
    synthetic_data = synthetic_data[columns_order]

    # Retrieve the Spark session from the input DataFrame and convert back to a Spark DataFrame
    # spark = df.sql_ctx.sparkSession
    # synthetic_spark_df = spark.createDataFrame(synthetic_data)

    return synthetic_data
