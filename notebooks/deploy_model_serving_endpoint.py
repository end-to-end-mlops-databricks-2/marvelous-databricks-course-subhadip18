# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/subhadip/hotel_reservation_data/Hotel_Reservation-0.0.1-py3-none-any.whl

# COMMAND ----------
# dbutils.library.restartPython()

# COMMAND ----------
import os
import time
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from Hotel_Reservation.config import ProjectConfig
from Hotel_Reservation.serving.model_serving import ModelServing

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.hotel_reservation_model_basic", endpoint_name="hotel_reservation_model-serving"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------
# Create a sample request body
required_columns = [
"no_of_adults",
"no_of_children",
"no_of_weekend_nights",
"no_of_week_nights",
"required_car_parking_space",
"lead_time",
"arrival_year",
"arrival_month",
"arrival_date",
"repeated_guest",
"no_of_previous_cancellations",
"no_of_previous_bookings_not_canceled",
"avg_price_per_room",
"no_of_special_requests",
"type_of_meal_plan",
"room_type_reserved",
"market_segment_type",
]

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'no_of_adults': 2,
  'no_of_children': 1,
  'no_of_weekend_nights': 2,
  'no_of_week_nights': 1,
  'required_car_parking_space': 1}]
"""

def call_endpoint(endpoint_name: str,record: List[Dict]):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

status_code, response_text = call_endpoint('hotel_reservation_model-serving',dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# "load test"

for i in range(len(dataframe_records)):
    call_endpoint('hotel_reservation_model-serving',dataframe_records[i])
    time.sleep(0.2)