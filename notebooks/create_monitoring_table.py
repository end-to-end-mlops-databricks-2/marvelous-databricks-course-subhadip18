# Databricks notebook source
# MAGIC
# %pip install /Workspace/Users/subhadip18@gmail.com/.bundle/dev/marvelous-databricks-course-subhadip18/artifacts/.internal/hotel_reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------
import datetime
import itertools
import logging
import time

import mlflow
import pandas as pd
import requests
import yaml
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from pyspark.dbutils import DBUtils
from pyspark.sql import functions as F
from pyspark.sql.functions import col, current_timestamp, to_utc_timestamp
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

from Hotel_Reservation.config import ProjectConfig
from Hotel_Reservation.data_processor import DataProcessor, generate_synthetic_data
from Hotel_Reservation.monitoring import create_or_refresh_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="prd")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

if "spark" not in locals():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
# COMMAND ----------

model_uri = "models:/mlops_dev.subhadip.hotel_reservation_model_basic@latest-model"
pipeline = mlflow.sklearn.load_model(model_uri)

# Assuming the model with feature_importances_ is the last step in the pipeline
model = pipeline.steps[-1][1]

# Extract feature importances
feature_importance = model.feature_importances_

# Assuming you have a list of feature names
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)
# display(importance_df.head(10))

# COMMAND ----------


train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

inference_data_skewed = generate_synthetic_data(train_set, drift=True, num_rows=200)

data_processor = DataProcessor(inference_data_skewed, config, spark)
data_processor.preprocess()

inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)
# display(inference_data_skewed_spark)

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------


# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="prd")

test_set = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    .withColumn("Id", col("Id").cast("string"))
    .toPandas()
)


inference_data_skewed = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")
    .withColumn("Id", col("Id").cast("string"))
    .toPandas()
)

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------


workspace = WorkspaceClient()

required_columns = [
    "Id",
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

sampled_skewed_records = inference_data_skewed[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")

# COMMAND ----------


def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/hotel_reservation_model-serving/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="hotel_reservation_model-serving", dataframe_records=[dataframe_record]
    )
    return response


# COMMAND ----------

end_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    print(record)
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------

end_time = datetime.datetime.now() + datetime.timedelta(minutes=5)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------


inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`hotel_reservation_payload`")
# display(inf_table)

# Define the schema
request_schema = StructType(
    [
        StructField(
            "dataframe_records",
            ArrayType(
                StructType(
                    [
                        StructField("Id", StringType(), True),
                        StructField("no_of_adults", IntegerType(), True),
                        StructField("no_of_children", IntegerType(), True),
                        StructField("no_of_weekend_nights", IntegerType(), True),
                        StructField("no_of_week_nights", IntegerType(), True),
                        StructField("required_car_parking_space", IntegerType(), True),
                        StructField("lead_time", IntegerType(), True),
                        StructField("arrival_year", IntegerType(), True),
                        StructField("arrival_month", IntegerType(), True),
                        StructField("arrival_date", IntegerType(), True),
                        StructField("repeated_guest", IntegerType(), True),
                        StructField("no_of_previous_cancellations", IntegerType(), True),
                        StructField("no_of_previous_bookings_not_canceled", IntegerType(), True),
                        StructField("avg_price_per_room", DoubleType(), True),
                        StructField("no_of_special_requests", IntegerType(), True),
                        StructField("type_of_meal_plan", StringType(), True),
                        StructField("room_type_reserved", StringType(), True),
                        StructField("market_segment_type", StringType(), True),
                    ]
                )
            ),
            True,
        )
    ]
)


response_schema = StructType([StructField("predictions", ArrayType(IntegerType()), True)])

inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))
# inf_table_parsed = inf_table_parsed.withColumn("parsed_response",
#                                             F.col("parsed_response.predictions"))


df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

df_final = df_exploded.select(
    F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
    "timestamp_ms",
    "databricks_request_id",
    "execution_time_ms",
    F.col("record.Id").alias("Id"),
    F.col("record.no_of_adults").alias("no_of_adults"),
    F.col("record.no_of_children").alias("no_of_children"),
    F.col("record.no_of_weekend_nights").alias("no_of_weekend_nights"),
    F.col("record.no_of_week_nights").alias("no_of_week_nights"),
    F.col("record.required_car_parking_space").alias("required_car_parking_space"),
    F.col("record.lead_time").alias("lead_time"),
    F.col("record.arrival_year").alias("arrival_year"),
    F.col("record.arrival_month").alias("arrival_month"),
    F.col("record.arrival_date").alias("arrival_date"),
    F.col("record.repeated_guest").alias("repeated_guest"),
    F.col("record.no_of_previous_cancellations").alias("no_of_previous_cancellations"),
    F.col("record.no_of_previous_bookings_not_canceled").alias("no_of_previous_bookings_not_canceled"),
    F.col("record.avg_price_per_room").alias("avg_price_per_room"),
    F.col("record.no_of_special_requests").alias("no_of_special_requests"),
    F.col("record.type_of_meal_plan").alias("type_of_meal_plan"),
    F.col("record.room_type_reserved").alias("room_type_reserved"),
    F.col("record.market_segment_type").alias("market_segment_type"),
    F.col("parsed_response.predictions")[0].alias("prediction"),
    F.lit("hotel_reservation_model_basic").alias("model_name"),
)
# display(df_final)
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

df_final_with_status = (
    df_final.join(test_set.select("Id", "booking_status"), on="Id", how="left")
    .withColumnRenamed("booking_status", "booking_status_test")
    .join(inference_set_skewed.select("Id", "booking_status"), on="Id", how="left")
    .withColumnRenamed("booking_status", "booking_status_inference")
    .select("*", F.coalesce(F.col("booking_status_test"), F.col("booking_status_inference")).alias("booking_status"))
    .drop("booking_status_test", "booking_status_inference")
    .withColumn("booking_status", F.col("booking_status").cast("double"))
    .withColumn("prediction", F.col("prediction").cast("double"))
    .dropna(subset=["booking_status", "prediction"])
)

# display(df_final_with_status)
df_final_with_status.write.format("delta").mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.model_monitoring"
)


# COMMAND ----------


spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml", env="prd")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)

# COMMAND ----------

# # from databricks.sdk import WorkspaceClient
# # workspace = WorkspaceClient()
# from databricks.sdk.service.catalog import (
#     MonitorInferenceLog,
#     MonitorInferenceLogProblemType,
# )

# monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

# workspace.quality_monitors.create(
#     table_name=monitoring_table,
#     assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
#     output_schema_name=f"{config.catalog_name}.{config.schema_name}",
#     inference_log=MonitorInferenceLog(
#         problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
#         prediction_col="prediction",
#         timestamp_col="timestamp",
#         granularities=["30 minutes"],
#         model_id_col="model_name",
#         label_col="booking_status",
#     ),
# )

# COMMAND ----------

# workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
# workspace.quality_monitors.run_refresh(
#         table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
#     )
# logger.info("Lakehouse monitoring table exist, refreshing.")
