# Databricks notebook source

# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_dev/subhadip/hotel_reservation_data/Hotel_Reservation-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------
# dbutils.library.restartPython()
# COMMAND ----------
import mlflow
from pyspark.sql import SparkSession

from Hotel_Reservation.config import ProjectConfig, Tags
from Hotel_Reservation.models.basic_model import BasicModel

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "feature_subh_data_process"})
# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()
# COMMAND ----------
basic_model.train()
basic_model.log_model()

run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel_reservation-basic"], filter_string="tags.branch='feature_subh_data_process'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)
