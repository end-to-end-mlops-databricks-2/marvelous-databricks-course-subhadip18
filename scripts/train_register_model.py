import mlflow
from pyspark.sql import SparkSession
from loguru import logger
import argparse
from pyspark.dbutils import DBUtils
from Hotel_Reservation.config import ProjectConfig, Tags
from Hotel_Reservation.models.basic_model import BasicModel



# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")
basic_model.load_data()
logger.info("Loading Data completed")
basic_model.prepare_features()
logger.info("Prepared Feature")
basic_model.train()
logger.info("Model Training Completed")
basic_model.log_model()
logger.info("Model Logging Completed")

run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel_reservation-basic"], filter_string="tags.branch='feature_subh_data_process'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()


# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()



test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)

model_improved = basic_model.model_improved(test_set=test_set.toPandas())
logger.info("Model evaluation completed, model improved: ", model_improved) 

if model_improved:
    # Register the model
    latest_version = basic_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)


