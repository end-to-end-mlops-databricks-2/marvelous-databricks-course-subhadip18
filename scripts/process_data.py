import pandas as pd
import argparse
import yaml
from loguru import logger
from Hotel_Reservation.config import ProjectConfig
from Hotel_Reservation.data_processor import DataProcessor, generate_synthetic_data

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

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

# Load configuration
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Initialize DataProcessor
filepath = "/Volumes/mlops_dev/subhadip/hotel_reservation_data/Hotel_Reservations.csv"
# Load the data
pandas_df = pd.read_csv(filepath)


# Generate synthetic data
### This is mimicking a new data arrival. In real world, this would be a new batch of data.
# df is passed to infer schema
synthetic_df = generate_synthetic_data(pandas_df, num_rows=100)
logger.info("Synthetic data generated")

if "spark" not in locals():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

# Initialize DataProcessor
data_processor = DataProcessor(synthetic_df, config, spark)

# Preprocess the data
logger.info("Preprocessing data")
data_processor.preprocess()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()

logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


    
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
