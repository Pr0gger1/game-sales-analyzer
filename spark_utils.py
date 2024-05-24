import os
from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType

from constants import DATASET_FOLDER

class SparkUtils:
    DATASET_STRUCT = StructType([
    StructField("img", StringType(), True),
    StructField("title", StringType(), True),
    StructField("console", StringType(), True),
    StructField("genre", StringType(), True),
    StructField("publisher", StringType(), True),
    StructField("developer", StringType(), True),
    StructField("critic_score", FloatType(), True),
    StructField("total_sales", FloatType(), True),
    StructField("na_sales", FloatType(), True),
    StructField("jp_sales", FloatType(), True),
    StructField("pal_sales", FloatType(), True),
    StructField("other_sales", FloatType(), True),
    StructField("release_date", DateType(), True),
    StructField("last_update", DateType(), True)
])
    
    @staticmethod
    def get_spark_session(app_name: str, master: str = "local[*]") -> SparkSession:
        return SparkSession.builder \
            .master(master) \
            .appName(app_name) \
            .getOrCreate()
            
    @staticmethod
    def load_from_csv(
        spark: SparkSession,
        filename: str, 
        sep: str = ",",
        header: bool = True,
        schema: StructType = None
    ) -> DataFrame:
        full_path: str = os.path.join(DATASET_FOLDER, filename)
        return spark.read.csv(path=full_path, sep=sep, header=header, schema=schema)
