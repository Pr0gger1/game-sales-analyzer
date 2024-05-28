import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType
from pyspark.sql import functions as f

from constants import DATASET_FOLDER

column_dict = {
    "img": "img",
    "title": "title",
    "console": "console",
    "genre": "genre",
    "publisher": "publisher",
    "developer": "developer",
    "critic_score": "critic_score",
    "total_sales": "total_sales",
    "na_sales": "na_sales",
    "jp_sales": "jp_sales",
    "pal_sales": "pal_sales",
    "other_sales": "other_sales",
    "release_date": "release_date",
    "last_update": "last_update"
}


class SparkUtils:

    DATASET_STRUCT = StructType([
        StructField(column_dict["img"], StringType(), True),
        StructField(column_dict["title"], StringType(), True),
        StructField(column_dict["console"], StringType(), True),
        StructField(column_dict["genre"], StringType(), True),
        StructField(column_dict["publisher"], StringType(), True),
        StructField(column_dict["developer"], StringType(), True),
        StructField(column_dict["critic_score"], FloatType(), True),
        StructField(column_dict["total_sales"], FloatType(), True),
        StructField(column_dict["na_sales"], FloatType(), True),
        StructField(column_dict["jp_sales"], FloatType(), True),
        StructField(column_dict["pal_sales"], FloatType(), True),
        StructField(column_dict["other_sales"], FloatType(), True),
        StructField(column_dict["release_date"], DateType(), True),
        StructField(column_dict["last_update"], DateType(), True)
    ])

    @staticmethod
    def get_spark_session(app_name: str, master: str = "local[*]") -> SparkSession:
        return SparkSession.builder \
            .master(master) \
            .appName(app_name) \
            .getOrCreate()

    @staticmethod
    def get_dataset(filename: str) -> str:
        return os.path.join(DATASET_FOLDER, filename)

    @staticmethod
    def round_float_columns(df: DataFrame, n: int = 2) -> DataFrame:
        for c_name, c_type in df.dtypes:
            if c_type in ('double', 'float'):
                df = df.withColumn(c_name, f.round(c_name, n))

        return df
