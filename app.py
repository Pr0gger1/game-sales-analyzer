from pyspark.sql import SparkSession, DataFrame

import constants
from scripts.analysis_by_regions import get_popular_games_by_region, \
    get_popular_genres_by_region, \
    get_popular_platforms_by_region, \
    get_game_sale_estimates, get_game_sales_by_regions

from scripts.predictions import predict_game_success
from utils.spark_utils import SparkUtils, column_dict as cd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import functions as f
from pyspark.ml.feature import VectorAssembler
import pandas as pd


def main():
    spark_session: SparkSession = SparkUtils.get_spark_session(app_name="app")
    df: DataFrame = spark_session.read.csv(
        path=constants.DATASET_FULLPATH,
        sep=",",
        header=True,
        schema=SparkUtils.DATASET_STRUCT
    ).drop("img")

    # get_popular_genres_by_region(df, 'other_sales')
    get_game_sales_by_regions(df, "The elder scrolls V")


if __name__ == "__main__":
    main()
