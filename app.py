from pyspark.sql import SparkSession, DataFrame

import constants
from scripts.analysis_by_regions import get_popular_games_by_region, \
    get_popular_genres_by_region, \
    get_popular_platforms_by_region, \
    get_game_sale_estimates

from utils.spark_utils import SparkUtils, column_dict


def main():
    spark_session: SparkSession = SparkUtils.get_spark_session(app_name="app")
    df: DataFrame = spark_session.read.csv(
        path=constants.DATASET_FULLPATH,
        sep=",",
        header=True,
        schema=SparkUtils.DATASET_STRUCT
    )
    # get_popular_genres_by_region(df, column_dict['na_sales'])
    # get_game_sale_estimates(df)
    # get_popular_games_by_region(df, 'na_sales')
    get_popular_platforms_by_region(df, 'na_sales')


if __name__ == "__main__":
    main()
