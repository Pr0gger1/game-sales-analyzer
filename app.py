from pyspark.sql import SparkSession, DataFrame

import constants
from scripts import games, developers, genres, developers, predictions, platforms, publisher

from utils.spark_utils import SparkUtils

def main():
    spark_session: SparkSession = SparkUtils.get_spark_session(app_name="app")
    df: DataFrame = spark_session.read.csv(
        path=constants.DATASET_FULLPATH,
        sep=",",
        header=True,
        schema=SparkUtils.DATASET_STRUCT,
    ).drop("img")

    # games.get_popular_games_by_region(df, "jp_sales")
    genres.get_top_genres_by_critic_score_and_sales(df)
    games.game_sale_estimates(df)
    # predict_game_success(df, spark_session, "Fifa", )
    # genres.get_popular_genres_by_region(df, "pal_sales")
    # platforms.get_popular_platforms_by_region(df, "jp_sales", date(2015, 1, 1), date(2024, 1, 1))
    # games.get_game_sales_by_regions(df, "Resident Evil")
    # predictions.predict_next_game_performance(spark_session, df, "Call of Duty")
    # get_successful_game_profile(df)

    # developers.get_top_developers_by_critic_score(df)
    # publisher.get_top_sales_performance_by_publisher(df)


if __name__ == "__main__":
    main()
