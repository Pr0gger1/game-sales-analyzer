from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from spark_utils import SparkUtils, column_dict


def get_popular_genres_by_region(df: DataFrame, region_column: str) -> DataFrame:
    grouped_by_genre_df = df.groupBy(column_dict['genre']).agg(
        f.sum(region_column).alias(region_column)
    ).orderBy(f.desc(region_column))

    SparkUtils.round_float_columns(grouped_by_genre_df, 3).show(50, False)
    return grouped_by_genre_df
