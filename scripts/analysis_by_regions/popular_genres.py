from pyspark.sql import DataFrame
from pyspark.sql import functions as f
import matplotlib.pyplot as plt

from utils.spark_utils import SparkUtils, column_dict


def get_popular_genres_by_region(df: DataFrame, region_column: str) -> DataFrame:
    grouped_by_genre_df = df.groupBy(column_dict['genre']).agg(
        f.sum(region_column).alias(region_column)
    ).orderBy(f.desc(region_column))

    SparkUtils.round_float_columns(grouped_by_genre_df, 3).show(50, False)
    visualize_popular_genres(grouped_by_genre_df, region_column)
    return grouped_by_genre_df


def visualize_popular_genres(df: DataFrame, region_column: str):
    pdf = df.toPandas()
    pdf.plot(kind='pie', y=region_column, labels=pdf[column_dict['genre']], legend=False)
    plt.title(f"Топ жанров по региону {region_column}")
    plt.show()
