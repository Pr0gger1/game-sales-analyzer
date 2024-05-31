import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from pandas.core.frame import DataFrame as PDataFrame
from utils.spark_utils import SparkUtils, column_dict as cd


def get_popular_platforms_by_region(df: DataFrame, region_column: str) -> DataFrame:
    grouped_by_genre_df = df.groupBy(cd['console']).agg(
        f.sum(region_column).alias(region_column)
    ).orderBy(f.desc(region_column))

    SparkUtils.round_float_columns(grouped_by_genre_df, 3).show(50, False)
    visualize_popular_platforms_by_region(grouped_by_genre_df, region_column)
    return grouped_by_genre_df


def visualize_popular_platforms_by_region(df: DataFrame, region_column: str):
    pdf: PDataFrame = df.toPandas()
    pdf.plot(kind='pie', y=region_column, labels=pdf[cd['console']], legend=False)
    plt.title(f"Platforms by sales volume by region: {region_column}")
    plt.show()
