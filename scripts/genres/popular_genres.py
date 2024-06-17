from pyspark.sql import DataFrame
from pyspark.sql import functions as f
import matplotlib.pyplot as plt
from datetime import date

from utils.Plot import Plot
from utils.spark_utils import SparkUtils, column_dict


def get_popular_genres_by_region(
    df: DataFrame,
    region_column: str, 
    start_date: date = None,
    end_date: date = None
) -> DataFrame:
    filtered_df: DataFrame = df

    if start_date is not None and end_date is not None:
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        filtered_df = df.filter(
            (df["release_date"] >= start_date_str)
            & (df["release_date"] <= end_date_str)
        )

    grouped_by_genre_df = (
        filtered_df.groupBy(column_dict["genre"])
        .agg(f.sum(region_column).alias(region_column))
        .orderBy(f.desc(region_column))
    )

    SparkUtils.round_float_columns(grouped_by_genre_df, 3).show(50, False)

    Plot.make_pie_plot(
        df=grouped_by_genre_df.toPandas(),
        legend_entity="genre",
        title=f"Top genres by region {region_column}",
        y_label=region_column,
    )
    return grouped_by_genre_df
