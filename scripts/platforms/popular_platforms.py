from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from utils.Plot import Plot
from datetime import date

from utils.spark_utils import SparkUtils


def get_popular_platforms_by_region(
    df: DataFrame,
    region_column: str,
    start_date: date = None,
    end_date: date = None
) -> DataFrame:
    filtered_df: DataFrame = df.na.drop()
    period_str = ""

    if start_date is not None and end_date is not None:
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        period_str = f"in period {start_date} - {end_date}"

        filtered_df = df.filter(
            (df["release_date"] >= start_date_str)
            & (df["release_date"] <= end_date_str)
        )

    grouped_by_genre_df = (
        filtered_df.groupBy("console")
        .agg(f.sum(region_column).alias(region_column))
        .orderBy(f.desc(region_column))
    )

    SparkUtils.round_float_columns(grouped_by_genre_df, 3).show(50, False)
    Plot.make_pie_plot(
        df=grouped_by_genre_df.toPandas(),
        legend_label_entity="console",
        title=f"Platforms by sales volume by region: {region_column} {period_str}".strip(),
        y_label=region_column,
    )

    return grouped_by_genre_df
