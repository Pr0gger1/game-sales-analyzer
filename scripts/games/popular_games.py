from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from utils.Plot import Plot
from utils.spark_utils import SparkUtils


def get_popular_games_by_region(df: DataFrame, region_column: str) -> DataFrame:
    top_sales = (
        df.groupBy("title")
        .agg(f.sum(region_column).alias(region_column))
        .orderBy(region_column, ascending=False)
    )

    top_sales = SparkUtils.round_float_columns(top_sales, 3).orderBy(
        f.desc(region_column)
    )

    top_sales.show(50)

    Plot.make_pie_plot(
        df=top_sales.toPandas().head(20),
        y_label=region_column,
        title=f"Top games by region {region_column}",
        legend_label_entity="title",
    )
    return top_sales
