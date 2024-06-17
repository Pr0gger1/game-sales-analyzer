from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from utils.Plot import Plot
from utils.spark_utils import SparkUtils, column_dict as cd


def get_popular_games_by_region(df: DataFrame, region_column: str) -> DataFrame:
    top_sales = df.groupBy(cd["title"]).agg(
        f.sum(region_column).alias(region_column),
        f.first(cd["genre"]).alias("genre"),
        f.first(cd["publisher"]).alias("publisher"),
        f.first(cd["developer"]).alias("developer"),
        f.first(cd["critic_score"]).alias("critic_score"),
        f.first(cd["release_date"]).alias("release_date"),
    ).orderBy(region_column, ascending=False)

    top_sales = SparkUtils.round_float_columns(top_sales, 3).orderBy(
        f.desc(region_column)
    )

    top_sales.show(50)
    
    Plot.make_pie_plot(
        df=top_sales.toPandas().head(20),
        y_label=region_column,
        title=f"Top games by region {region_column}",
        legend_entity="title"
    )
    return top_sales

