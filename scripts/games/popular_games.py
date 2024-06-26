from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from utils.Plot import Plot
from utils.spark_utils import SparkUtils
from datetime import date


def get_popular_games_by_region(
    df: DataFrame, 
    region_column: str,
    top_n: int = 10,
    start_date: date = None,
    end_date: date = None
) -> DataFrame:
    period_str = ""
    top_sales = (
        df.groupBy("title")
        .agg(
            f.sum(region_column).alias(region_column),
            f.first("release_date").alias("release_date")
            
        )
    )
    
    if start_date is not None and end_date is not None:
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        period_str = f"in period {start_date} - {end_date}"
        
        top_sales = top_sales.filter(
            (top_sales["release_date"] >= start_date_str) &
            (top_sales["release_date"] <= end_date_str)
        )

    top_sales = SparkUtils.round_float_columns(top_sales, 3).orderBy(
        region_column, ascending = False
    )

    top_sales.show(50)

    Plot.make_pie_plot(
        df=top_sales.toPandas().head(top_n),
        y_label=region_column,
        title=f"Top {top_n} games by region {region_column} {period_str}".strip(),
        legend_label_entity="title",
    )
    return top_sales
