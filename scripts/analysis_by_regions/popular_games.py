from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

from spark_utils import SparkUtils, column_dict


def get_popular_games_by_region(df: DataFrame, region_column: str) -> DataFrame:
    top_sales = df.groupBy(column_dict['title']).agg(
        f.sum(region_column).alias(region_column),
        f.first(column_dict['genre']).alias('genre'),
        f.first(column_dict['publisher']).alias('publisher'),
        f.first(column_dict['developer']).alias('developer'),
        f.first(column_dict['critic_score']).alias('critic_score'),
        f.first(column_dict['release_date']).alias('release_date')
    )

    top_sales = SparkUtils.round_float_columns(top_sales, 3) \
        .orderBy(f.desc(region_column))

    top_sales.show(50, truncate=False)
    return top_sales


def cluster_analysis(df: DataFrame):
    assembler = VectorAssembler(
        inputCols=[column_dict['na_sales'], column_dict['jp_sales'], column_dict['pal_sales'],
                   column_dict['other_sales']],
        outputCol='features'
    )

    cleaned_data = df.na.fill(0.0)
    features_df = assembler.transform(cleaned_data)
    kmeans = KMeans().setK(2).setSeed(1)
    model = kmeans.fit(features_df)
    predictions = model.transform(features_df)

    aggregated_sales = predictions.groupBy(column_dict['title']).agg(
        f.sum(column_dict['na_sales']).alias('total_na_sales'),
        f.sum(column_dict['jp_sales']).alias('total_jp_sales'),
        f.sum(column_dict['pal_sales']).alias('total_pal_sales'),
        f.sum(column_dict['other_sales']).alias('total_other_sales'),
        (f.sum(column_dict['na_sales']) + f.sum(column_dict['jp_sales']) +
         f.sum(column_dict['pal_sales']) + f.sum(column_dict['other_sales'])).alias('total_sales'),
        f.first('prediction')
    )

    SparkUtils.round_float_columns(aggregated_sales.orderBy(f.desc("total_sales"))).show(100, False)
