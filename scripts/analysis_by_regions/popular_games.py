from matplotlib import pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

from utils.spark_utils import SparkUtils, column_dict as cd


def get_popular_games_by_region(df: DataFrame, region_column: str) -> DataFrame:
    top_sales = df.groupBy(cd['title']).agg(
        f.sum(region_column).alias(region_column),
        f.first(cd['genre']).alias('genre'),
        f.first(cd['publisher']).alias('publisher'),
        f.first(cd['developer']).alias('developer'),
        f.first(cd['critic_score']).alias('critic_score'),
        f.first(cd['release_date']).alias('release_date')
    )

    top_sales = SparkUtils.round_float_columns(top_sales, 3) \
        .orderBy(f.desc(region_column))

    top_sales.show(50, truncate=False)
    visualize_popular_games(top_sales, region_column)
    return top_sales


def visualize_popular_games(df: DataFrame, region_column: str):
    pdf = df.toPandas()
    pdf.plot(kind='pie', y=region_column, labels=pdf[cd['title']], legend=False)
    plt.title(f"Game Top by region {region_column}")
    plt.show()

# def cluster_analysis(df: DataFrame):
#     assembler = VectorAssembler(
#         inputCols=[
#             cd['na_sales'],
#             cd['jp_sales'],
#             cd['pal_sales'],
#             cd['other_sales']
#         ],
#         outputCol='features'
#     )
#
#     cleaned_data = df.na.fill(0.0)
#     features_df = assembler.transform(cleaned_data)
#     kmeans = KMeans().setK(2).setSeed(1)
#     model = kmeans.fit(features_df)
#     predictions = model.transform(features_df)
#
#     aggregated_sales = predictions.groupBy(cd['title']).agg(
#         f.sum(cd['na_sales']).alias('total_na_sales'),
#         f.sum(cd['jp_sales']).alias('total_jp_sales'),
#         f.sum(cd['pal_sales']).alias('total_pal_sales'),
#         f.sum(cd['other_sales']).alias('total_other_sales'),
#         (f.sum(cd['na_sales']) + f.sum(cd['jp_sales']) +
#          f.sum(cd['pal_sales']) + f.sum(cd['other_sales'])).alias('total_sales'),
#         f.first('prediction')
#     )
#
#     SparkUtils.round_float_columns(aggregated_sales.orderBy(f.desc("total_sales"))).show(100, False)
