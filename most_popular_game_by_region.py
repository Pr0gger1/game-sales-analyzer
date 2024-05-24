from pyspark.sql import SparkSession, DataFrame

from spark_utils import SparkUtils
from pyspark.sql.functions import max, sum, round, asc, desc

def main(spark: SparkSession):
    filename: str = "vgchartz-2024.csv"
    df: DataFrame = SparkUtils.load_from_csv(
        spark=spark,
        filename=filename,
        schema=SparkUtils.DATASET_STRUCT
    ).drop('img')
    
    popular_games_by_region(df, 'na_sales')
    popular_games_by_region(df, 'jp_sales')
    popular_games_by_region(df, 'pal_sales')
    

    
def popular_games_by_region(df: DataFrame, region_column: str):
    total_region_sales: str = f"sum_{region_column}"
    
    top_sales: DataFrame = df.groupby('title') \
        .agg(sum(region_column).alias(total_region_sales))
        
    df_data = df.select(['title', 'console', 'genre', 'publisher', 'developer', 'critic_score', 'release_date', region_column])
    
    top_sales = top_sales.join(df_data, (df_data.title == top_sales.title) & (df_data[region_column] == top_sales[total_region_sales]))
    top_sales = top_sales.withColumn(total_region_sales, round(top_sales[total_region_sales], 3)).orderBy(desc(total_region_sales))
    top_sales.show(truncate=False)


if __name__ == '__main__':
    spark = SparkUtils.get_spark_session(app_name="most popular game by region")
    main(spark)