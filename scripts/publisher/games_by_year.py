from pyspark.sql import DataFrame, functions as f
from pandas import DataFrame as PDataFrame
import matplotlib.pyplot as plt
import seaborn as sns


def count_of_games_by_year_from_publisher(
    df: DataFrame, 
    start_year: int = None,
    end_year: int = None,
    top_n: int = 10
    ) -> DataFrame:
    transformed_df = df.withColumn("year", f.year("release_date"))
    
    if (start_year is not None and end_year is not None):
        transformed_df = transformed_df.filter(f.col("year").between(start_year, end_year))
    
    grouped_df = (
        transformed_df.groupBy("publisher", "year")
        .agg(f.count(f.col("title")).alias("released_title_count"))
        .orderBy("year", ascending=True)
    )

    grouped_df.show(50, False)
    
    visualized_df(grouped_df.toPandas(), top_n)
    return grouped_df


def visualized_df(df: PDataFrame, top_n: int = 10):
    top_publishers = df.groupby('publisher')['released_title_count'].sum().nlargest(top_n).index
    pdf_top = df[df['publisher'].isin(top_publishers)]
    

    plt.figure(figsize=(14, 8))
    ax = sns.histplot(
        data=pdf_top, 
        x='publisher', 
        y='year', 
        weights='released_title_count', 
        multiple='stack', 
        discrete=(True, True), 
        shrink=.8
    )
    ax.set_title(f'Top {top_n} Publishers by Number of Games Released per Year')
    plt.xticks(rotation=45)
    plt.show()