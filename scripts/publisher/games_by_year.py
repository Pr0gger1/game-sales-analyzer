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
    
    visualized_df(
        grouped_df.toPandas(),
        top_n,
        start_year,
        end_year
    )
    return grouped_df


def visualized_df(
    df: PDataFrame,
    top_n: int = 10,
    start_year: int = None,
    end_year: int = None
):
    top_publishers = df.groupby('publisher')['released_title_count'].sum().nlargest(top_n).index
    pdf_top = df[df['publisher'].isin(top_publishers)]

    # Создание сводной таблицы
    pivot_table = pdf_top.pivot_table(index='year', columns='publisher', values='released_title_count', aggfunc='sum', fill_value=0)
    pivot_table = pivot_table.sort_values(by="year", ascending=False)

    # Построение тепловой карты
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    
    title = f'Top {top_n} Publishers by Number of Games Released per Year'
    
    if start_year is not None and end_year is not None:
        title += f" ({start_year}-{end_year})"

    # Установка заголовка и подписи
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.show()