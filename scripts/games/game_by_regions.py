import math
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

import matplotlib.pyplot as plt
from pandas import DataFrame as PDataFrame


def get_game_sales_by_region(
    df: DataFrame,
    game_name: str
) -> DataFrame:
    df = df.filter(
        f.col("na_sales").isNotNull() &
        f.col("jp_sales").isNotNull() & 
        f.col("pal_sales").isNotNull() & 
        f.col("other_sales").isNotNull()
    )
    
    title: str = "title"
    filtered_df: DataFrame = df.filter(
        f.lower(f.col(title)).contains(game_name.lower())
    )
    grouped_df: DataFrame = filtered_df.groupBy([title]).agg(
        f.sum("na_sales").alias("na_sales"),
        f.sum("jp_sales").alias("jp_sales"),
        f.sum("pal_sales").alias("pal_sales"),
        f.sum("other_sales").alias("other_sales"),
    )

    grouped_df.show(truncate=False)
    visualize_df(grouped_df.toPandas())

    return grouped_df


def visualize_df(pdf: PDataFrame):
    regions: list[str] = [col for col in pdf.columns if col.endswith("sales")]
    game_list: list[str] = pdf.iloc[:, 0].values

    num_games = len(game_list)
    cols = 2
    rows = 2
    games_per_figure = rows * cols

    for fig_num in range(math.ceil(num_games / games_per_figure)):
        plt.figure(figsize=(12, 8))
        plt.suptitle("Game sales distribution by regions")

        for it in range(games_per_figure):
            index = fig_num * games_per_figure + it
            if index >= num_games:
                break
            sales = pdf.iloc[index, 1:].values.astype(float)
            plt.subplot(rows, cols, it + 1)
            plt.bar(regions, sales, color=["blue", "green", "red", "purple"], width=0.5)
            plt.xlabel("Regions")
            plt.ylabel("Sales volume (million)")
            plt.title(f"{game_list[index]}")

        plt.subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4
        )

        plt.show()
