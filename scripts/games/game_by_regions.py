import math
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

import matplotlib.pyplot as plt
from pandas import DataFrame as PDataFrame

from utils.file_utils import save_plot_as_image
from utils.spark_utils import column_dict as cd


def get_game_sales_by_regions(
    df: DataFrame, game_name: str, export_as_img: bool = True
) -> DataFrame:
    title: str = cd["title"]
    filtered_df: DataFrame = df.filter(
        f.lower(f.col(title)).contains(game_name.lower())
    )
    grouped_df: DataFrame = filtered_df.groupBy([title]).agg(
        f.sum(cd["na_sales"]).alias(cd["na_sales"]),
        f.sum(cd["jp_sales"]).alias(cd["jp_sales"]),
        f.sum(cd["pal_sales"]).alias(cd["pal_sales"]),
        f.sum(cd["other_sales"]).alias(cd["other_sales"]),
    )

    grouped_df.show(truncate=False)
    visualize_game_sales_by_regions(grouped_df, game_name, export_as_img)

    return grouped_df


def visualize_game_sales_by_regions(
    df: DataFrame, game_name: str, export_as_img: bool = True
):
    pdf: PDataFrame = df.na.drop().toPandas()
    regions: list[str] = [col for col in df.columns if col.endswith("sales")]
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
            
            if export_as_img:
                save_plot_as_image(plt.gcf(), f"{game_name}_{it + 1}_sales_by_regions.jpg")

        plt.subplots_adjust(
            left=0.1, 
            right=0.9, 
            top=0.9, 
            bottom=0.1,
            hspace=0.4, 
            wspace=0.4
        )
        
        plt.show()
