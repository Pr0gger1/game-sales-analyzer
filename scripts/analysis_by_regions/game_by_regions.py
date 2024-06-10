from pyspark.sql import DataFrame
from pyspark.sql import functions as f

import matplotlib.pyplot as plt
from pandas import DataFrame as PDataFrame

from utils.spark_utils import column_dict as cd


def get_game_sales_by_regions(df: DataFrame, game_name: str) -> DataFrame:
    title: str = cd["title"]
    filtered_df: DataFrame = df.filter(f.lower(f.col(title)).contains(game_name.lower()))
    grouped_df: DataFrame = filtered_df.groupBy([title]).agg(
        f.sum(cd["na_sales"]).alias(cd["na_sales"]),
        f.sum(cd["jp_sales"]).alias(cd["jp_sales"]),
        f.sum(cd["pal_sales"]).alias(cd["pal_sales"]),
        f.sum(cd["other_sales"]).alias(cd["other_sales"])
    )
     
    grouped_df.show(truncate=False)
    visualize_game_sales_by_regions(grouped_df)
    
    return grouped_df
    
def visualize_game_sales_by_regions(df: DataFrame):
    pdf: PDataFrame = df.na.drop().toPandas()
    regions: list[str] = [col for col in df.columns if col.endswith("sales")]
    game_list: list[str] = pdf.iloc[:, 0].values
    plt.figure(figsize=(12, 8))
    plt.suptitle("Распределение продаж игры по регионам")
    
    for it in range(len(pdf)):
        sales = pdf.iloc[it, 1:].values.astype(float)
        plt.subplot(2, 2, it + 1)
        plt.bar(
            regions,
            sales,
            color=['blue', 'green', 'red', 'purple'],
            width=0.5
        )
        plt.xlabel("Регионы")
        plt.ylabel("Объем продаж (млн)")
        plt.title(f"{game_list[it]}")
        
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
        
    plt.show()
    
