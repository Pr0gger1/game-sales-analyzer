from utils.spark_utils import column_dict as cd
from utils.file_utils import save_plot_as_image

import seaborn as sns
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame as PDataFrame
from pyspark.sql import DataFrame


def game_sale_estimates(df: DataFrame, export_as_img: bool = True):
    df_clean = df.na.drop(
        subset=[
            cd["critic_score"],
            cd["na_sales"],
            cd["jp_sales"],
            cd["pal_sales"],
            cd["other_sales"],
            cd["total_sales"],
        ]
    )

    # Вычисление корреляции между оценками критиков и продажами в каждом регионе
    correlations = {}
    sales_columns = [
        cd["na_sales"],
        cd["jp_sales"],
        cd["pal_sales"],
        cd["other_sales"],
        cd["total_sales"],
    ]

    for column in sales_columns:
        correlation = df_clean.stat.corr(cd["critic_score"], column)
        correlations[column] = correlation

    # Вывод корреляций
    for column, correlation in correlations.items():
        print(f"Correlation between critic_score and {column}: {correlation}")

    # Преобразование в Pandas DataFrame
    pandas_df: PDataFrame = df_clean.toPandas()
    build_sales_distribution_by_critic_score_plot(pandas_df, export_as_img)
    build_heatmap_correlation_matrix(pandas_df, sales_columns, export_as_img)


def build_sales_distribution_by_critic_score_plot(
    df: PDataFrame, export_as_img: bool = True
):
    # Визуализация с использованием Seaborn
    sns.set(style="whitegrid")

    # Гистограмма для оценок критиков
    plt.figure(figsize=(10, 6))
    sns.histplot(df["critic_score"], bins=20, kde=True)
    plt.title("Distribution of Critic Scores")
    plt.xlabel("Critic Score")
    plt.ylabel("Frequency")
    if export_as_img:
        save_plot_as_image(plt.gcf(), "critic_scores_distribution.jpg")
    plt.show()


def build_heatmap_correlation_matrix(
    df: PDataFrame, sales_columns: list[str], export_as_img: bool = True
):
    plt.figure(figsize=(10, 6))
    corr_matrix = df[[cd["critic_score"]] + sales_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Game Sales Correlation Matrix")

    if export_as_img:
        save_plot_as_image(plt.gcf(), "game_sales_correlation.jpg")

    plt.show()
