import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame, functions as f
from pandas import DataFrame as PDataFrame


def get_top_developers_by_critic_score(df: DataFrame):
    developer_stats = (
        df.toPandas()
        .groupby("developer")
        .agg({"critic_score": "mean", "total_sales": "sum"})
        .reset_index()
    )

    top_10_critic_score = developer_stats.sort_values(
        by=["critic_score", "total_sales"], kind="quicksort", ascending=[False, False]
    )

    filtered_stats = top_10_critic_score[top_10_critic_score["total_sales"] >= 10].head(10)
    visualize_df(filtered_stats)


def visualize_df(df: PDataFrame):
    plt.figure(figsize=(10, 6))
    sns.pointplot(x="critic_score", y="total_sales", hue="developer", data=df)
    plt.xticks(rotation=90)
    plt.title("Top developers by critic score")
    plt.legend(loc="best")
    plt.show()
