from pyspark.sql import DataFrame, functions as f
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


def get_top_genres_by_critic_score_and_sales(df: DataFrame):
    genre_df = (
        df.groupBy("genre")
        .agg(
            f.median("critic_score").alias("critic_score"),
            f.sum("total_sales").alias("total_sales"),
        )
        .sort(["critic_score", "total_sales"], ascending=[False, False])
        .toPandas()
    )
    visualize_df(genre_df)



def visualize_df(df: DataFrame):
    palette = sns.color_palette("tab20", len(df))
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)

    ax = sns.barplot(x=df["genre"], y=df["critic_score"], palette=palette)
    ax.bar_label(
        ax.containers[0],
        label_type="edge",
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
    )
    plt.title("Genres with Highest Critic Scores")
    plt.xlabel("Genre")
    plt.ylabel("Median Critic Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


    plt.subplot(2, 1, 2)
    ax = sns.barplot(x=df["genre"], y=df["total_sales"], palette=palette)
    ax.bar_label(
        ax.containers[0],
        label_type="edge",
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
    )
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    plt.title("Genres with Highest Total Sales")
    plt.xlabel("Genre")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.show()