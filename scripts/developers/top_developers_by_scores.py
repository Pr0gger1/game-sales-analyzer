import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame, functions as f
from pandas import DataFrame as PDataFrame


def get_top_developers_by_critic_score(df: DataFrame):    
    developer_stats = df.groupBy("developer").agg(
        f.sum("total_sales").alias("total_sales"),
        f.mean("critic_score").alias("critic_score")
    ).orderBy("total_sales", ascending=False)

    top_developers_by_critic_score = developer_stats.toPandas()
    
    print(top_developers_by_critic_score.head(20))

    filtered_stats = top_developers_by_critic_score.head(30)
    visualize_df(filtered_stats)


def visualize_df(df: PDataFrame):
    palette = sns.color_palette("terrain_r", n_colors=df['developer'].nunique())
     
    plt.figure(figsize=(10, 8))
    sns.pointplot(
        x="critic_score",
        y="total_sales",
        hue="developer",
        data=df,
        palette=palette
    )
    plt.xticks(rotation=90)
    plt.title("Top developers by critic score")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.show()