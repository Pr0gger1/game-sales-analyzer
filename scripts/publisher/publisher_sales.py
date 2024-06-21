from pyspark.sql import DataFrame, functions as f
from pandas import DataFrame as PDataFrame
import matplotlib.pyplot as plt
import seaborn as sns

def get_top_sales_performance_by_publisher(df: DataFrame) -> DataFrame:
    grouped_df = df.groupBy("publisher").agg(
        f.sum("total_sales").alias("total_sales"),
        f.count("title").alias("title")
    ).orderBy("total_sales", ascending=False)
    
    pdf = grouped_df.toPandas()
    sorted_publishers_by_sales = pdf.sort_values(by="total_sales", ascending=False).head(10)
    sorted_publishers_by_num = pdf.sort_values(by="title", ascending=False).head(10)
    
    grouped_df.show()
    
    visualize_df(
        sorted_publishers_by_num,
        sorted_publishers_by_sales
    )
    
    return grouped_df
    
    
def visualize_df(
    publishers_by_num: PDataFrame,
    publishers_by_sales: PDataFrame
):
    palette = sns.color_palette("tab20", len(publishers_by_num))

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="title", 
        y="publisher",
        data=publishers_by_num, 
        palette=palette,
        hue="publisher",
        legend=False
    )
    
    ax.bar_label(
        ax.containers[0],
        label_type="edge",
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.5)
    )
    
    plt.title("Number of Releases by Publisher")
    plt.xlabel("Number of Releases")
    plt.ylabel("Publisher")
    plt.tight_layout()
    plt.show()
    
    palette = sns.color_palette("tab20", len(publishers_by_sales))
    
    
    plt.figure(figsize=(12, 6))
    ax=sns.barplot(
        x="total_sales",
        y="publisher",
        data=publishers_by_sales,
        palette=palette,
        legend=False,
        hue="publisher"
    )
    
    ax.bar_label(
        ax.containers[0],
        label_type="edge",
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.5)
    )
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    plt.title("Total Sales Performance by Publisher")
    plt.xlabel("Total Sales")
    plt.ylabel("Publisher")
    plt.tight_layout()
    plt.show()