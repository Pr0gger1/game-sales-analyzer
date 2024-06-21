from pandas import DataFrame
import matplotlib.pyplot as plt


class Plot:
    @staticmethod
    def make_pie_plot(
        df: DataFrame, legend_label_entity: str, y_label: str, title: str
    ):
        df.plot(kind="pie", y=y_label, labels=None, autopct="%1.0f%%")
        plt.legend(
            labels=df[legend_label_entity], bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.title(title)
        plt.show()
