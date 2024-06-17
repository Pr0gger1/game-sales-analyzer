# import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
# from pyspark.ml.clustering import KMeans
# from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
# from pyspark.sql import DataFrame


# def get_game_profile(df: DataFrame):
    # df = df.na.drop(subset=["critic_score", "na_sales", "jp_sales", "pal_sales", "other_sales"])

    # # Индексирование строковой колонки 'developer'
    # # indexer = StringIndexer(inputCol="developer", outputCol="developer_index", handleInvalid="keep")
    # # df = indexer.fit(df).transform(df)

    # feature_columns = ["critic_score", "na_sales", "jp_sales", "pal_sales", "other_sales"]

    # # Преобразование данных с использованием VectorAssembler
    # assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    # data = assembler.transform(df.select(*feature_columns))

    # # Кластерный анализ
    # kmeans = KMeans(k=3, seed=1)
    # model = kmeans.fit(data)

    # # Визуализация профилей успешных игр
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)

    # # Преобразование центров кластеров в Pandas DataFrame для визуализации
    # centers_df = pd.DataFrame(centers, columns=feature_columns)

    # # Визуализация кластеров
    # plt.figure(figsize=(14, 7))
    # sns.heatmap(centers_df, annot=True, cmap="YlGnBu")
    # plt.xlabel('Features')
    # plt.ylabel('Clusters')
    # plt.title('Cluster Centers')
    # plt.show()
    
    # print(centers_df)

    # Преобразование индексированных значений обратно в строки для итогового DataFrame
    # converter = IndexToString(inputCol="developer_index", outputCol="developer_str", labels=indexer.labels)
    # result_df = converter.transform(df.select("developer_index"))

    # Показать преобразованные данные
    # result_df.select("developer_index", "developer_str").show()










import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

def get_successful_game_profile(df: DataFrame, score_threshold: float = 80, sales_threshold: float = 1.0, num_clusters: int = 3) -> DataFrame:
    df = df.na.drop()
    indexer_genre = StringIndexer(inputCol="genre", outputCol="genre_index", handleInvalid="keep")
    indexer_developer = StringIndexer(inputCol="developer", outputCol="developer_index", handleInvalid="keep")
    indexer_publisher = StringIndexer(inputCol="publisher", outputCol="publisher_index", handleInvalid="keep")
    indexer_platform = StringIndexer(inputCol="console", outputCol="platform_index", handleInvalid="keep")

    df = indexer_genre.fit(df).transform(df)
    df = indexer_developer.fit(df).transform(df)
    df = indexer_publisher.fit(df).transform(df)
    df = indexer_platform.fit(df).transform(df)

    # Колонки признаков, включая преобразованные колонки
    feature_columns = ["genre_index", "developer_index", "publisher_index", "platform_index", "critic_score", "total_sales"]

    # Преобразование данных с использованием VectorAssembler
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data = assembler.transform(df.select(*feature_columns))

    # Кластерный анализ
    kmeans = KMeans(k=3, seed=1)
    model = kmeans.fit(data)

    # Получение центров кластеров
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    # Преобразование центров кластеров в Pandas DataFrame для визуализации
    centers_df = pd.DataFrame(centers, columns=feature_columns)

    # Визуализация кластеров через Radar Chart
    labels = feature_columns
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i in range(len(centers)):
        values = centers[i].tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, values, label=f'Cluster {i+1}')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title('Radar Chart of Cluster Centers')
    plt.show()

    # Визуализация кластеров через Box Plots
    centers_df_melted = centers_df.melt(var_name='Feature', value_name='Value')
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Feature', y='Value', data=centers_df_melted)
    plt.title('Box Plot of Cluster Centers')
    plt.show()
    # Определение успешных игр по критериям оценки и продаж
    # successful_games = df.filter(
    #     (f.col('critic_score') >= score_threshold) &
    #     (f.col('total_sales') >= sales_threshold)
    # )

    # # Индексирование категориальных признаков
    # genre_indexer = StringIndexer(inputCol='genre', outputCol='genre_index')
    # publisher_indexer = StringIndexer(inputCol='publisher', outputCol='publisher_index')
    # developer_indexer = StringIndexer(inputCol='developer', outputCol='developer_index')
    
    # # Кодирование категориальных признаков
    # genre_encoder = OneHotEncoder(inputCol='genre_index', outputCol='genre_vec')
    # publisher_encoder = OneHotEncoder(inputCol='publisher_index', outputCol='publisher_vec')
    # developer_encoder = OneHotEncoder(inputCol='developer_index', outputCol='developer_vec')

    # # Создание вектора характеристик для кластерного анализа
    # assembler = VectorAssembler(
    #     inputCols=['critic_score', 'total_sales', 'na_sales', 'jp_sales', 'genre_vec', 'publisher_vec', 'developer_vec'],
    #     outputCol='features'
    # )
    
    # # Создание и выполнение конвейера
    # pipeline = Pipeline(stages=[genre_indexer, publisher_indexer, developer_indexer, genre_encoder, publisher_encoder, developer_encoder, assembler])
    # feature_df = pipeline.fit(successful_games).transform(successful_games)

    # # Кластерный анализ
    # kmeans = KMeans(k=num_clusters, seed=1)
    # model = kmeans.fit(feature_df)
    # clusters = model.transform(feature_df)

    # # Вычисление характеристик каждого кластера
    # cluster_centers = model.clusterCenters()
    # for i, center in enumerate(cluster_centers):
    #     print(f"Cluster {i} center: {center}")

    # # Оценка кластерного анализа
    # evaluator = ClusteringEvaluator()
    # silhouette = evaluator.evaluate(clusters)
    # print(f"Silhouette with squared euclidean distance = {silhouette}")

    # # Группировка игр по кластерам и вычисление средних значений характеристик
    # cluster_profile = clusters.groupBy('prediction').agg(
    #     f.avg('critic_score').alias('avg_critic_score'),
    #     f.avg('total_sales').alias('avg_total_sales'),
    #     f.avg('na_sales').alias('avg_na_sales'),
    #     f.avg('jp_sales').alias('avg_jp_sales'),
    #     f.count('prediction').alias('count')
    # )

    # cluster_profile.show(truncate=False)
    # return cluster_profile