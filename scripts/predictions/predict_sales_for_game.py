from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

from utils.spark_utils import column_dict as cd


def predict_game_success(df: DataFrame, spark: SparkSession, game_series_name: str, new_game_data: list):
    # Фильтруем данные по названию серии игры
    filtered_df = df.filter(lower(col("title")).like(f"%{game_series_name.lower()}%"))

    # Обрабатываем данные: выбираем необходимые колонки и заполняем пропуски
    processed_df = filtered_df.select(
        col("title"),
        col("critic_score").cast("double"),
        col("total_sales").cast("double"),
        col("na_sales").cast("double"),
        col("jp_sales").cast("double"),
        col("pal_sales").cast("double"),
        col("other_sales").cast("double"),
        col("release_date")
    ).na.drop()

    regions = list(filter(lambda r: r.endswith("sales"), list(cd.values())))

    # Разделяем данные на признаки и метку
    assembler = VectorAssembler(inputCols=[cd["critic_score"], *regions], outputCol="features")
    final_df = assembler.transform(processed_df)
    transformed_df = final_df.select("title", "features", "total_sales", "release_date").sort("release_date")

    transformed_df.show(50)

    # Разделяем данные на обучающую и тестовую выборки
    train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

    # Обучаем модель линейной регрессии
    lr = LinearRegression(labelCol="total_sales", featuresCol="features", regParam=0.1, solver="normal")
    lr_model = lr.fit(train_df)

    # Делаем прогнозы на тестовой выборке
    predictions = lr_model.transform(test_df)

    # Оцениваем модель
    evaluator = RegressionEvaluator(labelCol="total_sales", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE: {rmse}")

    # Прогнозируем успех следующей игры серии
    new_game: DataFrame = spark.createDataFrame(new_game_data, ["title", "critic_score", *regions, 'release_date'])
    new_game.show()

    # Обрабатываем новые данные
    new_game_features: DataFrame = assembler.transform(new_game).select(["title", "release_date", "features"])

    # Делаем прогноз
    new_game_prediction = lr_model.transform(new_game_features)
    new_game_prediction.show(truncate=False)

    # Возвращаем предсказанное значение
    return new_game_prediction.collect()
