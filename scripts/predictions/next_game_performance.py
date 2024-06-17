from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

def predict_next_game_performance(spark: SparkSession, df: DataFrame, series_name: str):
    df = df.na.drop()

    # Фильтрация данных для серии
    series_data = df.filter(lower(col("title")).like(f"%{series_name.lower()}%"))
    
    series_data.show(50, False)

   # Выбор числовых признаков
    numeric_columns = ['critic_score', 'total_sales', 'na_sales', 'jp_sales']
    features = numeric_columns.copy()
    features.remove('critic_score')  # Удаляем один из столбцов из признаков

    assembler = VectorAssembler(
        inputCols=features, 
        outputCol='features'
    )
    series_data = assembler.transform(series_data)

    # Выбор целевых переменных
    target_columns = ['critic_score', 'total_sales', 'na_sales', 'jp_sales']

    # Разделение данных на обучающую и тестовую выборки
    train_data, test_data = series_data.randomSplit([0.8, 0.2], seed=42)
    train_data.show()
    test_data.show()

    predictions = {}

    # Обучение модели для каждой целевой переменной и предсказание
    for target in target_columns:
        print(f"Training model for {target}...")
        
        # Определение модели
        rf = RandomForestRegressor(featuresCol='features', labelCol=target, numTrees=100, seed=42)
        
        # Обучение модели
        model = rf.fit(train_data)
        
        # Предсказание
        prediction = model.transform(test_data)
        
        # Оценка модели
        evaluator = RegressionEvaluator(labelCol=target, predictionCol='prediction', metricName='r2')
        r2 = evaluator.evaluate(prediction)
        print(f"R^2 for {target}: {r2}")
        
        # Предсказание для новой игры
        new_game_features = prediction.select('features').collect()[-1][0]
        new_game = spark.createDataFrame([(new_game_features,)], ['features'])
        next_game_prediction = model.transform(new_game)
        predicted_value = next_game_prediction.select('prediction').collect()[0][0]
        predictions[target] = predicted_value

    print("Predicted values for the next game in the series:")
    for target, value in predictions.items():
        print(f"{target}: {value}")

    # Остановка сессии Spark
    spark.stop()