from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, unix_timestamp

def predict_next_game_performance(
    spark: SparkSession,
    df: DataFrame, 
    series_name: str,
    platform: str = None
):
    df = df.drop("last_update").na.drop()

    # Фильтрация данных для серии
    filtered_data: DataFrame = df.filter(lower(col("title")).like(f"%{series_name.lower()}%"))
    
    if (platform is not None):
        filtered_data = filtered_data.filter(lower(col("console")).like(platform.lower()))
        
    filtered_data = filtered_data.withColumn('release_date_numeric', unix_timestamp('release_date'))
    filtered_data = filtered_data.orderBy('release_date')
    
    last_release_date_numeric: int = filtered_data.select('release_date_numeric').orderBy(col('release_date_numeric').desc()).first()[0]
    next_release_date_numeric: int = last_release_date_numeric + 365 * 24 * 60 * 60
    
    filtered_data.show(50, False)

    # Предсказание значений
    predictions = train_and_predict(spark, filtered_data, next_release_date_numeric)

    # Вывод предсказанных значений
    print(f"Predicted Critic Score: {predictions['critic_score']}")
    print(f"Predicted Total Sales: {predictions['total_sales']}")
    print(f"Predicted NA Sales: {predictions['na_sales']}")
    print(f"Predicted JP Sales: {predictions['jp_sales']}")


def train_and_predict(spark: SparkSession, df: DataFrame, next_release_date_numeric: int, seed=-4200000):
    predictions = {}
    features_col = 'release_date_numeric'
    targets = ['critic_score', 'total_sales', 'na_sales', 'jp_sales']
    
    # Создание вектора признаков один раз перед циклом
    assembler = VectorAssembler(inputCols=[features_col], outputCol="features", handleInvalid="skip")
    df = assembler.transform(df)
    
    # Разделение данных на обучающую и тестовую выборки
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
    
    #train_df.show()
    #test_df.show()
    
    for target in targets:
        # Определение и обучение модели линейной регрессии
        lr = LinearRegression(featuresCol='features', labelCol=target)
        lr_model = lr.fit(train_df)
        
        # Оценка модели на тестовой выборке
        predictions_test = lr_model.transform(test_df)
        evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions_test)
        print(f"evaluator: {evaluator}")
        print(f"Root Mean Squared Error (RMSE) on test data for {target}: {rmse}")
        
        # Предсказание для следующей даты релиза
        next_data = spark.createDataFrame([(next_release_date_numeric,)], [features_col])
        next_data = assembler.transform(next_data)
        prediction = lr_model.transform(next_data).select("prediction").collect()[0][0]
        
        predictions[target] = prediction

    return predictions




def predict_next_game_performance_random_tree(
    spark: SparkSession, 
    df: DataFrame, 
    series_name: str,
    platform: str = None
):
    df = df.na.drop()

    # Фильтрация данных для серии
    series_data = df.filter(lower(col("title")).like(f"%{series_name.lower()}%"))
    
    if (platform is not None):
        series_data = series_data.filter(lower(col("console")).like(platform.lower()))

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
    #train_data.show()
    #test_data.show()

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
