from pyspark.sql import SparkSession

if __name__ == "__main__":
    session: SparkSession = SparkSession.builder \
        .master("local[*]") \
        .appName("Game Sales Analysis") \
        .getOrCreate()
    
    print(session)
