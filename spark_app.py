import time
import psutil
import logging
import os
import json
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from utils import parse_arguments


def main(num_nodes: int, optimize: bool):
    # Memory
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Перевод в МБ

    # Time & memory
    def measure_time(func):
        def wrapper(*args, **kwargs):
            nonlocal total_time, cumulative_times, total_mems
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            cumulative_times.append(total_time)
            mem = get_memory_usage()
            total_mems.append(mem)
            logger.info(f"TIME taken for {func.__name__}: {end_time - start_time:.2f} seconds")
            logger.info(f"MEMORY usage for {func.__name__}: {mem:.2f} MB")
            return result
        return wrapper
    
    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Init SparkSession
    spark = SparkSession.builder \
        .appName("TF-IDF Classification") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    spark.conf.set('spark.sql.caseSensitive', True)

    spark.sparkContext.setLogLevel("ERROR")

    cumulative_times = []
    total_mems = []
    total_time = 0

    @measure_time
    def load_data():
        df = spark.read.parquet("hdfs://namenode:9001/user/root/data/Musical_Instruments_5.parquet")
        return df.dropna(subset=['reviewText']).drop("vote").drop("style").drop("image")

    df = load_data()
    if optimize:
        if int(num_nodes) > 1:
            df = df.repartition(int(num_nodes)).cache()
        else:
            df = df.cache()
    df.show()
    logger.info("Data loading is done.\n")
    # df.printSchema()

    if optimize:
        rdd = spark.sparkContext.parallelize(df.collect(), numSlices=10)
        df = spark.createDataFrame(rdd, schema=df.schema)
        df.cache()

    @measure_time
    def tokenize_data(df):
        tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
        return tokenizer.transform(df)

    tokenized_df = tokenize_data(df)
    logger.info("Tokenization is done.\n")

    @measure_time
    def remove_stopwords(df):
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        return remover.transform(df)

    filtered_df = remove_stopwords(tokenized_df)
    logger.info("Stopwords removing is done.\n")

    @measure_time
    def apply_hashing_tf(df):
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
        return hashingTF.transform(df)

    hashed_df = apply_hashing_tf(filtered_df)
    logger.info("TF calculating is done.\n")

    @measure_time
    def apply_idf(df):
        idf = IDF(inputCol="raw_features", outputCol="features")
        idf_model = idf.fit(df)
        return idf_model.transform(df)

    rescaled_df = apply_idf(hashed_df)
    logger.info("IDF calculating is done.\n")

    @measure_time
    def index_labels(df):
        label_indexer = StringIndexer(inputCol="overall", outputCol="label")
        return label_indexer.fit(df).transform(df)

    indexed_df = index_labels(rescaled_df)
    logger.info("Label encoding is done.\n")

    @measure_time
    def train_model(df):
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
        #pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, label_indexer, lr])
        return lr.fit(df)

    (train_df, test_df) = indexed_df.randomSplit([0.8, 0.2], seed=42)
    if optimize:
        train_df = train_df.cache()
        test_df = test_df.cache()
    model = train_model(train_df)
    logger.info("Model training is done.\n")

    @measure_time
    def evaluate_model(model, df):
        predictions = model.transform(df)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        return predictions

    predictions = evaluate_model(model, test_df)
    logger.info("Model evaluation is done.\n")
    predictions.select("reviewText", "overall", "prediction").show()

    logger.info(f"Cumulative times: {cumulative_times}")
    logger.info(f"Memory consumtion per step: {total_mems}")

    metrics = {
            "time": cumulative_times,
            "memory": total_mems
        }
    if optimize:
        prefix = ""
    else:
        prefix = "not_"
    with open('metrics.json', 'a') as f:
        json.dump({f"{prefix}optim_{num_nodes}_node": metrics}, f)
        f.write("\n")
    logger.info("Metrics appended to metrics.json")

if __name__ == '__main__':
    args = parse_arguments()
    main(args.num_nodes, args.optimize)