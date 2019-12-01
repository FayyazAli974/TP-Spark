package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}







object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    // lire le fichier sauvegardé précédemment
    val parquetFileDF = spark.read.parquet("/TP3_input/prepared_trainingset")

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("raw")


    val remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("filtered")


    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawfeatures")


    val idf = new IDF()
      .setInputCol("rawfeatures")
      .setOutputCol("tfidf")


    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")


    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))


    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa","goal","country_onehot","currency_onehot"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)


    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel,idf , indexer, indexer2, encoder, assembler, lr))


    val Array(training,test) = parquetFileDF.randomSplit(Array(0.9, 0.1))

    val model = pipeline.fit(training)

    val dfWithSimplePredictions = model.transform(test)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val accuracy = evaluator.evaluate(dfWithSimplePredictions)
    println(s"Accuracy reg log: ${accuracy}")


    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setParallelism(2)


    val model_best = trainValidationSplit.fit(training)

    val dfWithSimplePredictions_best = model_best.transform(test)



    dfWithSimplePredictions_best.select("features", "final_status", "predictions")
      .show()


    val accuracy_best = evaluator.evaluate(dfWithSimplePredictions_best)
    println(s"Accuracy best reg log: ${accuracy_best}")


    model.write.overwrite().save("/tmp/spark-log-model")
    model_best.write.overwrite().save("/tmp/spark-log-bestmodel")



    //


  }
}
