package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf, lower, concat, lit,when, length, round}




object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // ##################### UDF ########################
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }


    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    def NbJoursCamp(launched_at: Int, deadline: Int): Int = {
      (deadline - launched_at)/60/60/24
    }

    def HoursPrepa(created_at: Float, launched_at:Float): Double = {
      (launched_at - created_at)/60/60
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)
    val HoursPrepaUdf = udf(HoursPrepa _)
    val NbJoursCampUdf = udf(NbJoursCamp _)
    // ############################################################


    // ############# Créations de DataFrame ########################
    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("/data/train_clean.csv")


    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))


    val df2: DataFrame = dfCasted.drop("disable_communication")


    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")


    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")


    val df_finalstatus: DataFrame = dfCountry
      .where($"final_status" === "0" || $"final_status" === "1")


    val df_time: DataFrame = df_finalstatus
      .withColumn("days_campaign",NbJoursCampUdf($"launched_at",$"deadline"))
      .withColumn("hours_prepa",HoursPrepaUdf($"created_at",$"launched_at"))
      .withColumn("hours_prepa",round($"hours_prepa",3))
      .drop("launched_at","created_at","deadline")


    val df_texte: DataFrame = df_time
      .withColumn("name_2",lower($"name"))
      .withColumn("desc_2",lower($"desc"))
      .withColumn("keywords_2",lower($"keywords"))
      .drop("name","desc","keywords")
      .withColumnRenamed("name_2","name")
      .withColumnRenamed("desc_2","desc")
      .withColumnRenamed("keywords_2","keywords")
      .withColumn("text",concat(($"name"),lit(" "),($"desc"),lit(" "),($"keywords")))


    val df_final: DataFrame = df_texte
      .withColumn("days_campaign", when($"days_campaign".isNull, -1).otherwise($"days_campaign"))
      .withColumn("hours_prepa", when($"hours_prepa".isNull, -1).otherwise($"hours_prepa"))
      .withColumn("goal", when($"goal".isNull, -1).otherwise($"goal"))
      .withColumn("country2", when($"country2".isNull,"unknown").otherwise($"country2"))
      .withColumn("currency2", when($"currency2".isNull,"unknown").otherwise($"currency2"))
    //#########################################################


    // ##################### Ecriture ##########################
    df_final.write.parquet("/TP2_Output")
    // #########################################################


    // ########################## Print #######################
    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")
    println(s"Nombre de lignes : ${df.count}")
    println("\n")
    println(s"Nombre de colonnes : ${df.columns.length}")
    println("\n")
    df.show()
    println("\n")
    df.printSchema()
    println("\n")
    dfCasted.printSchema()
    println("\n")
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show
    println("\n")
    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)
    println("\n")
    dfCountry.filter($"country2" === "False")
      .groupBy("currency2")
      .count
      .orderBy($"count".desc)
      .show(50)
    println("\n")
    dfNoFutur
      .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      .drop("country", "currency")
      .printSchema()
    println("\n")
    dfCountry.groupBy("final_status")
      .count()
      .orderBy(-$"count")
      .show(50)
    println("\n")
    df_finalstatus.groupBy("final_status")
      .count()
      .orderBy(-$"count")
      .show(50)
    println("\n")
    df_time.select("days_campaign").show()
    println("\n")
    df_time.select("hours_prepa").show()
    println("\n")
    df_time.printSchema()
    println("\n")
    df_texte.printSchema()
    println("\n")
    df_texte.select("text").show()
    println("\n")
    df_final.select("currency2")
      .where($"currency2" === "unknown")
      .show(100)
    println("\n")
    df_final.groupBy("days_campaign")
      .count()
      .orderBy($"count")
      .show(100)


  }
}
