package com.corelogic.ttv.textMining

// Rebuilding text mining code in Scala (native for Spark) with Hive querying on IDAP

/////////////////////////////////////////////////////
//// LOADING AND DEFINING METHODS AND PARAMETERS ////
/////////////////////////////////////////////////////
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession._
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.hive._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.DataFrameStatFunctions
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{ HashingTF, IDF, RegexTokenizer, StopWordsRemover, NGram, StringIndexer, IndexToString, VectorAssembler, SQLTransformer }
import org.apache.spark.ml.classification.{ LogisticRegression }
import org.apache.spark.ml.regression.{ LinearRegression }
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import scala.collection.breakOut
import sys.process._
import org.apache.hadoop.fs.{ FileSystem, Path }

object ttvApp {

  def main(args: Array[String]) {

    // LOAD PARAMS
    val strs = loadStringMap(args(1))

    // BUILD SPARK SESSION
    val ss = createSS(args)
    
    // Fix for connection pool issue
    val sc = ss.sparkContext
    val hc = sc.hadoopConfiguration
    hc.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hc.setInt("fs.s3a.connection.maximum", 1000)
    
    ss.sparkContext.setLogLevel("ERROR")

    // Define save path
    val savePth = defineSavePth(args(1), ss)

    // CREATE EVALUATOR
    val evaluator = new MulticlassClassificationEvaluator().setMetricName(strs("evalMetric"))

    // CREATE HIVE QUERY
    val strSql = createSql(args(1))

    // RUN HIVE
    val dataSql = ss.sql(strSql)

    val indexer = new StringIndexer()
      .setInputCol("dep")
      .setOutputCol("label")
      .fit(dataSql) //.transform(dataSql)

    indexer.write.overwrite().save(s"${savePth}/testing")

    ss.stop()

  }

  // Functions to distribute throughout script

  // CREATE STRING PARAMETERS - WILL DIFFER BASED ON WHICH VARIABLE YOU WANT TO TRAIN ON
  def loadStringMap(varNm: String): Map[String, String] = {

    val doubs = loadDoublesMap()
    val ints = loadIntMap()
    val text = "PublicRemarks"

    // PARAMETERIZED STRINGS BASED ON VARIABLE SELECTION
    // **depCase follows conditional case format: WHEN varname IN ([list]) THEN varname|'condition' 
    // **depOther assigns ELSE in case statement
    // **cond adds to where clause (start w/ AND)
    // **classification to determine
    val (depCase: String, depOther: String, cond: String, classification: String) =
      if (varNm == "PoolPresent") {
        (s""" 
          WHEN $varNm IN ('Y') THEN $varNm 
          """, //depCase
          s""" 'N' """, // depOther
          s""" 
            $text IS NOT NULL AND $text NOT IN ('', ' ')
            AND $varNm IS NOT NULL AND $varNm NOT IN ('', ' ')
            """, // cond
          "true" // classification
          )
      } else if (varNm == "CoolingPresent") {
        (s""" 
          WHEN $varNm IN ('Y') THEN $varNm 
          """, //depCase
          s""" 'N' """, // depOther
          s""" 
            $text IS NOT NULL AND $text NOT IN ('', ' ')
            AND $varNm IS NOT NULL AND $varNm NOT IN ('', ' ')
            """, // cond
          "true" // classification
          )
      } else if (varNm == "FA_GarageStyle") {
        (s""" 
          WHEN $varNm IN ('A', 'I', 'T') THEN 'attached' 
          WHEN $varNm IN ('D', 'L', 'V') THEN 'detached'
          WHEN $varNm IN ('H') THEN 'none'
          """, //depCase
          s""" 'other' """, // depOther
          s""" 
            $text IS NOT NULL AND $text NOT IN ('', ' ')
            AND $varNm IS NOT NULL AND $varNm NOT IN ('', ' ')
            """, // cond
          "true" // classification
          )
      } else if (varNm == "FA_SquareFeet") {
        (s""" 
          WHEN $varNm > 0 THEN $varNm
          """, //depCase
          s""" 0 """, // depOther
          s""" 
            $text IS NOT NULL AND $text NOT IN ('', ' ')
            AND $varNm > 0
            """, // cond
          "false" // classification
          )
      } else if (varNm == "lotSizeAreaAcres") {
        (s""" 
          WHEN $varNm IS NULL AND lotsizeareasqfeet IS NOT NULL THEN lotsizeareasqfeet/43560
          """, //depCase
          s""" $varNm """, // depOther
          s""" $text IS NOT NULL AND $text not in ('', ' ')
            AND (lotsizeareasqfeet > 0 OR $varNm > 0)
            """, // cond
          "false" // classification
          )
      }

    // CREATE FINAL MAP AFTER IF CONDITIONS ARE EVALUATED FOR VARIABLE SELECTION
    val tableMap: Map[String, String] = Map(
      "text" -> text,
      "dep" -> varNm,
      "depCase" -> depCase,
      "depOther" -> depOther,
      "seed" -> "786",
      "trainProportion" -> doubs("trainProportion").toString,
      "tbl" -> "default.mlsquicksearchmaster_snappy",
      "limit" -> ("limit " + ints("smplSize").toString), // ALWAYS START WITH "limit"
      "cond" -> cond,
      "regexStr" -> "(?<=\\p{Alpha}|\\p{Space}|\\p{Punct})\\p{Punct}|\\p{Punct}(?=\\p{Alpha}|\\p{Space}|\\p{Punct})|\\s",
      "evalMetric" -> "accuracy",
      "classification" -> classification.toString)

    return tableMap

  }

  // INTEGER PARAMS - INCLUDES DYNAMIC HASH SIZE AND SAMPLE SIZE
  def loadIntMap(): Map[String, Int] = {

    val smplSize = 2000
    val hashPwrDecision = (scala.math.log(smplSize) / scala.math.log(2)).ceil // 4 for testing, dynamic dimensionality rounds up to nearest power of 2 for smpl size
    val hashPwr = if (hashPwrDecision < 24) { hashPwrDecision } else { 24 } // Hard ceiling for hash size at 2^24, 16.8M (probably will contain most relevant words & n-grams sans collisions) 

    val wordMap: Map[String, Int] = Map(
      "seed" -> 786,
      "hashSize" -> scala.math.pow(2, hashPwr).toInt,
      "kfolds" -> 2, // 2 for testing, 3 for very large datasets, preferably 5 or 10
      "logitMaxIter" -> scala.math.pow(10, 5).toInt, // 1 for testing
      "smplSize" -> smplSize)

    return wordMap

  }

  // DOUBLE PARAMS - ELASTIC NET AND TRAIN/TEST SPLIT
  def loadDoublesMap(): Map[String, Double] = {

    val doubleMap: Map[String, Double] = Map(
      "logitElasticNet" -> 0.8,
      "trainProportion" -> 0.9)

    return doubleMap

  }

  // CREATES CV OBJECT BY CREATING PIPELINE, EVALUATOR, AND PARAM GRID OBJECTS
  def createCV(varNm: String, data: DataFrame): org.apache.spark.ml.feature.StringIndexerModel = { //tuning.CrossValidator = {

    val strs = loadStringMap(varNm)
    val ints = loadIntMap()
    val doubs = loadDoublesMap()

    // Define pipeline objects
    val indexer = new StringIndexer()
      .setInputCol("dep")
      .setOutputCol("label")
      .fit(data)
    return indexer
  }

  // CREATE HIVE QUERY FOR PULLING DATA FROM DATABASE
  def createSql(varNm: String): String = {

    val strs = loadStringMap(varNm)

    val strSql = s"""
        select 
          case 
            ${strs("depCase")}
            else ${strs("depOther")}
          end AS dep
          , ${strs("text")} AS text
          --, syspropertyid, fa_listid, cmas_fips_code, cmas_parcel_id, cmas_parcel_seq_nbr, fa_landuse -- Unique IDs and LandUse
          --, addressstreetaddress, addresscity, addresscounty, addressstate, addresspostalcode -- Address Info
        from ${strs("tbl")}
        where ${strs("cond")}
        ${strs("limit")}
    """

    return strSql

  }

  // PROPORTIONALLY SPLITS GIVEN DATAFRAME USING SPARK SQL FUNCTIONS
  def stratifiedProportionalSplit(dataSql: DataFrame, varNm: String, ss: SparkSession): (DataFrame, DataFrame) = {

    val strs = loadStringMap(varNm)
    val doubs = loadDoublesMap()

    import ss.implicits._

    // Stratified, proportionally-balanced splitting
    val dataSql2 = dataSql.withColumn("rndm", rand())
    val dataSql3 = dataSql2.withColumn("nbr", row_number.over(Window.partitionBy("dep").orderBy("rndm")))
    val dataRaw = dataSql3.withColumn("prt", $"nbr" / max($"nbr").over(Window.partitionBy("dep")))

    // Split data into train and test
    val trainingData = dataRaw
      .where($"prt" < doubs("trainProportion"))
      .select($"dep", $"text",
        $"syspropertyid", $"fa_listid", $"cmas_fips_code", $"cmas_parcel_id", $"cmas_parcel_seq_nbr", $"fa_landuse",
        $"addressstreetaddress", $"addresscity", $"addresscounty", $"addressstate", $"addresspostalcode")
    val testData = dataRaw
      .where($"prt" >= doubs("trainProportion"))
      .select($"dep", $"text",
        $"syspropertyid", $"fa_listid", $"cmas_fips_code", $"cmas_parcel_id", $"cmas_parcel_seq_nbr", $"fa_landuse",
        $"addressstreetaddress", $"addresscity", $"addresscounty", $"addressstate", $"addresspostalcode")

    return (trainingData, testData)

  }

  // DEFINES ITERATIVE SAVE PATH FOR NEWEST MODEL
  def defineSavePth(varNm: String, ss: SparkSession): String = {

    val strs = loadStringMap(varNm)

    // Define save path

    val checkPth = new Path(s"/user/nsuguru/output/${strs("dep")}")
    FileSystem.get(ss.sparkContext.hadoopConfiguration).mkdirs(checkPth) // Make directory if not exist
    val iterCmd = s"hadoop fs -ls ${checkPth.toString}" #| "tail -n +2" #| "wc -l"
    val iterNum = iterCmd.!!.trim.toInt + 1
    println("MODEL ITERATION NUM:" + iterNum)
    val savePth = s"${checkPth.toString}/${iterNum.toString}"

    return savePth

  }

  def createSS(args: Array[String]): SparkSession = {

    // BUILD SPARK SESSION
    val ss = SparkSession.builder()
      .appName(s"""Text Analytics, Pipeline Training, Classification, ${args(1)}""")
      .master(args(0))
      .enableHiveSupport()
      .getOrCreate()

    return ss

  }
}