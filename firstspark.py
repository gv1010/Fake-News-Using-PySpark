# Databricks notebook source
import pyspark
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('Basics').getOrCreate()

# COMMAND ----------

train = sqlContext.sql("SELECT * FROM train_csv")

# COMMAND ----------

test = sqlContext.sql("SELECT * FROM test_csv1")

# COMMAND ----------

# DBTITLE 1,Checking for null values
from pyspark.sql.functions import isnan, when, count, col
train.select([count(when(col(c).isNull(), c)).alias(c) for c in train.columns]).show()

# COMMAND ----------

# DBTITLE 1,Filling Null values with empty string
train = train.fillna(' ', subset=['title', 'text'])
train.select([count(when(col(c).isNull(), c)).alias(c) for c in train.columns]).show()

# COMMAND ----------

from pyspark.sql.functions import when
train = train.withColumn("author", \
              when(train["author"] == 'nan', ' ').otherwise(train["author"]))
train.show()

# COMMAND ----------

# DBTITLE 1,Test Data filling Null values and nan
 
from pyspark.sql.functions import isnan, when, count, col
test.select([count(when(col(c).isNull(), c)).alias(c) for c in test.columns]).show()

# COMMAND ----------

test = test.fillna(' ', subset=['title', 'text'])
test.select([count(when(col(c).isNull(), c)).alias(c) for c in test.columns]).show()

# COMMAND ----------

test = test.withColumn("author", \
              when(test["author"] == 'nan', ' ').otherwise(test["author"]))
test.show(5)

# COMMAND ----------

df = train

# COMMAND ----------

# DBTITLE 1,Combining title, author and text columns
from pyspark.sql.functions import concat_ws, concat
df = train
df = df.withColumn('combi_text', concat_ws(' ', df.title, df.author, df.text))
df = df.drop('title','author','text')
df.show(1,truncate=False)

# COMMAND ----------

#Test data combining title, author and text
df_test = test
df_test = df_test.withColumn('combi_text', concat_ws(' ', df_test.title, df_test.author, df_test.text))
df_test = df_test.drop('title','author','text')
df_test.show(1,truncate=False)

# COMMAND ----------

# DBTITLE 1,Text Preprocessing
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, RegexTokenizer,Word2Vec, HashingTF
from pyspark.ml.feature import StandardScaler
 
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
def nlpTransform(data):
  tokenizer = Tokenizer(inputCol="combi_text", outputCol="words")
  wordsData = tokenizer.transform(data)
  hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
  featurizedData = hashingTF.transform(wordsData)
  scaler = StandardScaler(inputCol="rawFeatures", outputCol="features", withStd=True, withMean=False)
  featureData = scaler.fit(featurizedData)
  featureD = featureData.transform(featurizedData)
  return featureD

# COMMAND ----------

# DBTITLE 1,Train Data Featurization
data = nlpTransform(df)
data = data.drop('combi_text', 'words', 'rawFeatures')

# COMMAND ----------

data.show(3)

# COMMAND ----------

# DBTITLE 1,Test Data featurization
testData = nlpTransform(df_test)
testData = testData.drop('combi_text', 'words','rawFeatures')
testData.show(1)

# COMMAND ----------

# DBTITLE 1,Train Test Split of data
train_data, test_data = data.randomSplit([0.75, 0.25], seed=12345)

# COMMAND ----------

# DBTITLE 1,Modelling
from pyspark.ml.classification import LogisticRegression
model=LogisticRegression(labelCol='label',maxIter=5, regParam=0.001)           
model=model.fit(train_data)                                                          
summary=model.summary
summary.predictions.describe().show() 

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictions=model.evaluate(test_data)

# COMMAND ----------

evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
evaluator.evaluate(predictions.predictions)

# COMMAND ----------

#Achieved 96.3% accuracy on test data, lets predict on the unseen test data for kaggle submission.

# COMMAND ----------

results=model.transform(test_data)
results.select('features','prediction').show()

# COMMAND ----------

results.count()

# COMMAND ----------

results=model.transform(testData)
results.select('features','prediction').show()

# COMMAND ----------

results.count()

# COMMAND ----------

values = list(results.select('prediction').toPandas()['prediction'])

# COMMAND ----------

print(values)

# COMMAND ----------

import pandas as pd
res = pd.DataFrame(values)
res.head()
