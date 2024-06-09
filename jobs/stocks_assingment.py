import sys
from math import sqrt

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col
from pyspark.sql.functions import lag, avg, stddev, rank

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_INPUT_PATH', 'S3_OUTPUT_PATH'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# make sure to add arguments: --S3_INPUT_PATH and --S3_OUTPUT_PATH
s3_input_path = args['S3_INPUT_PATH']
s3_output_path = args['S3_OUTPUT_PATH']


# the code
TRADING_DAYS = 252

df = spark.read.csv(s3_input_path, header=True, inferSchema=True)

window_spec = Window.partitionBy("ticker").orderBy("date")

# avg daily returns
data_close_price = df.withColumn("prev_close", lag("close").over(window_spec))
data_daily_returns = data_close_price.withColumn("daily_return", ((col("close") - col("prev_close")) / col("prev_close")) * 100)
# caching daily returns because it's being used multiple times
data_daily_returns = data_daily_returns.filter(col("daily_return").isNotNull()).cache()
average_daily_return = data_daily_returns.groupBy("date").agg(avg("daily_return").alias("average_daily_return"))
average_daily_return.coalesce(1).write.csv(f"{s3_output_path}/01-average_daily_return", header=True, mode="overwrite")

# avg frequency
frequency = df.withColumn("frequency", col("close") * col("volume"))
average_frequency = frequency.groupBy("ticker").agg(avg("frequency").alias("average_frequency"))
average_frequency.coalesce(1).write.csv(f"{s3_output_path}/02-average_frequency", header=True, mode="overwrite")

# most volatile stocks
volatility = data_daily_returns.groupBy("ticker").agg((stddev("daily_return") * sqrt(TRADING_DAYS)).alias("standard_deviation"))
volatility.coalesce(1).write.csv(f"{s3_output_path}/03-volatility", header=True, mode="overwrite")

# top 30 day return dates
window_spec_rank = Window.partitionBy("ticker").orderBy(col("30_day_return").desc())
top_30_day_return_dates = (
        df.withColumn("prev_30_day_close", lag("close", 30).over(window_spec))
        .withColumn("30_day_return", ((col("close") - col("prev_30_day_close")) / col("prev_30_day_close")) * 100)
        .filter(col("30_day_return").isNotNull())
        .withColumn("rank", rank().over(window_spec_rank))
        .filter(col("rank") <= 3)
        .select("ticker", "date", "30_day_return")
)
top_30_day_return_dates.coalesce(1).write.csv(f"{s3_output_path}/04-top_30_day_return_dates", header=True, mode="overwrite")


job.commit()
