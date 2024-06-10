import sys
from math import sqrt

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col
from pyspark.sql.functions import lag, avg, stddev, row_number

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

df = spark.read.csv(s3_input_path, header=True, inferSchema=True, dateFormat="M/d/yyyy").select("date", "close", "volume", "ticker")

window_spec = Window.partitionBy("ticker").orderBy("date")

# avg daily returns
data_close_price = df.withColumn("prev_close", lag("close").over(window_spec))
data_daily_returns = data_close_price.withColumn("daily_return", ((col("close") - col("prev_close")) / col("prev_close")) * 100)
# caching daily returns because it's being used multiple times
data_daily_returns = data_daily_returns.filter(col("daily_return").isNotNull()).cache()
average_daily_return = data_daily_returns.groupBy("date").agg(avg("daily_return").alias("average_return"))
average_daily_return.show(50, False)
average_daily_return.write.mode("overwrite").parquet(f"{s3_output_path}/average_daily_return")

# avg frequency
frequency = df.withColumn("frequency", col("close") * col("volume"))
average_frequency = frequency.groupBy("ticker").agg(avg("frequency").alias("frequency"))
average_frequency.show(10, False)
average_frequency.write.mode("overwrite").parquet(f"{s3_output_path}/average_frequency")

# most volatile stocks
volatility = data_daily_returns.groupBy("ticker").agg((stddev("daily_return") * sqrt(TRADING_DAYS)).alias("standard_deviation"))
volatility.show(10, False)
volatility.write.mode("overwrite").parquet(f"{s3_output_path}/volatility")

# top 30 day return dates
top_30_day_return_dates = (
        df.withColumn("prev_30_day_close", lag("close", 30).over(window_spec))
        .withColumn("30_day_return", ((col("close") - col("prev_30_day_close")) / col("prev_30_day_close")) * 100)
        .filter(col("30_day_return").isNotNull())
        .select("ticker", "date", "30_day_return")
        .orderBy(col("30_day_return").desc())
)
top_30_day_return_dates.write.mode("overwrite").parquet(f"{s3_output_path}/top_30_day_return_dates")
top_30_day_return_dates.limit(3).show(3, False)

job.commit()
