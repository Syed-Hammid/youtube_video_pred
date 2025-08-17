from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, dayofweek, desc, count, sum as _sum

# Start Spark session
spark = SparkSession.builder \
    .appName("YouTube Trending Analysis - 2025") \
    .config("spark.jars", "/spark/jars/mysql-connector-j-java-8.0.33.jar") \
    .getOrCreate()

# Load the dataset
mysql_url = "jdbc:mysql://localhost:3306/youtube_videos?useSSL=false&zeroDateTimeBehavior=round"

properties = {
    "user": "root",
    "password": "Root123$",
    "driver": "com.mysql.cj.jdbc.Driver"
}

query = "(SELECT video_id,title,channel,country,views,likes,comments, cast(published_at as char) AS published_at,CAST(fetch_date as CHAR) AS fetch_date from youtube_videos) as tmp"

df = spark.read.jdbc(url=mysql_url, table=query, properties=properties)

# View schema and sample
df.printSchema()
df.show(5)

# Convert date columns to proper format
df = df.withColumn("published_at", to_date("published_at")) \
       .withColumn("fetch_date", to_date("fetch_date"))

# Drop nulls and duplicates
df_clean = df.dropna().dropDuplicates(["video_id"])

# Save cleaned data
df_clean.coalesce(1).write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/cleaned_dataset")

# -------------------------- ANALYSIS ----------------------------

# 1. Top 10 most viewed videos
df_clean.select("video_id", "title", "channel", "views") \
    .orderBy(desc("views")).limit(10) \
    .write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/top_10_views")

# 2. Top 10 most liked videos
df_clean.select("video_id", "title", "channel", "likes") \
    .orderBy(desc("likes")).limit(10) \
    .write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/top_10_likes")

# 3. Top 10 most commented videos
df_clean.select("video_id", "title", "channel", "comments") \
    .orderBy(desc("comments")).limit(10) \
    .write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/top_10_comments")

# 4. Country-wise total views
df_clean.groupBy("country") \
    .agg(_sum("views").alias("total_views")) \
    .orderBy(desc("total_views")) \
    .write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/views_by_country")

# 5. Channel-wise trending count
df_clean.groupBy("channel") \
    .agg(count("*").alias("video_count")) \
    .orderBy(desc("video_count")) \
    .write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/top_channels")

# 6. Trend by day of week (using fetch_date)
df_clean = df_clean.withColumn("day_of_week", dayofweek("fetch_date"))
df_clean.groupBy("day_of_week") \
    .agg(count("*").alias("video_count")) \
    .orderBy("day_of_week") \
    .write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/trend_by_day")

# 7. Yearly publishing trend
df_clean = df_clean.withColumn("publish_year", year("published_at"))
df_clean.groupBy("publish_year") \
    .agg(count("*").alias("video_count")) \
    .orderBy("publish_year") \
    .write.mode("overwrite").option("header", True).csv("file:///home/iamsy/Project/output/yearly_publish_trend")

# Stop Spark
spark.stop()
