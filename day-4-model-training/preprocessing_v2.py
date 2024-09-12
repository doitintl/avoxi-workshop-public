from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StringType, NumericType, DoubleType
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
import subprocess
import argparse
import sys

import phonenumbers
from phonenumbers import geocoder

#https://github.com/azharkhn/libphonenumber-api/blob/master/phonenumber/lib/phonenumber.py
def get_E164format(phonenumber):
    if phonenumber[:2] == '00':
        return '+1' + phonenumber
    else:
        return '+' + phonenumber 

def get_country_from_phone(phone_number):
    formated_num = get_E164format(phone_number)
    try:
        country = geocoder.country_name_for_number(phonenumbers.parse(formated_num), "en")
        return country if country else "Invalid" #Empty is invalid
    except phonenumbers.phonenumberutil.NumberParseException:
        return "Invalid"

# Define categorize_columns
def categorize_columns(df, list3_columns):
    column_types = dict(df.dtypes)
    list1 = [col for col, dtype in column_types.items() 
            if dtype not in ('int', 'double', 'float', 'long')]
    list2 = [col for col, dtype in column_types.items() 
            if dtype in ('int', 'double', 'float', 'long') 
            and not any(col in l3_col for l3_col in list3_columns)]
    return list1, list2

def normalize_dataset(df, numeric_cols):
    spark = SparkSession.builder.appName("NormalizeDataset").getOrCreate()

    # Assemble the numeric columns into a vector
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    assembled_df = assembler.transform(df)

    # Normalize the features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    # Extract the scaled features back into individual columns
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import DoubleType

    def extract_element(vec, index):
        return float(vec[index])

    extract_udf = udf(extract_element, DoubleType())

    for i, col_name in enumerate(numeric_cols):
        scaled_df = scaled_df.withColumn(col_name, extract_udf(col("scaledFeatures"), lit(i)))

    # Drop the intermediate columns
    result_df = scaled_df.drop("features", "scaledFeatures")

    print("Processed DataFrame; normalized DataFrame created")
    return result_df

def sparkShape(dataFrame):
    return (dataFrame.count(), len(dataFrame.columns))

def run(argv=None):
    import pyspark
    pyspark.sql.dataframe.DataFrame.shape = sparkShape
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        dest='input',
                        help='Input file to process.')
    parser.add_argument('--output',
                        dest='output',
                        help='Output file to write results to.')
    parser.add_argument('--anomaly_output',
                        dest='anomaly_output',
                        help='Output file for anomalies to write results to.')
    parser.add_argument('--no_anomaly_output',
                        dest='no_anomaly_output',
                        help='Output file for no anomalies to write results to.')
    parser.add_argument('--anomaly_normalized_output',
                        dest='anomaly_normalized_output',
                        help='Output file for anomalies to write results to.')
    parser.add_argument('--no_anomaly_normalized_output',
                        dest='no_anomaly_normalized_output',
                        help='Output file for no anomalies to write results to.')

    known_args, _ = parser.parse_known_args(argv)

    # Create a Spark session
    spark = SparkSession.builder \
        .appName('avoxi_data_parser') \
        .getOrCreate()

    # Load CSV files into a DataFrame
    print(f"Reading from {known_args.input}")
    df = spark.read.csv(known_args.input, header=True, inferSchema=True)
 
    expected_columns = {
        '164_from_caller_id': ['e164_from_caller_id', 'from_number'],
        '164_to_caller_id': ['e164_to_caller_id', 'to_number'],
        'mean_opinion_score': ['mos'],
        'duration': ['duration_seconds']
    }

    # Loop through the expected columns and rename if necessary
    for expected, alternatives in expected_columns.items():
        found = False
        for actual in alternatives:
            if actual in df.columns:
                df = df.withColumnRenamed(actual, expected)
                found = True
                break
        if not found and expected not in df.columns:
            raise ValueError(f"None of {expected} or alternatives {alternatives} found in the DataFrame")
        
    # Print Schema
    df.printSchema()
    print(df.shape())

    # Remove rows with any null values
    df = df.drop('status', 'data_center', 'carrier_id','label','origination','destination')
    df = df.dropna()
    columns = df.columns
    
    # Print Schema
    print("Shape after dropping nulls")
    print(df.shape())
    
    df = df.withColumn("164_from_caller_id", col("164_from_caller_id").cast("string"))
    df = df.withColumn("164_to_caller_id", col("164_to_caller_id").cast("string"))
    if 'carrier_id' in columns:
        df = df.withColumn("carrier_id", col("carrier_id").cast("string"))
    
    # Register UDF
    get_country_from_phone_udf = udf(get_country_from_phone, StringType())

    # Apply the UDF to the DataFrame
    df = df.withColumn("from_country", get_country_from_phone_udf(col('164_from_caller_id')))
    df = df.withColumn("to_country", get_country_from_phone_udf(col('164_to_caller_id')))

    # Filter out invalid phone numbers early
    df = df.filter((col('from_country') != 'Invalid') & (col('to_country') != 'Invalid'))

    # Print Schema
    print("Shape after removing invalid countries")
    print(df.shape())
    
    # Identify non-numeric columns
    columns_to_exclude = ['day', 'hour']
    non_numeric_cols, numeric_cols = categorize_columns(df, columns_to_exclude)

    print(f"Normalizing columns: {numeric_cols}")
    print(f"Omitting columns: {non_numeric_cols}")

    
    # Step 1: Compute quantiles for reuse
    jitter_median = df.approxQuantile('jitter', [0.50], 0.10)[0]
    mos_quartile_25 = df.approxQuantile('mean_opinion_score', [0.25], 0.05)[0]
    duration_median = df.approxQuantile('duration', [0.50], 0.05)[0]

    print(f"Jitter Median: {jitter_median}")
    print(f"Mean Opinion Score 25th Quartile: {mos_quartile_25}")
    print(f"Duration Median: {duration_median}")
    
    # Step 2: Create anomalies DataFrame
    anomalies_df = df.filter(
        (col('packet_loss') > 0.1) &
        (col('jitter') > jitter_median) &
        (col('mean_opinion_score') < mos_quartile_25)
    )

    # Step 3: Print Schema and Shape for anomalies_df
    anomalies_df.printSchema()
    print(f"Shape of anomalies_df: {anomalies_df.shape()}")

    # Step 4:  Normalized anomalies_df
    anomalies_df_normalized = normalize_dataset(anomalies_df, numeric_cols)

    # Step 5: Write anomalies DataFrame to GCS
    print(f"Writing anomalies to {known_args.anomaly_output}")
    anomalies_df.write.mode('overwrite').option("header", "true").csv(known_args.anomaly_output)
    
    print(f"Writing normalized anomalies to {known_args.anomaly_normalized_output}")
    anomalies_df.write.mode('overwrite').option("header", "true").csv(known_args.anomaly_normalized_output)

    # Step 6: Create no anomalies DataFrame
    no_anomalies_df = df.filter(
        (col('duration') > duration_median) &
        (col('mean_opinion_score') > 4) &
        (col('jitter') <= 1) &
        (col('packet_loss') <= 0)
    )

    # Step 7: Print Schema and Shape for no_anomalies_df
    no_anomalies_df.printSchema()
    print(f"Shape of no_anomalies_df: {no_anomalies_df.shape()}")
    
    # Step 8: Normalized no_anomalies_df
    no_anomalies_df_normalized = normalize_dataset(no_anomalies_df, numeric_cols)

    # Step 9: Write no anomalies DataFrame to GCS
    print(f"Writing 'normal data' to {known_args.no_anomaly_output}")
    no_anomalies_df.write.mode('overwrite').option("header", "true").csv(known_args.no_anomaly_output)
    
    print(f"Writing 'normal normalized data' to {known_args.no_anomaly_normalized_output}")
    no_anomalies_df_normalized.write.mode('overwrite').option("header", "true").csv(known_args.no_anomaly_normalized_output)    

    # Stop the Spark session
    spark.stop()
