from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import DoubleType

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

# Example usage
if __name__ == "__main__":
    spark = SparkSession.builder.appName("NormalizeDataset").getOrCreate()
    data = [
        (1, 2.0, 3.0, "a"),
        (2, 4.0, 6.0, "b"),
        (3, 6.0, 9.0, "c")
    ]
    columns = ["id", "num1", "num2", "category"]
    numeric_cols = ["num1", "num2"]
    df = spark.createDataFrame(data, columns)
    normalized_df = normalize_dataset(df, numeric_cols)
    normalized_df.show()
