import pandas as pd

df = pd.read_parquet("../embeddings/metadata.parquet")
print(df.head(10))