import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Dataset.csv")
print("these are the first 5 rows")
print(df.head())
print("these are the last 5 rows")
print(df.tail())
print("\nDataset Info:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())
# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:")
print(df.shape)
print("\nSummary:\n")
print(df.describe())
plt.figure(figsize=(8,5))

sns.histplot(df["Aggregate rating"], bins=10)

plt.title("Ratings Distribution")

plt.show()
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df["Votes"],
    y=df["Aggregate rating"]
)

plt.title("Votes vs Ratings")

plt.show()
top_cuisines = df["Cuisines"].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_cuisines.values,
    y=top_cuisines.index
)

plt.title("Top 10 Cuisines")

plt.show()