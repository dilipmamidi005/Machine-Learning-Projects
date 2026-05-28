import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Dataset.csv')
df = df[
    [
        "Restaurant Name",
        "Cuisines",
        "Price range",
        "Votes",
        "Aggregate rating"
    ]
]
# Remove missing values
df = df.dropna()

# First 5 rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Dataset info
print("\nDataset Info:\n")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())
correlation = df[
    [
        "Votes",
        "Price range",
        "Aggregate rating"
    ]
].corr()

print("\nCorrelation Matrix:\n")
print(correlation)
plt.figure(figsize=(6,4))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()
top_cuisines = df["Cuisines"].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_cuisines.values,
    y=top_cuisines.index
)

plt.title("Top 10 Cuisines")

plt.xlabel("Count")

plt.ylabel("Cuisine")

plt.show()

# Votes vs Ratings
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df["Votes"],
    y=df["Aggregate rating"]
)

plt.title("Votes vs Ratings")

plt.show()

# Average rating by price range
avg_rating = df.groupby(
    "Price range"
)["Aggregate rating"].mean()

plt.figure(figsize=(6,4))

avg_rating.plot(kind="bar")

plt.title("Average Rating by Price Range")

plt.xlabel("Price Range")

plt.ylabel("Average Rating")

plt.show()

# Insights
print("\nInsights:")

print(
    "\n1. Restaurants with more votes generally have higher ratings."
)

print(
    "\n2. Certain cuisines appear more frequently in the dataset."
)

print(
    "\n3. Price range influences average restaurant ratings."
)