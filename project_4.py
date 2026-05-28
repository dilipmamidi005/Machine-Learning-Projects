import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# First 5 rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Dataset info
print("\nDataset Info:\n")
print(df.info())

# Missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())
top_products = df.groupby("Product Name")["Sales"].sum().sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title("Top 10 Products by Sales")

plt.xlabel("Sales")

plt.ylabel("Products")

plt.show()
# Sales by Region
region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(6,4))

region_sales.plot(kind="bar")

plt.title("Sales by Region")

plt.xlabel("Region")

plt.ylabel("Sales")

plt.show()
correlation = df[
    [
        "Sales",
        "Profit",
        "Quantity",
        "Discount"
    ]
].corr()

plt.figure(figsize=(6,4))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()
X = df[
    [
        "Sales",
        "Quantity",
        "Discount"
    ]
]

y = df["Profit"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)

print("\nR2 Score:", r2)

# Actual vs Predicted Profit
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Profit")

plt.ylabel("Predicted Profit")

plt.title("Actual vs Predicted Profit")

plt.show()

# Insights
print("\nInsights:")

print(
    "\n1. Certain products generate significantly higher sales."
)

print(
    "\n2. Regional sales performance varies across regions."
)

print(
    "\n3. Sales and profit are positively correlated."
)

print(
    "\n4. Discounts can negatively affect profit."
)
