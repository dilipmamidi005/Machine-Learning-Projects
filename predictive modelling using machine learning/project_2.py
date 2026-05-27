import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Dataset.csv")

# Select important columns
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

# Encode categorical data
le = LabelEncoder()

df["Cuisines"] = le.fit_transform(df["Cuisines"])

# Features
X = df[
    [
        "Cuisines",
        "Price range",
        "Votes"
    ]
]

# Target
y = df["Aggregate rating"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict ratings
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)

print("\nR2 Score:", r2)

# Visualization
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Ratings")

plt.ylabel("Predicted Ratings")

plt.title("Actual vs Predicted Ratings")

plt.show()

# User prediction
restaurant_name = input("\nEnter Restaurant Name: ")

restaurant = df[
    df["Restaurant Name"].str.lower()
    ==
    restaurant_name.lower()
]

if not restaurant.empty:

    sample = restaurant[
        [
            "Cuisines",
            "Price range",
            "Votes"
        ]
    ]

    prediction = model.predict(sample)

    print(
        "\nPredicted Rating:",
        round(prediction[0], 2)
    )

else:
    print("\nRestaurant not found!")