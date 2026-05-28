import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Dataset

data = {
    "Restaurant": [
        "Pizza Hut",
        "Dominos",
        "KFC",
        "Burger King",
        "Paradise",
        "Bawarchi",
        "Starbucks",
        "Subway",
        "Foodies",
        "Spicy House"
    ],

    "Cuisine": [
        "Italian",
        "Italian",
        "Fast Food",
        "Fast Food",
        "Biryani",
        "Biryani",
        "Cafe",
        "Healthy",
        "Italian",
        "Biryani"
    ],

    "Price_Range": [
        "Medium",
        "Low",
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
        "Medium",
        "High",
        "Medium"
    ],

    "Votes": [
        200,
        180,
        150,
        140,
        300,
        280,
        220,
        170,
        260,
        240
    ],

    # Some ratings are missing
    "Rating": [
        4.3,
        4.1,
        3.9,
        3.8,
        4.7,
        4.6,
        4.4,
        4.0,
        None,
        None
    ]
}

df = pd.DataFrame(data)

# Split training and prediction data

train_df = df[df["Rating"].notnull()]
predict_df = df[df["Rating"].isnull()]

# Encoding

le_cuisine = LabelEncoder()
le_price = LabelEncoder()

df["Cuisine"] = le_cuisine.fit_transform(df["Cuisine"])
df["Price_Range"] = le_price.fit_transform(df["Price_Range"])

train_df["Cuisine"] = le_cuisine.transform(train_df["Cuisine"])
train_df["Price_Range"] = le_price.transform(train_df["Price_Range"])

predict_df["Cuisine"] = le_cuisine.transform(predict_df["Cuisine"])
predict_df["Price_Range"] = le_price.transform(predict_df["Price_Range"])

# Features and target

X_train = train_df[["Cuisine", "Price_Range", "Votes"]]
y_train = train_df["Rating"]

# Train model

model = LinearRegression()
model.fit(X_train, y_train)

# Ask user for restaurant

restaurant_name = input("Enter Restaurant Name: ")

restaurant = predict_df[
    predict_df["Restaurant"].str.lower() == restaurant_name.lower()
]

if not restaurant.empty:

    sample = restaurant[["Cuisine", "Price_Range", "Votes"]]

    prediction = model.predict(sample)

    print("\nPredicted Rating:", round(prediction[0], 2))

else:
    print("\nRestaurant not found OR already has rating.")