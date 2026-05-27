import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("restaurants.csv")

# Combine features
df["Features"] = df["Cuisine"] + " " + df["Price_Range"]

# Convert text into vectors
cv = CountVectorizer()
matrix = cv.fit_transform(df["Features"])

# Calculate similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(cuisine, price):

    user_input = cuisine + " " + price

    user_vector = cv.transform([user_input])

    scores = cosine_similarity(user_vector, matrix)

    scores = scores.flatten()

    recommended_indices = scores.argsort()[::-1]

    print("\nRecommended Restaurants:\n")

    for i in recommended_indices[:5]:
        print(
            df.iloc[i]["Restaurant"],
            "-",
            df.iloc[i]["Cuisine"],
            "-",
            df.iloc[i]["Price_Range"]
        )

# Sample user preference
recommend("Fast Food", "Low")