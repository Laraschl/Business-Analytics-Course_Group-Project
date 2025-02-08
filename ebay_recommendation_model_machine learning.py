
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Simulated eBay data: User-Product Ratings
data = {
    "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    "product_id": [101, 102, 103, 101, 104, 102, 103, 105, 101, 105],
    "rating": [5, 4, 3, 5, 4, 4, 2, 3, 4, 5]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Feature engineering: Convert user_id and product_id into categorical values
df["user_id"] = df["user_id"].astype("category").cat.codes
df["product_id"] = df["product_id"].astype("category").cat.codes

# Define features (X) and target (y)
X = df[["user_id", "product_id"]]
y = df["rating"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict ratings on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Show results
print("Root Mean Squared Error (RMSE):", rmse)

# Predict ratings for a specific user (e.g., user_id=1)
user_id = 1
user_products = df[df["user_id"] == user_id]["product_id"].unique()
all_products = df["product_id"].unique()
unrated_products = [p for p in all_products if p not in user_products]

# Create input data for unrated products
input_data = pd.DataFrame({"user_id": [user_id] * len(unrated_products), "product_id": unrated_products})
input_data["user_id"] = input_data["user_id"].astype("category").cat.codes
input_data["product_id"] = input_data["product_id"].astype("category").cat.codes

# Predict ratings for unrated products
predicted_ratings = model.predict(input_data)
recommended_products = sorted(zip(unrated_products, predicted_ratings), key=lambda x: x[1], reverse=True)

# Display recommended products
print("\nRecommended Products for User 1:")
for product, rating in recommended_products:
    print(f"Product ID: {product}, Predicted Rating: {rating:.2f}")
