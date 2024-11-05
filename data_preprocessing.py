import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("Current working directory:", os.getcwd())

# Load the dataset
data = pd.read_csv('social_media_usage.csv')
print("Data loaded successfully.")

# Check for missing values and data types
print("Missing values in each column:")
print(data.isnull().sum())

print("\nData types of each column:")
print(data.dtypes)

# Select features and target
features = ['Daily_Minutes_Spent', 'Posts_Per_Day', 'App', 'Likes_Per_Day', 'Follows_Per_Day']
target = 'High_Engagement'  # Target column

X = data[features].copy()
y = data[target]

# Encode the 'App' column
le_app = LabelEncoder()
X['App'] = le_app.fit_transform(X['App'])
joblib.dump(le_app, 'le_app.pkl')

# Save processed data
processed_data = pd.concat([X, y], axis=1)
processed_data.to_csv('processed_data.csv', index=False)

print("Data preprocessing completed. Processed data saved to 'processed_data.csv'.")
