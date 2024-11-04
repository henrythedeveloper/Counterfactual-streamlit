# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

try:
    print("Current working directory:", os.getcwd())

    # Load the dataset
    data = pd.read_csv('social_media_usage.csv')

    print("Data loaded successfully.")

    # Check for missing values
    print("Missing values in each column:")
    print(data.isnull().sum())

    # Data types
    print("\nData types of each column:")
    print(data.dtypes)

    # Define the target variable 'High_Engagement' based on likes and follows
    likes_mean = data['Likes_Per_Day'].mean()
    follows_mean = data['Follows_Per_Day'].mean()

    # Create the target variable
    data['High_Engagement'] = np.where(
        (data['Likes_Per_Day'] >= likes_mean) & (data['Follows_Per_Day'] >= follows_mean),
        1,
        0
    )

    # Select features and target
    features = ['Daily_Minutes_Spent', 'Posts_Per_Day', 'App']
    target = 'High_Engagement'

    X = data[features]
    y = data[target]

    # Encode the 'App' column
    le_app = LabelEncoder()
    X['App'] = le_app.fit_transform(X['App'])

    # Save the label encoder
    joblib.dump(le_app, 'le_app.pkl')

    # Optionally, save the processed data
    processed_data = pd.concat([X, y], axis=1)
    processed_data.to_csv('processed_data.csv', index=False)

    print("\nData preprocessing completed. Processed data saved to 'processed_data.csv'.")

except FileNotFoundError as fnf_error:
    print("File not found error:")
    print(fnf_error)
except Exception as e:
    print("An unexpected error occurred:")
    print(e)
