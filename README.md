# Social Media Engagement Predictor with Counterfactual Explanations

This application uses machine learning to predict user engagement levels on social media platforms and provides counterfactual explanations to help users understand how they can achieve their desired engagement levels.

## Overview
This project leverages a machine learning model to predict whether a user will have **High Engagement** or **Low Engagement** on social media platforms based on their usage patterns. It integrates the **DiCE (Diverse Counterfactual Explanations)** library to provide actionable suggestions (counterfactual explanations) that guide users on how to adjust their behavior to achieve their desired engagement level.

## Features
- **Predict Engagement Level**: Input your social media usage data to receive a prediction of your engagement level.
- **Counterfactual Explanations**: Get personalized suggestions on how to change your usage patterns to achieve your desired engagement level.
- **Interactive Interface**: User-friendly Streamlit app for seamless interaction.
- **Explainable AI**: Understand the impact of different features on your engagement level.

## Installation
### Prerequisites

- Python 3.7 or higher (Python 3.10 is recommended)
- pip package manager
### Clone the Repository
```
git clone https://github.com/henrythedeveloper/social-media-engagement-predictor.git
cd social-media-engagement-predictor
```

### Set Up a Virtual Environment (Recommended)
#### Windows:
```
python -m venv venv
venv\Scripts\activate
```
#### Unix or MacOS:
```
python3 -m venv venv
source venv/bin/activate
```
### Install Dependencies
```
pip install -r requirements.txt
```
## Usage
### Step 1: Data Preprocessing
Run the `data_preprocessing.py` script to load and preprocess the data.
```
python data_preprocessing.py
```
**What It Does**:

- Loads the `social_media_usage.csv` dataset.
- Creates a target variable `High_Engagement`.
- Encodes categorical variables.
- Saves the processed data to `processed_data.csv`.
- Encodes the `App` column from the data set to `le_app.pkl`

### Step 2: Model Training
Train the machine learning model using the `model_training.py` script.
```
python model_training.py
```
**What It Does**:

- Loads the processed data.
- Splits the data into training and testing sets.
- Trains a Random Forest Classifier.
- Evaluates the model's performance.
- Saves the trained model to `model.pkl`.