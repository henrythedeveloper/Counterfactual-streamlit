# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import dice_ml
from dice_ml import Dice

# Load the trained model and label encoder
model = joblib.load('model.pkl')
le_app = joblib.load('le_app.pkl')

# Prepare data for DiCE
data = pd.read_csv('processed_data.csv')

# Define continuous and categorical features
continuous_features = ['Daily_Minutes_Spent', 'Posts_Per_Day']
categorical_features = ['App']
outcome_name = 'High_Engagement'

dice_data = dice_ml.Data(
    dataframe=data,
    continuous_features=continuous_features,
    outcome_name=outcome_name
)

# Wrap the model
model_wrapper = dice_ml.Model(model=model, backend='sklearn')

# Initialize DiCE
dice_exp = Dice(dice_data, model_wrapper, method='random')

# Streamlit App
st.title('Social Media Engagement Predictor')

st.header('Input Your Social Media Usage')

# User Inputs
daily_minutes = st.slider('Daily Minutes Spent', min_value=5, max_value=500, value=60)
posts_per_day = st.slider('Posts Per Day', min_value=0, max_value=20, value=1)
app_options = le_app.classes_
app = st.selectbox('Social Media Platform', app_options)

# Encode 'App' input
app_encoded = le_app.transform([app])[0]

# Create input DataFrame
input_data = pd.DataFrame({
    'Daily_Minutes_Spent': [daily_minutes],
    'Posts_Per_Day': [posts_per_day],
    'App': [app_encoded]
})

# Predict engagement
prediction = model.predict(input_data)[0]
prediction_label = 'High Engagement' if prediction == 1 else 'Low Engagement'

st.write(f'**Predicted Engagement Level:** {prediction_label}')

# Desired Outcome
desired_outcome = st.selectbox('Desired Engagement Level', ['High Engagement', 'Low Engagement'])
desired_class = 1 if desired_outcome == 'High Engagement' else 0

# Generate Counterfactual Explanations
if st.button('Generate Counterfactual Explanations'):
    with st.spinner('Generating explanations...'):
        exp = dice_exp.generate_counterfactuals(
            input_data,
            total_CFs=3,
            desired_class=desired_class,
            features_to_vary=['Daily_Minutes_Spent', 'Posts_Per_Day']
        )
    st.success('Counterfactual explanations generated!')

    # Display the explanations
    cf_df = exp.cf_examples_list[0].final_cfs_df

    # Decode 'App' for display
    cf_df['App'] = le_app.inverse_transform(cf_df['App'].astype(int))

    st.header('Counterfactual Explanations')
    st.write(cf_df)
