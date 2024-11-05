import streamlit as st
import pandas as pd
import numpy as np
import joblib
import dice_ml
from dice_ml import Dice
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and label encoder
model = joblib.load('model.pkl')
le_app = joblib.load('le_app.pkl')

# Prepare data for DiCE
data = pd.read_csv('processed_data.csv')

# Define continuous and categorical features
continuous_features = ['Daily_Minutes_Spent', 'Posts_Per_Day', 'Likes_Per_Day', 'Follows_Per_Day']
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

# Get feature importances from the model 
importances = model.feature_importances_
feature_names = ['Daily Minutes Spent', 'Posts Per Day', 'App', 'Likes Per Day', 'Follows Per Day']
feat_importances = pd.Series(importances, index=feature_names)


# Streamlit App
st.title('Social Media Engagement Predictor')

# Display feature importances
st.subheader('Feature Importances')
fig, ax = plt.subplots()
sns.barplot(x=feat_importances, y=feat_importances.index, ax=ax)
st.pyplot(fig)

st.header('Input Your Social Media Usage')

# User Inputs
app_options = le_app.classes_
app = st.selectbox('Social Media Platform', app_options)
daily_minutes = st.slider('Daily Minutes Spent', min_value=5, max_value=500, value=60, help='Select the number of minutes spent on social media daily')
posts_per_day = st.slider('Posts Per Day', min_value=0, max_value=20, value=1, help='Select the number of posts made per day')
likes_per_day = st.slider('Likes Per Day', min_value=0, max_value=1000, value=50, help='Select the number of likes received per day')
follows_per_day = st.slider('Follows Per Day', min_value=0, max_value=1000, value=10, help='Select the number of follows received per day')

# Encode 'App' input
app_encoded = le_app.transform([app])[0]

# Create input DataFrame
input_data = pd.DataFrame({
    'Daily_Minutes_Spent': [daily_minutes],
    'Posts_Per_Day': [posts_per_day],
    'App': [app_encoded],
    'Likes_Per_Day': [likes_per_day],
    'Follows_Per_Day': [follows_per_day]
})

# Predict engagement
prediction = model.predict(input_data)[0]
prediction_label = 'High Engagement' if prediction == 1 else 'Low Engagement'

st.write(f'**Predicted Engagement Level:** {prediction_label}')

# Desired Outcome
desired_outcome = st.selectbox('Desired Engagement Level', ['High Engagement', 'Low Engagement'])
desired_class = 1 if desired_outcome == 'High Engagement' else 0

# Function to generate explanations
def generate_explanation(original_input, cf_instance):
    # Extract original values
    original_minutes = original_input['Daily_Minutes_Spent']
    original_posts = original_input['Posts_Per_Day']
    original_app = le_app.inverse_transform([original_input['App']])[0]

    # Extract counterfactual values
    cf_minutes = cf_instance['Daily_Minutes_Spent']
    cf_posts = cf_instance['Posts_Per_Day']
    cf_app = cf_instance['App']

    # Initialize the explanation
    explanation = ""

    # Compare 'Daily_Minutes_Spent'
    if cf_minutes != original_minutes:
        if cf_minutes > original_minutes:
            explanation += f"- **Increase your daily time spent** from {original_minutes} minutes to {cf_minutes} minutes.\n"
        else:
            explanation += f"- **Decrease your daily time spent** from {original_minutes} minutes to {cf_minutes} minutes.\n"
    else:
        explanation += f"- **Keep your daily time spent** at {original_minutes} minutes.\n"

    # Compare 'Posts_Per_Day'
    if cf_posts != original_posts:
        if cf_posts > original_posts:
            explanation += f"- **Increase your posts per day** from {original_posts} to {cf_posts}.\n"
        else:
            explanation += f"- **Decrease your posts per day** from {original_posts} to {cf_posts}.\n"
    else:
        explanation += f"- **Keep your posts per day** at {original_posts}.\n"

    # Compare 'App' (if 'App' is allowed to vary)
    if cf_app != original_app:
        explanation += f"- **Change your social media platform** from {original_app} to {cf_app}.\n"
    else:
        explanation += f"- **Continue using** {original_app}.\n"

    # Add the predicted outcome
    desired_engagement = 'High Engagement' if cf_instance['High_Engagement'] == 1 else 'Low Engagement'
    explanation += f"\nBy making these changes, you could achieve **{desired_engagement}**."

    return explanation

# Generate Counterfactual Explanations
if st.button('Generate Counterfactual Explanations'):
    with st.spinner('Generating explanations...'):
        exp = dice_exp.generate_counterfactuals(
            input_data,
            total_CFs=3,
            desired_class=desired_class,
            features_to_vary=['Daily_Minutes_Spent', 'Posts_Per_Day', 'Likes_Per_Day', 'Follows_Per_Day']
        )
    st.success('Counterfactual explanations generated!')

    # Retrieve the counterfactual examples
    cf_df = exp.cf_examples_list[0].final_cfs_df

    # Decode 'App' for display
    cf_df['App'] = le_app.inverse_transform(cf_df['App'].astype(int))

    st.header('Counterfactual Explanations')

    # Display the counterfactual table
    st.subheader('Counterfactual Examples')
    st.write(cf_df)

    # Explain each counterfactual instance
    st.subheader('Detailed Explanations')
    for index, row in cf_df.iterrows():
        st.markdown(f"**Option {index + 1}:**")
        explanation = generate_explanation(input_data.iloc[0], row)
        st.write(explanation)
