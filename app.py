import streamlit as st
import pandas as pd
from joblib import load

# load the pretarined model
model = load('pipeline.pkl')
my_feature_dict = load('my_feature_dict.pkl')

# function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction
st.title('Employee Churn Prediction App')
st.subheader('Created by Hina Moin')
st.subheader('CATEGORICAL Features')
education = st.selectbox('Education', ['Bachelors', 'Masters', 'PHD'])
city = st.selectbox('City', ['Bangalore', 'New Delhi', 'Pune'])
gender = st.selectbox('Gender', ['Female', 'Male'])
everbenched = st.selectbox('EverBenched', ['No', 'Yes'])
st.subheader('Numerical Features')
joining_year = st.selectbox('Joining Year', [2012, 2013, 2014, 2015, 2016, 2017, 2018])
payment_tier = st.selectbox('Payment Tier', [1, 2, 3])
age = st.number_input('Age', min_value=22, max_value=41, value=25, step=1)
experience_in_current_domain = st.number_input('Experience In Current Domain', min_value=0, max_value=10, step=1)

input_data = pd.DataFrame([{
    'Education': education,
    'City': city,
    'Gender': gender,
    'EverBenched': everbenched,
    'JoiningYear': joining_year,
    'PaymentTier': payment_tier,
    'Age': age,
    'ExperienceInCurrentDomain': experience_in_current_domain,
}])

if st.button('Predict Churn'):
    prediction = predict_churn(input_data)
    if prediction[0] == 0:
        st.success("Prediction: No (Employee is likely to STAY)")
else:
        st.error("Prediction: Yes Employee is likely to CHURN)")