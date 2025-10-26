import pickle
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


model = tf.keras.models.load_model(f'model.h5')

with open(f'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'onehot_encoder.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)

with open(f'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

st.title('Customer Churn Prediction')

#User Inputs
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

gender = label_encoder.transform([gender])[0]

#Perpare input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#Encode categorical features
geo_encoded = geo_encoder.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

#Combine encoded features with input data
input_data = pd.concat([input_data.drop(['Geography'],axis=1), geo_encoded_df], axis=1)
scale_data = scaler.transform(input_data)

#Prediction
prediction = model.predict(scale_data)
churn_probability = prediction[0][0]
print("Churn Probability: ", scale_data,prediction)

if churn_probability > 0.5:
    churn_probability = churn_probability * 100
    st.write(f'The customer is likely to churn with a probability of: {churn_probability:.2f}%')
else:
    churn_probability = churn_probability * 100
    st.write(f'The customer is unlikely to churn with a probability of: {churn_probability:.2f}%')
