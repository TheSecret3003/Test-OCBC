__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.simplefilter(action='ignore', category=FutureWarning)
import streamlit as st
from streamlit_option_menu import option_menu
import json
import pandas as pd
import numpy as np
import joblib
from helper_function import generate_response


#loading the saved model of Bad User prediction
with open("Preprocessing File/user_model/scaler.pkl", 'rb') as f:
    user_scaler = joblib.load(f)
with open("Preprocessing File/user_model/transformer_age.pkl", 'rb') as f:
    transformer_age = joblib.load(f)
with open("Preprocessing File/user_model/transformer_friend.pkl", 'rb') as f:
    transformer_friend = joblib.load(f)
with open("Preprocessing File/user_model/transformer_salary.pkl", 'rb') as f:
    transformer_salary = joblib.load(f)
with open("Models/user_model/best_model_XGBoost.pkl", 'rb') as f:
    loaded_model_xgb = joblib.load(f)


def prediction_user(input_data):

    #loading data
    columns_input = ['log_ph_total_contacts','de_age','de_num_friends','de_children','de_employment_duration',
                   'de_monthly_salary','de_gender','de_employment_type','de_education','de_marital_status','is_same_gender',
                   'fb_gender_encoded','device_codename_encoded','brand_encoded']
    columns_to_used = ['log_ph_total_contacts','yeo-johnson_de_age','yeo-johnson_de_num_friends','de_children','de_employment_duration',
                   'yeo-johnson_de_monthly_salary','de_gender','de_employment_type','de_education','de_marital_status','is_same_gender',
                   'fb_gender_encoded','device_codename_encoded','brand_encoded']

    original_data = pd.read_csv("Datasets/dataset_final.csv", low_memory=False)

    input_df = pd.DataFrame([input_data], columns=columns_input)

    input_df['log_ph_total_contacts'] =  np.log1p(input_df['log_ph_total_contacts'])
    input_df['yeo-johnson_de_monthly_salary'] = transformer_salary.transform(input_df[['de_monthly_salary']])
    input_df['yeo-johnson_de_age'] = transformer_age.transform(input_df[['de_age']])
    input_df['yeo-johnson_de_num_friends'] = transformer_friend.transform(input_df[['de_num_friends']])

    columns_to_scale = ['log_ph_total_contacts','yeo-johnson_de_age','yeo-johnson_de_num_friends','de_children','de_employment_duration','yeo-johnson_de_monthly_salary']
    input_df[columns_to_scale] = user_scaler.transform(input_df[columns_to_scale])

    
    device_freq = original_data['device_codename'].value_counts(normalize=True)
    device_name = input_df['device_codename_encoded'].iloc[0]
    if device_name not in list(device_freq.index):
        input_df.loc[0,'device_codename_encoded'] = device_freq.values[0]
    else:
        input_df['device_codename_encoded'] = input_df['device_codename_encoded'].map(device_freq)

    brand_freq = original_data['brand'].value_counts(normalize=True)
    brand_name = input_df['brand_encoded'].iloc[0]
    if brand_name not in list(brand_freq.index):
        input_df.loc[0,'brand_encoded'] = brand_freq.values[0]
    else:
        input_df['brand_encoded'] = input_df['brand_encoded'].map(brand_freq)

    input_df['brand_encoded'] = input_df['brand_encoded'].astype(float)
    input_df['device_codename_encoded'] = input_df['device_codename_encoded'].astype(float)

     
    #predictions
    prediction = loaded_model_xgb.predict(input_df[columns_to_used])
  
    return prediction

def main():
    # sidebar for navigate

    with st.sidebar:
    
        selected = option_menu('User Classification and Financial Chatbot',
                           
                            ['Bad User Prediction',
                            'Financial Chatbot'],
                           
                           icons = ['person-circle','bank'],
                           
                           default_index = 0)

    # Financial Chatbot Page
    if( selected == 'Financial Chatbot'):
        
        st.title("Financial Statement Chatbot 📊💰")

        query = st.text_input("🔍 Ask about the financial statements:")
        if query:
            answer, sources = generate_response(query)
            st.subheader("💡 Answer:")
            st.write(answer)
            
            st.subheader("📖 Sources:")
            if sources:  # Ensure sources exist
                for i, src in enumerate(sources):
                    source_text = src.get("text", "No text available")[:300]  # Preview
                    source_name = src.get("source", "Unknown Source")  # Handle missing source
                    
                    st.write(f"**Source {i+1} (from {source_name}):** {source_text}...")
            else:
                st.write("No sources found for this query.")
    
     # Bad User Prediction Page
    if( selected == 'Bad User Prediction'):
         
        #giving a title
        st.title('Media Sosial User Prediction')
        
        #getting input data from user

        col1 , col2 = st.columns(2)
        
        with col1:
            option1 = st.selectbox('Education',('Elementary School', 'Senior High School','Diploma','Undergraduate','Postgraduate'))
            if option1 == 'Elementary School':
                education = 1
            elif option1 == 'Senior High School':
                education = 2
            elif option1 == 'Diploma':
                education = 3
            elif option1 == 'Undergraduate':
                education = 4
            else:
                education = 5

        with col2:
            option2 = st.selectbox('Employment Type',('Full-Time', 'Part-Time','Business Owner'))
            if option2 == 'Full-Time':
                employment = 1
            elif option2 == 'Part-Time':
                employment = 2
            elif option2 == 'Business Owner':
                employment = 3

        with col1:
            option3 = st.selectbox('Facebook Gender',('Male', 'Female'))
            if option3 == 'Male':
                fbgender = 1
            elif option3 == 'Female':
                fbgender = 2
        
        with col2:
            option4 = st.selectbox('Gender',('Male', 'Female'))
            if option4 == 'Male':
                gender = 1
            elif option4 == 'Female':
                gender = 2

        with col1:
            option5 = st.selectbox('Marital Status',('Single','Married','Divorced','Widow'))
            if option5 == 'Single':
                marital = 1
            elif option5 == 'Married':
                marital = 2
            elif option5 == 'Divorced':
                marital = 3
            elif option5 == 'Widow':
                marital = 4
        
        with col2:
            brand = st.text_input("Enter Phone Brand Here")
            brand = brand.lower()
        
        with col1:
            device = st.text_input("Enter Phone device Here")
            device = device.lower()
        
               
        with col2:
           age = st.number_input("Age of the person",min_value=0, max_value=100, value=10, step=1, format="%d")

        with col1:
           salary = st.number_input("Salary",min_value=0, step=100000, format="%d") 
        
        with col2:
           children = st.number_input("Children",min_value=0, step=1, format="%d")

        with col1:
           duration = st.number_input("Employment Duration",min_value=0, step=1, format="%d")
        
        with col2:
           friend = st.number_input("Number of Friends",min_value=1, step=10, format="%d")

        with col1:
           contacts = st.number_input("Number of Contatcs",min_value=1, step=1, format="%d")
        
        
        # code for prediction
        if gender == fbgender:
            same_gender = 0
        else:
            same_gender = 1

        user_prediction = ''
        
        
        user_prediction = prediction_user([contacts,age,friend,children,duration,salary,gender,employment,
                                            education,marital,same_gender,fbgender,device,brand])
        
        
        #creating a button for Prediction
        if st.button("Predict User"):
            if(user_prediction[0]==0):
                prediction = 'Good User' 
            else:
                prediction = 'Bad User'
            st.write(f"Prediction: {prediction}")
    
if __name__ == '__main__':
    main()





