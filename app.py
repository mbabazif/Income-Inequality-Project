import streamlit as st
import pandas as pd
import pickle
import joblib
import os

# Load the model and encoder
SRC = os.path.abspath('/home/florence/Desktop/Capston-Project//SRC')

    # Construct full file paths
pipeline_path = os.path.join(SRC, 'rfc_pipeline.joblib')
encoder_path = os.path.join(SRC, 'encoder.joblib')

    # Load the model and encoder using joblib
model = joblib.load(pipeline_path)
encoder = joblib.load(encoder_path)

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ["Prediction", "Model Information", "Feedback"])


# Prediction Page
if options == "Prediction":
    st.title("Income Classification - Prediction")

    # Input fields
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ['Female', 'Male'])
    education = st.selectbox("Education", ['High School', 'Children', 'Middle School', 'Masters', 'Bachelors Degree',
                                           'Elementary', 'College Dropout', 'Associates Degree', 'Professional Degree',
                                           'Doctorate'])
    worker_class = st.selectbox("Worker Class", ['Private', 'Federal government', 'Never worked', 'Local government',
                                                 'Self-employed-incorporated', 'Self-employed-not incorporated',
                                                 'State government', 'Without pay'])
    marital_status = st.selectbox("Marital Status", ['Widowed', 'Never married', 'Married-civilian spouse present',
                                                     'Divorced', 'Married-spouse absent', 'Separated',
                                                     'Married-A F spouse present'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian or Pacific Islander', 'Amer Indian Aleut or Eskimo', 'Other'])
    is_hispanic = st.selectbox("Is Hispanic", ['All other', 'Mexican-American', 'Central or South American',
                                               'Mexican (Mexicano)', 'Puerto Rican', 'Other Spanish', 'Cuban',
                                               'Do not know', 'Chicano'])
    employment_commitment = st.selectbox("Employment Commitment", ['Not in labor force', 'Children or Armed Forces',
                                                                   'Full-time schedules', 'PT for econ reasons usually PT',
                                                                   'Unemployed full-time',
                                                                   'PT for non-econ reasons usually FT',
                                                                   'PT for econ reasons usually FT',
                                                                   'Unemployed part- time'])
    employment_stat = st.number_input("Employment Status", min_value=0, max_value=2, step=1)
    wage_per_hour = st.number_input("Wage per Hour", min_value=0)
    working_week_per_year = st.number_input("Working Week per Year", min_value=0)
    industry_code = st.number_input("Industry Code", min_value=0)
    industry_code_main = st.selectbox("Industry Code Main", ['Not in universe or children', 'Hospital services',
                                                             'Retail trade', 'Finance insurance and real estate',
                                                             'Manufacturing-nondurable goods', 'Transportation',
                                                             'Business and repair services', 'Medical except hospital',
                                                             'Education', 'Construction', 'Manufacturing-durable goods',
                                                             'Public administration', 'Agriculture',
                                                             'Other professional services', 'Mining',
                                                             'Utilities and sanitary services', 'Private household services',
                                                             'Personal services except private HH', 'Wholesale trade',
                                                             'Communications', 'Entertainment', 'Social services',
                                                             'Forestry and fisheries', 'Armed Forces'])
    occupation_code = st.number_input("Occupation Code", min_value=0)
    occupation_code_main = st.selectbox("Occupation Code Main", ['Unknown', 'Adm support including clerical',
                                                                 'Executive admin and managerial', 'Sales',
                                                                 'Machine operators assmblrs & inspctrs', 'Other service',
                                                                 'Precision production craft & repair',
                                                                 'Professional specialty', 'Handlers equip cleaners etc',
                                                                 'Transportation and material moving',
                                                                 'Farming forestry and fishing', 'Private household services',
                                                                 'Technicians and related support', 'Protective services',
                                                                 'Armed Forces'])
    total_employed = st.selectbox("Total Employed", [0, 1, 2, 3, 4, 5, 6])
    household_summary = st.selectbox("Household Summary", ['Householder', 'Child 18 or older', 
                                                           'Child under 18 never married', 'Spouse of householder', 
                                                           'Nonrelative of householder', 'Other relative of householder', 
                                                           'Group Quarters- Secondary individual', 'Child under 18 ever married'])
    vet_benefit = st.number_input("Vet Benefit", min_value=0, max_value=2, step=1)
    tax_status = st.selectbox("Tax Status", ['Head of household', 'Single', 'Nonfiler', 'Joint both 65+',
                                             'Joint both under 65', 'Joint one under 65 & one 65+'])
    gains = st.number_input("Gains", min_value=0)
    losses = st.number_input("Losses", min_value=0)
    stocks_status = st.number_input("Stocks Status", min_value=0)
    citizenship = st.selectbox("Citizenship", ['citizen', 'foreigner'])
    importance_of_record = st.number_input("Importance of Record", min_value=0.0, format='%f')

    if st.button('Predict Income Level'):
        input_data = pd.DataFrame([[
            age, gender, education, worker_class, marital_status, race, is_hispanic, employment_commitment,
            employment_stat, wage_per_hour, working_week_per_year, industry_code, industry_code_main, occupation_code,
            occupation_code_main, total_employed, household_summary, vet_benefit, tax_status, gains, losses,
            stocks_status, citizenship, importance_of_record
        ]], columns=[
            'age', 'gender', 'education', 'worker_class', 'marital_status', 'race', 'is_hispanic', 
            'employment_commitment', 'employment_stat', 'wage_per_hour', 'working_week_per_year', 
            'industry_code', 'industry_code_main', 'occupation_code', 'occupation_code_main', 'total_employed', 
            'household_summary', 'vet_benefit', 'tax_status', 'gains', 'losses', 'stocks_status', 
            'citizenship', 'importance_of_record'
        ])

        # Predict and display results
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data).max(axis=1)[0]
        result = "Above Limit" if prediction[0] == 1 else "Below Limit"
        st.success(f'Income Level Prediction: {result}')
        st.info(f'Prediction Probability: {probability:.2f}')


# Model Information Page
elif options == "Model Information":
    st.title("Model Information")
    st.write("""
        ### Model Description
        In a world where understanding financial demographics is key to tailor services and opportunities, our model serves as a powerful tool to predict an individual's income level. This insight can be instrumental for businesses, policymakers, and researchers in making informed decisions.

        - **Model Type:** Random Forest Classifier
          - The Random Forest is a versatile and robust machine learning method that combines multiple decision trees to produce more accurate and stable predictions. It's known for its high accuracy, ability to handle large datasets with higher dimensionality, and its robustness to overfitting.

        - **Training Data:** 
          - Our model is trained on comprehensive census data, encompassing a wide range of features such as age, education, marital status, race, occupation, and more. This rich dataset ensures a nuanced understanding of the socio-economic factors influencing income levels.

        - **Accuracy:** 92%
          - With an accuracy of 92%, our model stands as a reliable predictor, demonstrating its effectiveness in understanding and categorizing income levels.

        - **What It Aims to Solve:**
          - **Economic Research:** Assists in socio-economic studies, understanding income distribution, and identifying key factors influencing income levels.
          - **Targeted Marketing:** Enables businesses to tailor their marketing strategies by understanding the income brackets of their potential customer base.
          - **Policy Making:** Aids policymakers in crafting targeted welfare schemes and tax brackets.
          - **Personalized Services:** Financial institutions can offer more personalized financial advice or services based on predicted income levels.

        - **Ethical Considerations:**
          - We are committed to ethical AI practices. We recognize the importance of privacy, fairness, and inclusivity in our model's application and strive to prevent biases.

    """)


# Feedback Page
elif options == "Feedback":
    st.title("Feedback")
    st.write("We value your feedback! Please let us know your thoughts about the model and interface.")
    feedback = st.text_area("Enter your feedback here")
    if st.button("Submit"):
        # Logic to store feedback, can be a simple file write or a database insert
        with open("feedback.txt", "a") as file:
            file.write(f"{feedback}\n")
        st.success("Thank you for your feedback!")
