import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import base64

# base 64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode()
    return base64_string

# Load your dataset
dataset = pd.read_csv('D:/Customer Churn Prediction App/customer_data.csv')

# Remove unnecessary columns
dataset = dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Perform one-hot encoding if needed
dataset = pd.get_dummies(data=dataset, drop_first=True)

# Separate features and target variable
X = dataset.drop(columns='Exited')
y = dataset['Exited']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Function for churn prediction
def churn_prediction(input_data):
    # Standardize the input data
    input_reshaped = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_reshaped)
    prediction = clf.predict(std_data)

    if prediction[0] == 0:
        return "Customer will not churn"
    else:
        return "Customer will churn"

def main():
    
    base64_image = get_base64_image('D:/Customer Churn Prediction App/back2.jpg')

    # Set background image using CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Customer Churn Prediction App")
    st.subheader("Enter Required Details of Customer")
    # Input fields
    CreditScore = st.text_input("Enter Credit Score")
    Age = st.text_input("Enter Age")
    Tenure = st.text_input("Enter Tenure")
    Balance = st.text_input("Enter Balance")
    NumOfProducts = st.text_input("Enter No. of Products")
    HasCrCard = st.text_input("Have you credit card or not (0/1)")
    IsActiveMember = st.text_input("Is you Active member (0/1)")
    EstimatedSalary = st.text_input("Enter Estimated Salary")
    Geography_Germany = st.text_input("Is the Geography Germany (0/1)")
    Geography_Spain = st.text_input("Is the Geography Spain (0/1)")
    Gender_Male = st.text_input("Is the Gender Male (0/1)")

    # Ensure all inputs are valid numeric values
    try:
        input_data = [
            float(CreditScore),
            float(Age),
            float(Tenure),
            float(Balance),
            float(NumOfProducts),
            float(HasCrCard),
            float(IsActiveMember),
            float(EstimatedSalary),
            float(Geography_Germany),
            float(Geography_Spain),
            float(Gender_Male)
        ]
    except ValueError:
        st.error("Please enter valid numeric values.")
        return
    st.write("Check whether customer will churn or not")
    # Make prediction
    diagnosis = ""
    if st.button("Predict Churn"):
        diagnosis = churn_prediction(input_data)
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
