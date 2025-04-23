import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import pymysql
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Function to load data from MySQL (optional, for reference or data display)
def load_data_from_mysql():
    try:
        conn = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='Dob@1980',  # Replace with your MySQL password
            port=3306,
            database='tumor'
        )
        query = "SELECT * FROM titanic"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except pymysql.Error as e:
        st.error(f"MySQL Error: {e}")
        return None

# Load the trained XGBoost model
model = joblib.load('xg_boost_model.pkl')

# Streamlit app
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival probability using the XGBoost model.")

# Input fields for features
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Preprocess input data
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

# Fit encoders (using sample data to mimic notebook behavior)
sample_data = pd.DataFrame({
    'Sex': ['male', 'female'],
    'Embarked': ['C', 'Q', 'S']
})
le_sex.fit(sample_data['Sex'])
le_embarked.fit(sample_data['Embarked'])

# Transform inputs
sex_encoded = le_sex.transform([sex])[0]
embarked_encoded = le_embarked.transform([embarked])[0]

# Create input DataFrame for prediction
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})

# Predict survival
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of survival

    if prediction == 1:
        st.success(f"The passenger is predicted to **SURVIVE** with a probability of {probability:.2f}.")
    else:
        st.error(f"The passenger is predicted to **NOT SURVIVE** with a probability of survival {probability:.2f}.")

# Optional: Display sample data from MySQL
if st.checkbox("Show Sample Titanic Data"):
    df = load_data_from_mysql()
    if df is not None:
        st.write("Sample Data from Titanic Dataset:")
        st.dataframe(df.head())

# Footer
st.markdown("---")
st.write("Built with Streamlit and XGBoost. Model trained on Titanic dataset.")