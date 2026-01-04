import streamlit as st
import pandas as pd
import joblib

st.title("HR Analytics")

df = pd.read_csv('train_wns.csv')

# input fields

#st.selectbox
#st.number_input

st.write("App is running successfully!")

categorical_features = [
    "department", "region", "education",
    "gender", "recruitment_channel",
    "KPIs_met >80%", "awards_won?"
]

numerical_features = [
    "no_of_trainings", "age",
    "length_of_service", "avg_training_score",
    "previous_year_rating"
]

department = st.selectbox("department", pd.unique(df["department"]))
region = st.selectbox("region", pd.unique(df["region"]))
education = st.selectbox("education", pd.unique(df["education"]))
gender = st.selectbox("gender", pd.unique(df["gender"]))
recruitment_channel = st.selectbox("recruitment_channel", pd.unique(df["recruitment_channel"]))
KPIs_met_80 = st.selectbox("KPIs_met >80%", pd.unique(df["KPIs_met >80%"]))
awards_won = st.selectbox("awards_won?", pd.unique(df["awards_won?"]))
no_of_trainings = st.number_input("no_of_trainings")
age = st.number_input("age")
length_of_service = st.number_input("length_of_service")
avg_training_score = st.number_input("avg_training_score")
previous_year_rating = st.number_input("previous_year_rating")

inputs = {
     "department": department,
     "region": region,
     "education": education,
     "gender": gender,
     "recruitment_channel": recruitment_channel,
     "KPIs_met >80%": KPIs_met_80,
      "awards_won?": awards_won,
      "no_of_trainings": no_of_trainings,
      "age": age,
      "length_of_service": length_of_service,
      "avg_training_score": avg_training_score,
      "previous_year_rating": previous_year_rating
}

if st.button("Predict"):
  model = joblib.load('jobchg_pipeline_model.pkl')
  X_input = pd.DataFrame([inputs])
  prediction = model.predict(X_input)
  st.write(prediction)
