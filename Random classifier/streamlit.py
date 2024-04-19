# Python code
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
@st.cache
def load_data():
    df = pd.read_csv("E:\360 digi tmg\Project raw material\data (2).csv")
    return df

# Load your model
@st.cache(allow_output_mutation=True)
def load_model():
    model = RandomForestClassifier()  # replace with your model
    model.load('stl_model.joblib')  # replace with your model file
    return model

def predict(model, input_df):
    predictions = model.predict(input_df)
    return predictions

def main():
    st.title("Raw Materials and Minerals Prediction App")
    
    df = load_data()
    model = load_model()

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df)
    
    st.subheader('Make predictions')
    input_features = st.text_input("Enter your features here")  # replace with your input method
    input_df = pd.DataFrame([input_features])  # replace with your preprocessing steps
    
    if st.button('Predict'):
        output = predict(model, input_df)
        st.write('Prediction: ', output)

if __name__ == '__main__':
    main()
