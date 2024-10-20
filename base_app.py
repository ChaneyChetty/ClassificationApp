"""
    Simple Streamlit webserver application for serving developed classification
    models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Load the vectorizer and model
vectorizer = joblib.load(open("vectorizer.pkl", "rb"))
model = joblib.load(open("best_logistic_regression_model.pkl", "rb"))

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analysing news articles")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = vectorizer.transform([news_text]).toarray()
            # Make predictions
            prediction = model.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction[0]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
