# Importing libraries
import pickle
import streamlit as st
import pandas as pd
from script import featureExtract

if __name__ == "__main__":
    # Load model
    url_model = pickle.load(open('URLModel.pkl', 'rb'))

    # Homepage layout
    # Setting page config
    st.set_page_config(
        page_title = "Malicious URL Checker",
        layout = "wide"
    )

    # Heading container
    with st.container():
        st.title("Malicious URL Detector")

    st.divider()

    # Taking input for the URL
    with st.container():
        st.markdown("Please enter the URL to check if it is malicious or not below")
        inputURL = st.text_input(label="Enter URL")

    if st.button(label="Check"):

        # Converting the data into pandas dataframe
        # as desired input for the model
        inputDataDict = {"url" : [str(inputURL)]}
        inputDataPandas = pd.DataFrame.from_dict(inputDataDict)

        # Extract features from data
        featureExtractedData = featureExtract(inputDataPandas)
        featureExtractedData = featureExtractedData.iloc[:, 1:]
        predictedValue = url_model.predict(featureExtractedData)

        # Printing result
        if predictedValue[0] == 0:
            st.success("The given URL is NOT MALICIOUS!")
        else:
            st.error("The given URL is MALICIOUS!") 