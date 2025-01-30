import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

# AWS S3 Configuration
BUCKET_NAME = "osas"  # S3 Bucket Name
LOCAL_PATH = "tinybert-sentiment-analysis"  # Local directory to store the downloaded model
S3_PREFIX = "ml-models/tinybert-sentiment-analysis/"  # Path to the model in S3


def download_dir(local_path, s3_prefix):
    """Downloads all files from an S3 directory to a local directory."""
    s3 = boto3.client('s3')
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    
    for result in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                
                # Download file from S3 to local path
                s3.download_file(BUCKET_NAME, s3_key, local_file)

# Streamlit UI Title
st.title("Machine Learning Model Deployment at the Server")

# Button to download the model
if st.button("Download Model"):
    with st.spinner("Downloading... Please wait!"):
        download_dir(LOCAL_PATH, S3_PREFIX)
        st.success("Model Downloaded Successfully!")

# Text input for user review
text = st.text_area("Enter Your Review")

# Button to trigger prediction
if st.button("Predict"):
    if text.strip():  # Ensure input is not empty
        with st.spinner("Predicting..."):
            # Set device to GPU if available, otherwise use CPU
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            classifier = pipeline('text-classification', model=LOCAL_PATH, device=device)
            
            # Perform sentiment prediction
            output = classifier(text)
            
            # Display result
            st.write(output)
    else:
        st.warning("Please enter text for prediction.")
