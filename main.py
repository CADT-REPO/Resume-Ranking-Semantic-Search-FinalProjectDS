import streamlit as st

import streamlit as st
# from openai import OpenAI


# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

import streamlit as st
import pdfplumber
import os
import pandas as pd
import uuid as uuid_lib
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords from NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Define a function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    # text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Rejoin tokens into a cleaned string
    return ' '.join(tokens)

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Function to get the next available UUID (as a string) for each PDF row
def generate_uuid():
    return str(uuid_lib.uuid4())

# Function to load the S-BERT model and calculate the similarity score
def calculate_similarity_score(resume_text, job_description_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use the pre-trained model
    # Encode the texts
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_desc_embedding = model.encode(job_description_text, convert_to_tensor=True)
    # Calculate cosine similarity
    similarity_score = util.cos_sim(resume_embedding, job_desc_embedding)[0][0].item()
    return similarity_score

# Streamlit app
st.title("PDF Text Extraction and Job Description Input")

# Define the CSV file path to store the data
csv_file = "extracted_dataset.csv"

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Prompt for the job description that will be applied to all PDFs
job_description_text = st.text_area("Enter the job description for all PDFs", height=150)

if uploaded_files and job_description_text:
    all_rows = []
    # Preprocess the job description text
    preprocessed_job_desc = preprocess_text(job_description_text)

    for uploaded_file in uploaded_files:
        st.write(f"Processing {uploaded_file.name}...")

        # Extract text from the uploaded PDF using pdfplumber
        resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text:
            # Preprocess the resume text
            preprocessed_resume_text = preprocess_text(resume_text)
            
            # Generate a unique UUID for each file's row
            file_uuid = generate_uuid()

            # Calculate similarity score using S-BERT
            similarity_score = calculate_similarity_score(preprocessed_resume_text, preprocessed_job_desc)

            # Add the row to the list with the extracted resume text, job description, and similarity score
            all_rows.append({
                "uuid": file_uuid,
                "resume_text": resume_text,
                "job_description_text": job_description_text,
                "similarity_score": similarity_score
            })

            st.success(f"Text extracted from {uploaded_file.name}.")
        else:
            st.warning(f"No text could be extracted from {uploaded_file.name}.")
    
    # If rows are collected, append to the CSV
    if all_rows:
        new_df = pd.DataFrame(all_rows)

        # Check if the CSV file exists, and append data accordingly
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(csv_file, index=False)
        else:
            new_df.to_csv(csv_file, index=False)

        st.success(f"Data saved to {csv_file}.")
        
        # Display the updated CSV content
        st.write("Updated CSV data:")
        st.dataframe(pd.read_csv(csv_file))
else:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    if not job_description_text:
        st.warning("Please enter the job description text.")


# Example Streamlit app
st.title("Score Prediction App")

# Inputs
skill_sim = st.number_input("Skill Similarity", value=0.0)
edu_sim = st.number_input("Education Similarity", value=0.0)
experience_sim = st.number_input("Experience Similarity", value=0.0)

# Calculate final score
if st.button("Predict"):
    skill_weight = 0.4
    edu_weight = 0.2
    experience_weight = 0.4
    final_score = (skill_sim * skill_weight) + (edu_sim * edu_weight) + (experience_sim * experience_weight)
    st.write(f"Final Score: {final_score}")
