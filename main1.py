import streamlit as st
import pdfplumber
import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import uuid as uuid_lib
import re

# Download stopwords from NLTK
# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Define a function to preprocess text
def preprocess_text(text):
    # # Convert text to lowercase
    # text = text.lower()
    # # Remove punctuation
    # text = text.translate(str.maketrans('', '', string.punctuation))
    # # Tokenize text
    # # tokens = word_tokenize(text)
    # # Remove stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word not in stop_words]
    # return tokens
    """
    Cleans text data by removing unwanted characters, normalizing whitespace,
    and handling case sensitivity.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """

    # Remove HTML tags (if any)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Function to get the next available ID (formatted as ID00001, ID00002, etc.)
def generate_id(existing_ids):
    # Generate a new ID based on existing IDs
    if not existing_ids:
        return "ID00001"  # Start from the first ID if no existing IDs
    # Extract the numeric part of the last ID and increment
    last_id = max(existing_ids, key=lambda x: int(x[2:]))
    next_id_num = int(last_id[2:]) + 1
    return f"ID{next_id_num:05d}"

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
st.title("PDF Text Extraction, Job Description Input, and Similarity Calculation")

# Define the CSV file path to store the data
csv_file = "extracted_data_with_similarity.csv"

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Prompt for the job description that will be applied to all PDFs
job_description_text = st.text_area("Enter the job description for all PDFs", height=150)

# Add a submit button to trigger the processing when clicked
submit_button = st.button("Submit")


if submit_button and uploaded_files and job_description_text:
    all_rows = []
    # Preprocess the job description text
    preprocessed_job_desc = preprocess_text(job_description_text)

    # Read existing data from CSV if it exists
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        existing_ids = existing_df['uuid'].tolist()  # Collect existing IDs
    else:
        existing_ids = []  # No existing IDs if the file doesn't exist

    for uploaded_file in uploaded_files:
        st.write(f"Processing {uploaded_file.name}...")

        # Extract text from the uploaded PDF using pdfplumber
        resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text:
            # Preprocess the resume text
            preprocessed_resume_text = preprocess_text(resume_text)

            # Generate a sequential ID for this resume
            file_id = generate_id(existing_ids)
            existing_ids.append(file_id)  # Update the list of existing IDs

            # Calculate similarity score using S-BERT
            similarity_score = calculate_similarity_score(preprocessed_resume_text, preprocessed_job_desc)

            # Add the row to the list with the extracted resume text, job description, and similarity score
            all_rows.append({
                "uuid": file_id,
                "resume_text": preprocessed_resume_text,
                "job_description_text": preprocessed_job_desc,
                "similarity_score": similarity_score
            })

            st.success(f"Text extracted from {uploaded_file.name}. Similarity score: {similarity_score:.4f}")
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
