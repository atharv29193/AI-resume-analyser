import streamlit as st
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF
def input_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to get Gemini response
def get_gemini_response(input_prompt, text, jd):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_prompt.format(text=text, jd=jd))
    return response.text

# Streamlit app
st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")
st.caption("Aim of this project is to check whether a candidate is qualified for a role based on their education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")

# Uploaders
uploadedJD = st.file_uploader("Upload Job Description", type="pdf")
uploadedResume = st.file_uploader("Upload Resume", type="pdf")

# Buttons for different functionalities
if st.button("Analyze Resume Match"):
    if uploadedJD is not None and uploadedResume is not None:
        # Extract Job Description
        try:
            job_description = input_pdf_text(uploadedJD)
        except Exception as e:
            st.error(f"Error extracting job description text: {e}")
            job_description = None

        # Extract Resume
        try:
            resume = input_pdf_text(uploadedResume)
        except Exception as e:
            st.error(f"Error extracting resume text: {e}")
            resume = None

        # Calculate Match Percentage
        if job_description and resume:
            def getResult(JD_txt, resume_txt):
                content = [JD_txt, resume_txt]
                cv = CountVectorizer()
                matrix = cv.fit_transform(content)
                similarity_matrix = cosine_similarity(matrix)
                match = similarity_matrix[0][1] * 100
                return match

            match = getResult(job_description, resume)
            match = round(match, 2)
            st.write("Match Percentage: ", match, "%")
        else:
            st.warning("Please ensure both files are valid.")

if st.button("Analyze Resume with Gemini"):
    if uploadedJD is not None and uploadedResume is not None:
        # Extract Job Description and Resume
        try:
            job_description = input_pdf_text(uploadedJD)
        except Exception as e:
            st.error(f"Error extracting job description text: {e}")
            job_description = None

        try:
            resume = input_pdf_text(uploadedResume)
        except Exception as e:
            st.error(f"Error extracting resume text: {e}")
            resume = None

        # Get Gemini Response
        if job_description and resume:
            input_prompt1 = """
            You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description.
            Please share your professional evaluation on whether the candidate's profile aligns with the role.
            resume:{text}
            job_description:{jd}

            Highlight the strengths and weaknesses of the applicant in relation to the specified job description.
            The structure should be like:
            Strengths:
                . strength1
                . strength2
                . strength3
                ...
            Weaknesses:
                . weakness1
                . weakness2
                . weakness3
                ...
            Overall summary: summary
            """

            response = get_gemini_response(input_prompt1, resume, job_description)
            st.subheader("Resume Analysis:")
            st.write(response)
        else:
            st.warning("Please ensure both files are valid.")

st.caption(" ~ made by GetHire.AI")
