# app.py
import streamlit as st
from main import CoverLetterAgent, extract_text_from_file

# --- Page Config ---
st.set_page_config(page_title="Career Agent", layout="wide")

# --- Session State (to remember which button was clicked) ---
if 'active_agent' not in st.session_state:
    st.session_state['active_agent'] = 'cover_letter'  # Default view

# --- Sidebar: Navigation & Settings ---
with st.sidebar:
    st.title("🤖 Agent Command")
    
    # The two requested buttons
    if st.button("📝 Cover Letter Agent", use_container_width=True):
        st.session_state['active_agent'] = 'cover_letter'
        
    if st.button("✂️ Resume Tailor Agent", use_container_width=True):
        st.session_state['active_agent'] = 'resume_tailor'
    
    st.markdown("---")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

# --- Page Logic ---

# 1. PAGE: Cover Letter Agent
if st.session_state['active_agent'] == 'cover_letter':
    st.header("📝 Cover Letter Agent")
    st.markdown("Upload your resume and paste the Job Description (JD) to generate a letter.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Your Inputs")
        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=['pdf', 'docx'])
        jd_text = st.text_area("Paste Job Description", height=300)
        
        generate_btn = st.button("Generate Cover Letter", type="primary")

    with col2:
        st.subheader("2. Result")
        if generate_btn:
            if not api_key:
                st.error("Please enter your API Key in the sidebar.")
            elif not uploaded_file or not jd_text:
                st.warning("Please upload both a resume and a job description.")
            else:
                with st.spinner("AI is writing your letter..."):
                    # Extract text using helper from main.py
                    resume_text = extract_text_from_file(uploaded_file)
                    
                    # Initialize Agent from main.py
                    agent = CoverLetterAgent(api_key)
                    
                    # Generate
                    result = agent.analyze_and_generate(resume_text, jd_text)
                    
                    st.text_area("Generated Letter", value=result, height=400)
                    st.download_button("Download", result, "cover_letter.txt")

# 2. PAGE: Resume Tailor Agent
elif st.session_state['active_agent'] == 'resume_tailor':
    st.header("✂️ Resume Tailor Agent")
    st.info("This agent is waiting to be built! (You can add logic for this in main.py later)")