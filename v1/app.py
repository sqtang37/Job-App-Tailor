# app.py
import streamlit as st
from llm_utils import extract_text_from_file, stream_cover_letter

# --- Page Config ---
st.set_page_config(page_title="Career Agent", layout="wide")

# --- Session State ---
if "active_agent" not in st.session_state:
    st.session_state["active_agent"] = "cover_letter"

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Agent Command")
    # Only keep the cover letter agent (no resume tailor, no API key)
    if st.button("📝 Cover Letter Agent", use_container_width=True):
        st.session_state["active_agent"] = "cover_letter"

# --- Page: Cover Letter Agent ---
if st.session_state["active_agent"] == "cover_letter":
    st.header("📝 Cover Letter Agent")
    st.markdown("Upload your resume and paste the Job Description (JD) to generate a letter.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Your Inputs")
        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
        jd_text = st.text_area("Paste Job Description", height=300)
        generate_btn = st.button("Generate Cover Letter", type="primary")

    with col2:
        st.subheader("2. Result")

        if generate_btn:
            if not uploaded_file or not jd_text:
                st.warning("Please upload both a resume and a job description.")
            else:
                resume_text = extract_text_from_file(uploaded_file)

                status = st.caption("Generating…")
                live_box = st.empty()   # live output box (not a widget)
                full_text = ""

                # Stream chunks and update live preview
                for chunk in stream_cover_letter(resume_text, jd_text):
                    full_text += chunk
                    # Use markdown code block for a stable live “typing” effect
                    live_box.markdown(f"```text\n{full_text}\n```")

                status.empty()

                # Show final output ONCE as text_area (no duplicates)
                st.text_area(
                    "Generated Letter",
                    value=full_text,
                    height=400,
                    key="generated_letter_final",
                )
                st.download_button("Download", full_text, "cover_letter.txt")
