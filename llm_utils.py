# llm_utils.py
import PyPDF2
import docx
import openai

def extract_text_from_file(uploaded_file):
    """Extracts text from PDF or DOCX files."""
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text()
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return "Error: Unsupported file format. Please upload PDF or DOCX."
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
    return text

def generate_cover_letter(resume_text, jd_text, api_key):
    """
    Generates a cover letter using OpenAI API.
    """
    if not api_key:
        return "Please enter your OpenAI API Key in the sidebar to proceed."

    client = openai.OpenAI(api_key=api_key)

    prompt = f"""
    You are an expert career coach. Write a professional, persuasive cover letter based on the following:
    
    RESUME:
    {resume_text}
    
    JOB DESCRIPTION:
    {jd_text}
    
    The cover letter should align the candidate's skills with the job requirements. 
    Keep it concise, professional, and engaging.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful career assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the AI generation: {str(e)}"