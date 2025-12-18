# main.py
import PyPDF2
import docx
import openai

class CoverLetterAgent:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None

    def analyze_and_generate(self, resume_text, jd_text):
        if not self.client:
            return "Error: API Key is missing."

        prompt = f"""
        You are an expert career coach. Write a professional cover letter based on:
        
        RESUME:
        {resume_text}
        
        JOB DESCRIPTION:
        {jd_text}
        
        Align the candidate's skills with the job requirements. Keep it professional.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful career assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {str(e)}"

# Helper function to read files
def extract_text_from_file(uploaded_file):
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
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return text