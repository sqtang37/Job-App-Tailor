import os
import threading

import PyPDF2
import docx
import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

from input_parser import parse_resume_and_jd


# ---- File extraction stays the same ----
def extract_text_from_file(uploaded_file):
    """Extracts text from PDF or DOCX files."""
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return "Error: Unsupported file format. Please upload PDF or DOCX."
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return text


@st.cache_resource
def get_local_model_and_tokenizer(
    base_model_name="Qwen/Qwen3-0.6B-Base",
    adapter_path="./results",
):
    """
    Loads base model + LoRA adapter once per Streamlit session.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=None,          # avoid accelerate offload issues
        trust_remote_code=True,
    ).to(device)

    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()
    return model, tokenizer, device


def _build_prompt_from_fields(fields: dict) -> str:
    job_title = fields.get("Job Title", "Job Position")
    hiring_company = fields.get("Hiring Company", "Hiring Company")
    applicant_name = fields.get("Applicant Name", "Applicant")
    skills = fields.get("Skillsets", "")
    experience = fields.get("Current Working Experience", "")
    qualifications = fields.get("Qualifications", "")

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a complete, professional cover letter (3–4 short paragraphs) for the position of {job_title} at {hiring_company}.

Formatting rules (IMPORTANT):
- Output plain text only (no Markdown, no **bold**, no bullet points, no headers).
- Do NOT include any template placeholders like [Your Name], [Address], [Email], [Phone], or [Date].
- Do NOT include a title like "Cover Letter" and do NOT include separator lines like "---".
- Start with: "Dear Hiring Manager,"
- End with a simple sign-off and the applicant name only (e.g., "Sincerely,\\n{applicant_name}").

### Input:
Applicant: {applicant_name}
Skills: {skills}
Experience: {experience}
Qualifications: {qualifications}

### Response:
Dear Hiring Manager,
"""
    return prompt


def generate_cover_letter(
    resume_text: str,
    jd_text: str,
    api_key: str = None,  # ignored; kept for compatibility
    base_model_name: str = "Qwen/Qwen3-0.6B-Base",
    adapter_path: str = "./results",
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """
    Non-streaming version: returns the whole cover letter string.
    """
    model, tokenizer, device = get_local_model_and_tokenizer(base_model_name, adapter_path)

    fields = parse_resume_and_jd(resume_text, jd_text)
    prompt = _build_prompt_from_fields(fields)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            do_sample=do_sample,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Return only after marker if present
    marker = "### Response:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def stream_cover_letter(
    resume_text: str,
    jd_text: str,
    base_model_name: str = "Qwen/Qwen3-0.6B-Base",
    adapter_path: str = "./results",
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """
    Streaming version: yields text chunks as generation happens.
    Use this in Streamlit to show live typing.
    """
    model, tokenizer, device = get_local_model_and_tokenizer(base_model_name, adapter_path)

    fields = parse_resume_and_jd(resume_text, jd_text)
    prompt = _build_prompt_from_fields(fields)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
        do_sample=do_sample,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Run generation in background thread
    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    # Yield chunks as they appear
    for chunk in streamer:
        yield chunk
