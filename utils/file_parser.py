import pdfplumber
import docx

def extract_text(filepath):
    if filepath.endswith('.pdf'):
        return extract_text_from_pdf(filepath)
    elif filepath.endswith(('.doc', '.docx')):
        return extract_text_from_word(filepath)
    else:
        return None

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_word(word_path):
    text = ""
    doc = docx.Document(word_path)
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text.strip()
