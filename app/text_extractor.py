from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text()
        return extracted_text
    except Exception as e:
        print(f"Error while extracting text: {e}")
        return None
