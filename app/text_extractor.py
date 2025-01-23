from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file for use with an LLM.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text()
        return extracted_text
    except Exception as e:
        print(f"Error while extracting text: {e}")
        return None
#
# def extract_text_from_docx(docx_path):
#     """
#     Extract text from a Word document.
#     :param docx_path: Path to the Word document.
#     :return: Extracted text as a string.
#     """
#     doc = Document(docx_path)
#     text = ""
#     for paragraph in doc.paragraphs:
#         text += paragraph.text + "\n"
#     return text
