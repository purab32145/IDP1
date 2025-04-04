import os
import sys
import json
import tempfile
import logging
import fitz
import pytesseract
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("document_classifier")

# Use the same API key as in backend.py


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    logger.info(f"Extracting text from PDF: {file_path}")
    try:
        document_text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                document_text += page.get_text() + "\n"
        return document_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
        return None

def is_scanned_document(file_path):
    """Check if a PDF document is scanned or digitized."""
    logger.info(f"Checking if document is scanned: {file_path}")
    try:
        document_text = extract_text_from_pdf(file_path)
        # If there's very little text extracted, it's likely a scanned document
        if document_text and len(document_text.strip()) > 100:
            logger.info("Document appears to be digitized")
            return False, document_text
        logger.info("Document appears to be scanned")
        return True, None
    except Exception as e:
        logger.error(f"Error checking if document is scanned: {e}", exc_info=True)
        return True, None  # Default to treating as scanned if there's an error

def extract_text_from_image(file_path):
    """Extract text from an image file using OCR."""
    logger.info(f"Extracting text from image: {file_path}")
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        logger.info(f"OCR extraction complete, extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}", exc_info=True)
        return None

def pdf_to_images(pdf_path):
    """Convert a PDF file to a series of images and perform OCR."""
    logger.info(f"Converting PDF to images for OCR: {pdf_path}")
    try:
        output_folder = os.path.join(tempfile.gettempdir(), f"pdf_images_for_classification")
        os.makedirs(output_folder, exist_ok=True)

        images_text = ""
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            logger.info(f"Processing {total_pages} pages in PDF")

            for page_num in range(total_pages):
                logger.info(f"Processing page {page_num+1} of {total_pages}")
                page = doc.load_page(page_num)

                # Render at higher resolution for better OCR
                zoom = 2.0  # Increase zoom factor for better quality
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                image_path = os.path.join(output_folder, f"page_{page_num+1:03d}.png")
                pix.save(image_path)

                # Extract text from the image with improved OCR settings
                try:
                    img = Image.open(image_path)

                    # Try different page segmentation modes for better results
                    for psm in [6, 3, 4]:  # 6: Assume single block of text, 3: Auto, 4: Single column
                        config = f'--psm {psm} --oem 3'
                        page_text = pytesseract.image_to_string(img, config=config)

                        if page_text and len(page_text.strip()) > 50:
                            # Found good text, use this
                            break

                    if page_text:
                        images_text += page_text + "\n\n"
                        logger.info(f"OCR successful for page {page_num+1}, extracted {len(page_text)} characters")
                    else:
                        logger.warning(f"OCR failed to extract text from page {page_num+1}")
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}", exc_info=True)

        if images_text:
            logger.info(f"Successfully extracted {len(images_text)} characters from {total_pages} pages")
        else:
            logger.warning("Failed to extract any text from PDF images")

        return images_text
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}", exc_info=True)
        return None

def get_document_text(file_path):
    """Get text from document regardless of type (PDF, image, scanned, digitized)."""
    logger.info(f"Getting document text from: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        is_scanned, text = is_scanned_document(file_path)
        if is_scanned:
            logger.info("Processing PDF as scanned document using OCR")
            # For classification, always use the OCR method for PDFs that appear to be scanned
            ocr_text = pdf_to_images(file_path)
            if ocr_text and len(ocr_text.strip()) > 50:
                logger.info(f"Successfully extracted {len(ocr_text)} characters using OCR")
                return ocr_text
            else:
                logger.warning("OCR extraction produced minimal text, trying alternate method")
                try:
                    # Try using PyPDF2 as a fallback
                    import PyPDF2
                    pdf_text = ""
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text += page_text + "\n"

                    if pdf_text and len(pdf_text.strip()) > 100:
                        logger.info(f"Fallback method extracted {len(pdf_text)} characters")
                        return pdf_text
                except Exception as e:
                    logger.error(f"Error in fallback text extraction: {e}", exc_info=True)
        else:
            logger.info("Processing PDF as digitized document")
            if text and len(text.strip()) > 50:
                return text
            else:
                # Minimal text extracted, try OCR as fallback
                logger.info("Digitized PDF has minimal text, trying OCR as fallback")
                ocr_text = pdf_to_images(file_path)
                if ocr_text and len(ocr_text.strip()) > 50:
                    return ocr_text
                return text
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        logger.info("Processing image file using OCR")
        text = extract_text_from_image(file_path)
        if text:
            return text
        else:
            return "Unable to extract text from image."
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        return None

def classify_document(document_text):
    """Classify document based on its content."""
    if not document_text or len(document_text.strip()) < 50:
        logger.warning("Insufficient text for classification")
        return "unknown"

    logger.info("Classifying document based on content")
    try:
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")

        # Log a sample of the text being sent for classification
        text_sample = document_text[:300] + ("..." if len(document_text) > 300 else "")
        logger.info(f"Document text sample for classification: {text_sample}")

        system_prompt = """
        You are a document classification expert. Your task is to classify the provided document text into one
        of the following categories:
        - invoice: Documents related to billing, payments, receipts, etc.
        - personal: Personal documents like resumes, ID cards, passports, etc.
        - accounting: Financial statements, balance sheets, expense reports, etc.
        - legal: Contracts, agreements, court documents, etc.
        - technical: Technical specifications, manuals, guides, etc.
        - regulatory: Compliance documents, policies, regulations, etc.

        If the document doesn't clearly fall into one of these categories or if the text is too ambiguous,
        classify it as "unknown".

        Your response should be ONLY ONE WORD: the classification category. Do not include any explanations or additional text.
        """

        human_prompt = f"""
        Please classify the following document text:

        {document_text[:4000]}  

        Remember to respond with only one word from the categories: invoice, personal, accounting, legal, technical, regulatory, or unknown.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        logger.info("")
        response = model(messages)
        classification = response.content.strip().lower()

        # Validate the classification
        valid_categories = ["invoice", "personal", "accounting", "legal", "technical", "regulatory", "unknown"]
        if classification not in valid_categories:
            logger.warning(f"Invalid classification: {classification}. Defaulting to 'unknown'")
            classification = "unknown"

        logger.info(f"Document classified as: {classification}")
        return classification
    except Exception as e:
        logger.error(f"Error classifying document: {e}", exc_info=True)
        return "unknown"

def process_document_for_classification(file_path):
    """Process a document to extract its text and classify it."""
    logger.info(f"Processing document for classification: {file_path}")
    document_text = get_document_text(file_path)
    if document_text:
        classification = classify_document(document_text)

      
        classification_file = os.path.join(tempfile.gettempdir(), "document_classification.json")
        with open(classification_file, "w", encoding="utf-8") as f:
            json.dump({"classification": classification, "file_path": file_path}, f)

        logger.info(f"Classification saved to: {classification_file}")
        return classification
    else:
        logger.error("Failed to extract text from document")
        return "unknown"

def get_classification_prompt(document_type):
    """Get a system prompt specific to the document type for extraction."""
    prompts = {
        "invoice": """You are processing an INVOICE document. Focus on extracting information like invoice number,
                     date, vendor details, line items, totals, payment terms, and tax information.""",

        "personal": """You are processing a PERSONAL document. Focus on extracting information like personal
                      identifiers, names, contact details, addresses, educational or employment history,
                      and personal attributes.""",

        "accounting": """You are processing an ACCOUNTING document. Focus on extracting financial data like
                        account numbers, transaction details, amounts, dates, account balances, financial
                        metrics, and accounting periods.""",

        "legal": """You are processing a LEGAL document. Focus on extracting information like parties involved,
                   dates, terms and conditions, clauses, legal requirements, obligations, and any defined
                   legal terms or references.""",

        "technical": """You are processing a TECHNICAL document. Focus on extracting specifications, procedures,
                       technical parameters, requirements, system components, and technical definitions.""",

        "regulatory": """You are processing a REGULATORY document. Focus on extracting compliance requirements,
                        regulatory references, dates of implementation, affected parties, scope of regulations,
                        and compliance procedures.""",

        "unknown": """You are processing a document of unknown type. Extract any relevant structured information
                     that appears to be important based on the document's context."""
    }

    return prompts.get(document_type.lower(), prompts["unknown"])


if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*Session state does not function.*")

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            classification = process_document_for_classification(file_path)
            print(f"Document classified as: {classification}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Usage: python classification.py <file_path>")

        
        try:
            import streamlit as st
            st.title("Document Classification")
            uploaded_file = st.file_uploader("Upload a document for classification", type=["pdf", "png", "jpg", "jpeg"])
            if uploaded_file:
                file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Classifying document..."):
                    classification = process_document_for_classification(file_path)
                st.success(f"Document classified as: {classification}")
        except:

            pass
