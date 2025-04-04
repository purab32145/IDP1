
# import streamlit as st

# st.set_page_config(page_title="Intelligent Document Processor", layout="wide")

# import os
# import json
# import tempfile
# import subprocess
# import time
# import sys
# import csv
# import io
# import logging

# import pytesseract
# from PIL import Image
# import re
# from langchain_openai import ChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage
# import openai
# from datetime import datetime


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("frontend.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("frontend")

# OPENAI_API_KEY = "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"
# openai.api_key = OPENAI_API_KEY
# def get_fitz_module():
#     """Get the fitz module (PyMuPDF) with proper error handling."""
#     logger.info("Trying to import PyMuPDF")
#     try:
#         try:
#             import pymupdf
#             logger.info("Imported pymupdf successfully")
#             return pymupdf
#         except ImportError:
#             try:
#                 import fitz
#                 logger.info("Imported fitz successfully")
#                 return fitz
#             except ImportError as e:
#                 if "No module named 'tools'" in str(e):
#                     # Clean up sys.modules to remove the problematic import
#                     if 'fitz' in sys.modules:
#                         del sys.modules['fitz']
#                     # Try a different import approach for PyMuPDF
#                     import PyMuPDF
#                     logger.info("Imported PyMuPDF successfully")
#                     return PyMuPDF
#                 else:
#                     raise
#     except Exception as e:
#         logger.error(f"Failed to import PyMuPDF: {e}")
#         return None

# steps = ["Configure Extraction Fields", "Upload Document", "View Results"]

# if "current_step_index" not in st.session_state:
#     st.session_state.current_step_index = 0
# if "extraction_fields" not in st.session_state:
#     st.session_state.extraction_fields = []
# if "extracted_data" not in st.session_state:
#     st.session_state.extracted_data = None
# if "processing_document" not in st.session_state:
#     st.session_state.processing_document = False
# if "uploaded_file_path" not in st.session_state:
#     st.session_state.uploaded_file_path = None
# if "document_classification" not in st.session_state:
#     st.session_state.document_classification = None
# if "navigation" not in st.session_state:
#     st.session_state.navigation = "home"
# if "document_type" not in st.session_state:
#     st.session_state.document_type = None
# if "extracted_text" not in st.session_state:
#     st.session_state.extracted_text = None

# def go_next():
#     """Navigate to the next step in the workflow."""
#     if st.session_state.current_step_index < len(steps) - 1:
#         st.session_state.current_step_index += 1
#     st.rerun()

# def go_previous():
#     """Navigate to the previous step in the workflow."""
#     if st.session_state.current_step_index > 0:
#         st.session_state.current_step_index -= 1
#     st.rerun()

# # Classification functions implemented directly in the frontend
# def extract_text_from_pdf(file_path):
#     """Extract text directly from a PDF file."""
#     logger.info(f"Extracting text from PDF: {file_path}")
#     try:
#         fitz = get_fitz_module()
#         if not fitz:
#             logger.error("PyMuPDF module not available")
#             return ""

#         document_text = ""
#         try:
#             with fitz.open(file_path) as doc:
#                 for page in doc:
#                     document_text += page.get_text() + "\n"
#             logger.info(f"Extracted {len(document_text)} characters from PDF using PyMuPDF")
#             return document_text
#         except Exception as e:
#             logger.error(f"Error extracting text from PDF with PyMuPDF: {e}")
#             return ""
#     except Exception as e:
#         logger.error(f"Error in extract_text_from_pdf: {e}")
#         return ""

# def is_scanned_document(file_path):
#     """Check if a PDF is a scanned document and extract text if digitized.
#     Returns a tuple of (is_scanned, text_if_not_scanned)"""
#     logger.info(f"Checking if document is scanned: {file_path}")

#     try:
#         # Try direct text extraction
#         document_text = extract_text_from_pdf(file_path)

#         # Log sample of extracted text
#         text_sample = document_text[:100] + "..." if document_text and len(document_text) > 100 else document_text
#         logger.info(f"Text sample from PDF: {text_sample}")

#         # If there's very little text extracted, it's likely a scanned document
#         if document_text and len(document_text.strip()) > 100:
#             logger.info(f"Document appears to be digitized based on text content ({len(document_text)} chars)")
#             return False, document_text

#         # Check for images in the PDF
#         fitz = get_fitz_module()
#         if fitz:
#             try:
#                 with fitz.open(file_path) as doc:
#                     total_pages = len(doc)
#                     images_count = 0

#                     for page_num in range(total_pages):
#                         page = doc[page_num]
#                         images_count += len(page.get_images())

#                     logger.info(f"PDF has {images_count} images across {total_pages} pages")
#                     if images_count > 0:
#                         logger.info("Document appears to be scanned (contains images)")
#                         return True, document_text
#             except Exception as e:
#                 logger.error(f"Error checking images in PDF: {e}")

#         logger.info("Document appears to be scanned based on minimal text content")
#         return True, document_text
#     except Exception as e:
#         logger.error(f"Error in is_scanned_document: {e}")
#         return True, ""  # Default to treating as scanned if there's an error

# def extract_text_from_image(file_path):
#     """Extract text from an image file using OCR."""
#     logger.info(f"Extracting text from image: {file_path}")
#     try:
#         image = Image.open(file_path)
#         text = pytesseract.image_to_string(image)
#         logger.info(f"OCR extraction complete, extracted {len(text)} characters")
#         return text
#     except Exception as e:
#         logger.error(f"Error during OCR processing: {e}")
#         return None

# def pdf_to_images(pdf_path):
#     """Convert a PDF file to a series of images for OCR."""
#     logger.info(f"Converting PDF to images for OCR: {pdf_path}")
#     try:
#         # Try to get PyMuPDF
#         fitz = get_fitz_module()
#         if not fitz:
#             logger.error("PyMuPDF (fitz) module not available")
#             return ""

#         output_folder = os.path.join(tempfile.gettempdir(), f"pdf_images_for_classification")
#         os.makedirs(output_folder, exist_ok=True)

#         images_text = ""
#         try:
#             with fitz.open(pdf_path) as doc:
#                 total_pages = len(doc)
#                 logger.info(f"Processing {total_pages} pages in PDF")

#                 for page_num in range(total_pages):
#                     logger.info(f"Processing page {page_num+1} of {total_pages}")
#                     page = doc.load_page(page_num)

#                     # Render at higher resolution for better OCR
#                     zoom = 2.0  # Increase zoom factor for better quality
#                     mat = fitz.Matrix(zoom, zoom)
#                     pix = page.get_pixmap(matrix=mat, alpha=False)

#                     image_path = os.path.join(output_folder, f"page_{page_num+1:03d}.png")
#                     pix.save(image_path)

#                     # Extract text from the image with improved OCR settings
#                     try:
#                         img = Image.open(image_path)

#                         # Try different page segmentation modes for better results
#                         page_text = ""
#                         for psm in [6, 3, 4]:  # 6: Assume single block of text, 3: Auto, 4: Single column
#                             config = f'--psm {psm} --oem 3'
#                             text = pytesseract.image_to_string(img, config=config)

#                             if text and len(text.strip()) > len(page_text.strip()):
#                                 page_text = text

#                         if page_text:
#                             images_text += page_text + "\n\n"
#                             logger.info(f"OCR successful for page {page_num+1}, extracted {len(page_text)} characters")
#                         else:
#                             logger.warning(f"OCR failed to extract text from page {page_num+1}")
#                     except Exception as e:
#                         logger.error(f"Error processing image {image_path}: {e}")
#         except Exception as e:
#             logger.error(f"Error opening PDF with PyMuPDF: {e}")
#             return ""

#         if images_text:
#             logger.info(f"Successfully extracted {len(images_text)} characters from {total_pages} pages")
#             return images_text
#         else:
#             logger.warning("Failed to extract any text from PDF images")
#             return ""
#     except Exception as e:
#         logger.error(f"Error in pdf_to_images: {e}")
#         return ""

# def get_document_text(file_path):
#     """Get text from document regardless of type (PDF, image, scanned, digitized)."""
#     logger.info(f"Getting document text from: {file_path}")
#     file_extension = os.path.splitext(file_path)[1].lower()

#     if file_extension == '.pdf':
#         # Use PyMuPDF for both digitized and scanned PDFs
#         is_scanned, text = is_scanned_document(file_path)

#         if is_scanned:
#             logger.info("Processing PDF as scanned document using OCR")
#             ocr_text = pdf_to_images(file_path)
#             if ocr_text and len(ocr_text.strip()) > 50:
#                 logger.info(f"Successfully extracted {len(ocr_text)} characters using OCR")
#                 return ocr_text
#             elif text and len(text.strip()) > 0:
#                 logger.info(f"Using text extracted directly ({len(text)} chars)")
#                 return text
#             else:
#                 logger.warning("Failed to extract meaningful text")
#                 return "This document appears to be a scanned PDF that could not be processed effectively."
#         else:
#             logger.info("Processing PDF as digitized document")
#             if text and len(text.strip()) > 50:
#                 return text
#             else:
#                 logger.warning("Minimal text extracted from digitized PDF")
#                 return "This document appears to be a PDF with minimal extractable text content."

#     elif file_extension in ['.jpg', '.jpeg', '.png']:
#         logger.info("Processing image file using OCR")
#         text = extract_text_from_image(file_path)
#         if text:
#             return text
#         else:
#             return "Unable to extract text from image."
#     else:
#         logger.error(f"Unsupported file type: {file_extension}")
#         return "Unsupported document type for classification."

# # Use the classification module for document classification
# def classify_document_via_frontend(file_path):
#     """Classify document by triggering classification.py"""
#     try:
#         with st.spinner("Classifying document... (This may take a moment)"):
#             # Import classification module and use it instead of direct implementation
#             import classification
#             classification_result = classification.process_document_for_classification(file_path)
#             return classification_result
#     except Exception as e:
#         st.warning(f"Error during document classification: {str(e)}")
#         return "unknown"

# current_step = steps[st.session_state.current_step_index]
# st.title(current_step)

# if current_step == "Configure Extraction Fields":
#     st.header("Step 1: Configure Extraction Fields")

#     # Unified Extraction Configuration
#     with st.expander("Extraction Field Configuration", expanded=True):
#         st.subheader("Load Field Configuration from CSV")
#         uploaded_csv = st.file_uploader("Upload CSV for field configuration", type=["csv"], key="csv_config")
#         if uploaded_csv:
#             try:
#                 csv_string = io.StringIO(uploaded_csv.getvalue().decode("utf-8"))
#                 reader = csv.DictReader(csv_string)
#                 csv_fields = []
#                 for row in reader:
#                     new_field = {
#                         "name": row.get("Field Name", row.get("name", "")).strip(),
#                         "type": row.get("Field Type", row.get("type", "")).strip(),
#                         "description": row.get("Field Description", row.get("description", "")).strip()
#                     }
#                     if new_field["type"] == "List":
#                         new_field["item_type"] = row.get("List Item Type", row.get("item_type", "")).strip()
#                     csv_fields.append(new_field)
#                 st.session_state.extraction_fields = csv_fields
#                 st.success("Field configuration loaded from CSV successfully.")
#             except Exception as e:
#                 st.error(f"Error loading CSV configuration: {e}")

#         st.subheader("Add New Field")
#         col1, col2 = st.columns(2)
#         with col1:
#             field_name = st.text_input("Field Name", key="field_name")
#             field_type = st.selectbox("Field Type",
#                                       ["String", "Number", "Date", "Boolean", "List", "Nested Object", "Table"],
#                                       key="field_type")
#         with col2:
#             field_description = st.text_area("Description",
#                                              placeholder="Describe what to extract (e.g., from text or tables)...",
#                                              key="field_description")
#         if field_type == "Nested Object":
#             st.info("For nested objects, add child fields after creating this parent field.")
#         elif field_type == "List":
#             list_item_type = st.selectbox("List Item Type",
#                                           ["String", "Number", "Date", "Boolean", "Object"],
#                                           key="list_item_type")
#         elif field_type == "Table":
#             st.info("For table extraction, describe the table structure and content you want to extract.")

#         if st.button("Add Field", key="add_field"):
#             if field_name:
#                 new_field = {
#                     "name": field_name,
#                     "type": field_type,
#                     "description": field_description
#                 }
#                 if field_type == "List":
#                     new_field["item_type"] = list_item_type
#                 st.session_state.extraction_fields.append(new_field)
#                 st.success(f"Added field: {field_name}")

#         st.subheader("Configured Fields")
#         if not st.session_state.extraction_fields:
#             st.info("No fields configured yet. Add fields above or load a CSV configuration.")
#         else:
#             for i, field in enumerate(st.session_state.extraction_fields):
#                 col1, col2, col3, col4 = st.columns([2, 1, 3, 1])
#                 col1.write(field["name"])
#                 col2.write(field["type"])
#                 description = field["description"]
#                 col3.write(description[:50] + "..." if len(description) > 50 else description)
#                 if col4.button("Delete", key=f"delete_{i}"):
#                     st.session_state.extraction_fields.pop(i)
#                     st.rerun()

#         if st.session_state.extraction_fields:
#             if st.button("Save Field Configuration"):
#                 config_file = os.path.join(tempfile.gettempdir(), "extraction_fields.json")
#                 with open(config_file, "w") as f:
#                     json.dump(st.session_state.extraction_fields, f, indent=2)
#                 st.success(f"Configuration saved to {config_file}")

#     if st.button("Next", key="next1"):
#         go_next()

# elif current_step == "Upload Document":
#     st.header("Step 2: Upload and Process Document")
#     st.subheader("Upload Document for Processing")

#     uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"], key="file_upload")

#     if uploaded_file:
#         if not st.session_state.uploaded_file_path or os.path.basename(st.session_state.uploaded_file_path) != uploaded_file.name:
#             file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             st.session_state.uploaded_file_path = file_path

#             # Classify document directly through frontend
#             with st.spinner("Classifying document..."):
#                 document_classification = classify_document_via_frontend(file_path)
#                 st.session_state.document_classification = document_classification
#                 # Also set document_type for compatibility with other code
#                 st.session_state.document_type = document_classification

#             st.success(f"New file uploaded: {uploaded_file.name}")
#         else:
#             file_path = st.session_state.uploaded_file_path

#         # Display document classification
#         if st.session_state.document_classification:
#             st.info(f"📄 Document classified as: **{st.session_state.document_classification.upper()}**")

#             # Map classifications to descriptions
#             classification_descriptions = {
#                 "invoice": "Invoice documents typically contain billing information, payment details, and itemized lists of products or services.",
#                 "personal": "Personal documents contain individual information like identification, resumes, or personal records.",
#                 "accounting": "Accounting documents contain financial data, statements, balance sheets, and other financial records.",
#                 "legal": "Legal documents contain contractual information, legal terms, and legal provisions or requirements.",
#                 "technical": "Technical documents contain specifications, procedures, or detailed technical information.",
#                 "regulatory": "Regulatory documents contain compliance information, policies, standards, or regulations.",
#                 "unknown": "The document type could not be determined with confidence."
#             }

#             doc_type = st.session_state.document_classification.lower()
#             if doc_type in classification_descriptions:
#                 st.caption(classification_descriptions[doc_type])

#         # Unified extraction section
#         st.subheader("Extract Data from Document")
#         if st.session_state.extraction_fields:
#             if st.button("Process Document", disabled=st.session_state.processing_document):
#                 st.session_state.processing_document = True
#                 st.session_state.extracted_data = None

#                 output_dir = os.path.join(tempfile.gettempdir(), "extraction_output")
#                 os.makedirs(output_dir, exist_ok=True)

#                 config_file = os.path.join(tempfile.gettempdir(), "extraction_fields.json")
#                 with open(config_file, "w") as f:
#                     json.dump(st.session_state.extraction_fields, f, indent=2)

#                 status = st.empty()
#                 with st.spinner("Processing document. This may take a few moments..."):
#                     try:
#                         # Use the new unified backend
#                         unified_backend_script = "unified_backend.py"
#                         if not os.path.exists(unified_backend_script):
#                             unified_backend_script = os.path.join(os.path.dirname(__file__), "unified_backend.py")

#                         cmd = [sys.executable, unified_backend_script, file_path, config_file, output_dir]
#                         process = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

#                         if process.returncode != 0:
#                             st.error(f"Error processing document: {process.stderr}")
#                             status.error("Processing failed. See error details below.")
#                             st.code(process.stderr)
#                         else:
#                             extracted_fields_path = os.path.join(output_dir, "extracted_data.json")
#                             if not os.path.exists(extracted_fields_path):
#                                 alt_path = "results.json"
#                                 if os.path.exists(alt_path):
#                                     extracted_fields_path = alt_path

#                             if os.path.exists(extracted_fields_path):
#                                 with open(extracted_fields_path, 'r') as f:
#                                     extracted_data = f.read()
#                                 try:
#                                     parsed_data = json.loads(extracted_data)
#                                     st.session_state.extracted_data = parsed_data
#                                     # Also set extracted_text for compatibility
#                                     st.session_state.extracted_text = True
#                                     status.success("Document processed successfully!")
#                                 except json.JSONDecodeError:
#                                     st.session_state.extracted_data = {}
#                                     status.warning("Document processed but output is not valid JSON.")
#                             else:
#                                 status.error("Processing completed but extracted data file was not found.")
#                     except subprocess.TimeoutExpired:
#                         status.error("Processing timed out. The document may be too large or complex.")
#                     except Exception as e:
#                         status.error(f"An error occurred: {str(e)}")
#                 st.session_state.processing_document = False
#         else:
#             st.warning("Please configure extraction fields before processing.")

#     else:
#         st.info("Upload a file to begin processing. The document will be automatically classified upon upload.")

#     col1, col2 = st.columns(2)
#     if col1.button("Previous", key="prev2"):
#         go_previous()
#     if col2.button("Next", key="next2"):
#         go_next()

# elif current_step == "View Results":
#     st.header("Step 3: View Extraction Results")

#     # Display document classification if available
#     if st.session_state.document_classification:
#         st.info(f"📄 Document Type: **{st.session_state.document_classification.upper()}**")

#     if st.session_state.extracted_data:
#         st.subheader("Extracted Data")

#         # If using our new structure with document_classification and extracted_data fields
#         if isinstance(st.session_state.extracted_data, dict) and "extracted_data" in st.session_state.extracted_data:
#             extracted_data = st.session_state.extracted_data["extracted_data"]

#             # Get classification from response if available
#             if "document_classification" in st.session_state.extracted_data:
#                 doc_class = st.session_state.extracted_data["document_classification"]
#                 if doc_class != st.session_state.document_classification:
#                     st.info(f"Note: The model classified this document as: **{doc_class.upper()}**")

#             # Display extracted data
#             if isinstance(extracted_data, dict):
#                 for key, value in extracted_data.items():
#                     st.write(f"**{key}:**")
#                     if value is None or (isinstance(value, str) and not value.strip()):
#                         st.write("NULL")
#                     elif isinstance(value, dict):
#                         st.json(value)
#                     elif isinstance(value, list):
#                         # Check if this is a table
#                         if all(isinstance(item, dict) for item in value) and len(value) > 0:
#                             st.dataframe(value)
#                         else:
#                             st.json(value)
#                     else:
#                         st.write(value)
#             else:
#                 st.write("No structured data found.")

#         # Legacy format handling
#         else:
#             for key, value in st.session_state.extracted_data.items():
#                 st.write(f"**{key}:**")
#                 if value is None or (isinstance(value, str) and not value.strip()):
#                     st.write("NULL")
#                 elif isinstance(value, dict):
#                     st.json(value)
#                 elif isinstance(value, list):
#                     # Check if this might be a table structure
#                     if all(isinstance(item, dict) for item in value) and len(value) > 0:
#                         st.dataframe(value)
#                     else:
#                         st.json(value)
#                 else:
#                     st.write(value)
#     else:
#         st.info("No extraction data available. Please process a document first.")

#     if st.button("Previous", key="prev3"):
#         go_previous()

# def check_dependencies():
#     """Check if all required dependencies are installed."""
#     missing_deps = []
#     optional_deps = []

#     # Essential dependencies
#     try:
#         import openai
#     except ImportError:
#         missing_deps.append("openai")

#     try:
#         import PyPDF2
#     except ImportError:
#         missing_deps.append("PyPDF2")

#     try:
#         import pytesseract
#     except ImportError:
#         missing_deps.append("pytesseract")

#     # Optional dependencies - we have fallbacks for these
#     try:
#         import PyMuPDF
#     except ImportError:
#         try:
#             import fitz
#         except ImportError:
#             optional_deps.append("PyMuPDF or fitz")

#     try:
#         from pdf2image import convert_from_path
#     except ImportError:
#         optional_deps.append("pdf2image")

#     try:
#         from wand.image import Image
#     except ImportError:
#         if "pdf2image" in optional_deps:
#             optional_deps.append("wand")

#     return missing_deps, optional_deps

# # Main app
# def main():
#     st.set_page_config(page_title="Intelligent Document Processor", layout="wide")

#     # Initialize session state
#     if 'processed_data' not in st.session_state:
#         st.session_state.processed_data = None
#     if 'uploaded_file' not in st.session_state:
#         st.session_state.uploaded_file = None
#     if 'document_text' not in st.session_state:
#         st.session_state.document_text = None
#     if 'document_type' not in st.session_state:
#         st.session_state.document_type = None
#     if 'extracted_text' not in st.session_state:
#         st.session_state.extracted_text = None
#     if 'navigation' not in st.session_state:
#         st.session_state.navigation = "home"
#     if 'current_step_index' not in st.session_state:
#         st.session_state.current_step_index = 0

#     # Check dependencies
#     missing_deps, optional_deps = check_dependencies()
#     if missing_deps:
#         st.error(f"Missing required dependencies: {', '.join(missing_deps)}. Please install them using `pip install {' '.join(missing_deps)}`")
#         st.stop()

#     if optional_deps:
#         st.warning(f"Some optional dependencies are missing: {', '.join(optional_deps)}. "
#                    f"The app will use alternative methods, but installing these may improve performance: "
#                    f"`pip install {' '.join(optional_deps)}`")

#     # Navigation sidebar
#     st.sidebar.title("Navigation")
#     st.sidebar.write("Steps:")
#     for i, step in enumerate(steps):
#         if st.sidebar.button(step, key=f"nav_{i}"):
#             st.session_state.current_step_index = i
#             st.rerun()

#     # The current_step and corresponding UI are handled in the main part of the script
#     # We don't need to add any additional page rendering here

# if __name__ == "__main__":
#     main()
import streamlit as st

st.set_page_config(page_title="Intelligent Document Processor", layout="wide")

import os
import json
import tempfile
import subprocess
import time
import sys
import csv
import io
import logging

import pytesseract
from PIL import Image
import re
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import openai
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("frontend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("frontend")

OPENAI_API_KEY=""
openai.api_key = OPENAI_API_KEY

def get_fitz_module():
    """Get the fitz module (PyMuPDF) with proper error handling."""
    logger.info("Trying to import PyMuPDF")
    try:
        try:
            import pymupdf
            logger.info("Imported pymupdf successfully")
            return pymupdf
        except ImportError:
            try:
                import fitz
                logger.info("Imported fitz successfully")
                return fitz
            except ImportError as e:
                if "No module named 'tools'" in str(e):
                    if 'fitz' in sys.modules:
                        del sys.modules['fitz']
                    import PyMuPDF
                    logger.info("Imported PyMuPDF successfully")
                    return PyMuPDF
                else:
                    raise
    except Exception as e:
        logger.error(f"Failed to import PyMuPDF: {e}")
        return None

steps = ["Configure Extraction Fields", "Upload Document", "View Results"]

if "current_step_index" not in st.session_state:
    st.session_state.current_step_index = 0
if "extraction_fields" not in st.session_state:
    st.session_state.extraction_fields = []
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "processing_document" not in st.session_state:
    st.session_state.processing_document = False
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "document_classification" not in st.session_state:
    st.session_state.document_classification = None
if "navigation" not in st.session_state:
    st.session_state.navigation = "home"
if "document_type" not in st.session_state:
    st.session_state.document_type = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None

def go_next():
    """Navigate to the next step in the workflow."""
    if st.session_state.current_step_index < len(steps) - 1:
        st.session_state.current_step_index += 1
    st.rerun()

def go_previous():
    """Navigate to the previous step in the workflow."""
    if st.session_state.current_step_index > 0:
        st.session_state.current_step_index -= 1
    st.rerun()

# Classification functions implemented directly in the frontend
def extract_text_from_pdf(file_path):
    """Extract text directly from a PDF file."""
    logger.info(f"Extracting text from PDF: {file_path}")
    try:
        fitz = get_fitz_module()
        if not fitz:
            logger.error("PyMuPDF module not available")
            return ""
        document_text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    document_text += page.get_text() + "\n"
            logger.info(f"Extracted {len(document_text)} characters from PDF using PyMuPDF")
            return document_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF with PyMuPDF: {e}")
            return ""
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {e}")
        return ""

def is_scanned_document(file_path):
    """Check if a PDF is a scanned document and extract text if digitized.
    Returns a tuple of (is_scanned, text_if_not_scanned)"""
    logger.info(f"Checking if document is scanned: {file_path}")
    try:
        document_text = extract_text_from_pdf(file_path)
        text_sample = document_text[:100] + "..." if document_text and len(document_text) > 100 else document_text
        logger.info(f"Text sample from PDF: {text_sample}")
        if document_text and len(document_text.strip()) > 100:
            logger.info(f"Document appears to be digitized based on text content ({len(document_text)} chars)")
            return False, document_text
        fitz = get_fitz_module()
        if fitz:
            try:
                with fitz.open(file_path) as doc:
                    total_pages = len(doc)
                    images_count = 0
                    for page_num in range(total_pages):
                        page = doc[page_num]
                        images_count += len(page.get_images())
                    logger.info(f"PDF has {images_count} images across {total_pages} pages")
                    if images_count > 0:
                        logger.info("Document appears to be scanned (contains images)")
                        return True, document_text
            except Exception as e:
                logger.error(f"Error checking images in PDF: {e}")
        logger.info("Document appears to be scanned based on minimal text content")
        return True, document_text
    except Exception as e:
        logger.error(f"Error in is_scanned_document: {e}")
        return True, ""

def extract_text_from_image(file_path):
    """Extract text from an image file using OCR."""
    logger.info(f"Extracting text from image: {file_path}")
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        logger.info(f"OCR extraction complete, extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}")
        return None

def pdf_to_images(pdf_path):
    """Convert a PDF file to a series of images for OCR."""
    logger.info(f"Converting PDF to images for OCR: {pdf_path}")
    try:
        fitz = get_fitz_module()
        if not fitz:
            logger.error("PyMuPDF (fitz) module not available")
            return ""
        output_folder = os.path.join(tempfile.gettempdir(), f"pdf_images_for_classification")
        os.makedirs(output_folder, exist_ok=True)
        images_text = ""
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                logger.info(f"Processing {total_pages} pages in PDF")
                for page_num in range(total_pages):
                    logger.info(f"Processing page {page_num+1} of {total_pages}")
                    page = doc.load_page(page_num)
                    zoom = 2.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    image_path = os.path.join(output_folder, f"page_{page_num+1:03d}.png")
                    pix.save(image_path)
                    try:
                        img = Image.open(image_path)
                        page_text = ""
                        for psm in [6, 3, 4]:
                            config = f'--psm {psm} --oem 3'
                            text = pytesseract.image_to_string(img, config=config)
                            if text and len(text.strip()) > len(page_text.strip()):
                                page_text = text
                        if page_text:
                            images_text += page_text + "\n\n"
                            logger.info(f"OCR successful for page {page_num+1}, extracted {len(page_text)} characters")
                        else:
                            logger.warning(f"OCR failed to extract text from page {page_num+1}")
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {e}")
        except Exception as e:
            logger.error(f"Error opening PDF with PyMuPDF: {e}")
            return ""
        if images_text:
            logger.info(f"Successfully extracted {len(images_text)} characters from {total_pages} pages")
            return images_text
        else:
            logger.warning("Failed to extract any text from PDF images")
            return ""
    except Exception as e:
        logger.error(f"Error in pdf_to_images: {e}")
        return ""

def get_document_text(file_path):
    """Get text from document regardless of type (PDF, image, scanned, digitized)."""
    logger.info(f"Getting document text from: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        is_scanned, text = is_scanned_document(file_path)
        if is_scanned:
            logger.info("Processing PDF as scanned document using OCR")
            ocr_text = pdf_to_images(file_path)
            if ocr_text and len(ocr_text.strip()) > 50:
                logger.info(f"Successfully extracted {len(ocr_text)} characters using OCR")
                return ocr_text
            elif text and len(text.strip()) > 0:
                logger.info(f"Using text extracted directly ({len(text)} chars)")
                return text
            else:
                logger.warning("Failed to extract meaningful text")
                return "This document appears to be a scanned PDF that could not be processed effectively."
        else:
            logger.info("Processing PDF as digitized document")
            if text and len(text.strip()) > 50:
                return text
            else:
                logger.warning("Minimal text extracted from digitized PDF")
                return "This document appears to be a PDF with minimal extractable text content."
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        logger.info("Processing image file using OCR")
        text = extract_text_from_image(file_path)
        if text:
            return text
        else:
            return "Unable to extract text from image."
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        return "Unsupported document type for classification."

# Use the classification module for document classification
def classify_document_via_frontend(file_path):
    """Classify document by triggering classification.py"""
    try:
        with st.spinner("Classifying document... (This may take a moment)"):
            import classification
            classification_result = classification.process_document_for_classification(file_path)
            return classification_result
    except Exception as e:
        st.warning(f"Error during document classification: {str(e)}")
        return "unknown"

# New helper: Display extracted results in a simple key-value format
def display_extraction_results(data):
    st.subheader("Extracted Data")
    # Check if the data follows the new structure
    if isinstance(data, dict) and "extracted_data" in data:
        extracted_data = data["extracted_data"]
        # Optionally, check if the document classification from the model differs from the one in session state
        if "document_classification" in data:
            doc_class = data["document_classification"]
            if doc_class.lower() != st.session_state.document_classification.lower():
                st.info(f"Note: The model classified this document as: **{doc_class.upper()}**")
        # Loop over each field and display the value
        if isinstance(extracted_data, dict):
            for key, value in extracted_data.items():
                st.write(f"**{key}:**")
                if value is None or (isinstance(value, str) and not value.strip()):
                    st.write("NULL")
                elif isinstance(value, dict):
                    st.json(value)
                elif isinstance(value, list):
                    if all(isinstance(item, dict) for item in value) and len(value) > 0:
                        st.dataframe(value)
                    else:
                        st.json(value)
                else:
                    st.write(value)
        else:
            st.write("No structured data found.")
    else:
        # Legacy format handling
        for key, value in data.items():
            st.write(f"**{key}:**")
            if value is None or (isinstance(value, str) and not value.strip()):
                st.write("NULL")
            elif isinstance(value, dict):
                st.json(value)
            elif isinstance(value, list):
                if all(isinstance(item, dict) for item in value) and len(value) > 0:
                    st.dataframe(value)
                else:
                    st.json(value)
            else:
                st.write(value)

current_step = steps[st.session_state.current_step_index]
st.title(current_step)

if current_step == "Configure Extraction Fields":
    st.header("Step 1: Configure Extraction Fields")
    with st.expander("Extraction Field Configuration", expanded=True):
        st.subheader("Load Field Configuration from CSV")
        uploaded_csv = st.file_uploader("Upload CSV for field configuration", type=["csv"], key="csv_config")
        if uploaded_csv:
            try:
                csv_string = io.StringIO(uploaded_csv.getvalue().decode("utf-8"))
                reader = csv.DictReader(csv_string)
                csv_fields = []
                for row in reader:
                    new_field = {
                        "name": row.get("Field Name", row.get("name", "")).strip(),
                        "type": row.get("Field Type", row.get("type", "")).strip(),
                        "description": row.get("Field Description", row.get("description", "")).strip()
                    }
                    if new_field["type"] == "List":
                        new_field["item_type"] = row.get("List Item Type", row.get("item_type", "")).strip()
                    csv_fields.append(new_field)
                st.session_state.extraction_fields = csv_fields
                st.success("Field configuration loaded from CSV successfully.")
            except Exception as e:
                st.error(f"Error loading CSV configuration: {e}")
        st.subheader("Add New Field")
        col1, col2 = st.columns(2)
        with col1:
            field_name = st.text_input("Field Name", key="field_name")
            field_type = st.selectbox("Field Type",
                                      ["String", "Number", "Date", "Boolean", "List", "Nested Object", "Table"],
                                      key="field_type")
        with col2:
            field_description = st.text_area("Description",
                                             placeholder="Describe what to extract (e.g., from text or tables)...",
                                             key="field_description")
        if field_type == "Nested Object":
            st.info("For nested objects, add child fields after creating this parent field.")
        elif field_type == "List":
            list_item_type = st.selectbox("List Item Type",
                                          ["String", "Number", "Date", "Boolean", "Object"],
                                          key="list_item_type")
        elif field_type == "Table":
            st.info("For table extraction, describe the table structure and content you want to extract.")
        if st.button("Add Field", key="add_field"):
            if field_name:
                new_field = {
                    "name": field_name,
                    "type": field_type,
                    "description": field_description
                }
                if field_type == "List":
                    new_field["item_type"] = list_item_type
                st.session_state.extraction_fields.append(new_field)
                st.success(f"Added field: {field_name}")
        st.subheader("Configured Fields")
        if not st.session_state.extraction_fields:
            st.info("No fields configured yet. Add fields above or load a CSV configuration.")
        else:
            for i, field in enumerate(st.session_state.extraction_fields):
                col1, col2, col3, col4 = st.columns([2, 1, 3, 1])
                col1.write(field["name"])
                col2.write(field["type"])
                description = field["description"]
                col3.write(description[:50] + "..." if len(description) > 50 else description)
                if col4.button("Delete", key=f"delete_{i}"):
                    st.session_state.extraction_fields.pop(i)
                    st.rerun()
        if st.session_state.extraction_fields:
            if st.button("Save Field Configuration"):
                config_file = os.path.join(tempfile.gettempdir(), "extraction_fields.json")
                with open(config_file, "w") as f:
                    json.dump(st.session_state.extraction_fields, f, indent=2)
                st.success(f"Configuration saved to {config_file}")
    if st.button("Next", key="next1"):
        go_next()

elif current_step == "Upload Document":
    st.header("Step 2: Upload and Process Document")
    st.subheader("Upload Document for Processing")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"], key="file_upload")
    if uploaded_file:
        if not st.session_state.uploaded_file_path or os.path.basename(st.session_state.uploaded_file_path) != uploaded_file.name:
            file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file_path = file_path
            with st.spinner("Classifying document..."):
                document_classification = classify_document_via_frontend(file_path)
                st.session_state.document_classification = document_classification
                st.session_state.document_type = document_classification
            st.success(f"New file uploaded: {uploaded_file.name}")
        else:
            file_path = st.session_state.uploaded_file_path
        if st.session_state.document_classification:
            st.info(f"📄 Document classified as: **{st.session_state.document_classification.upper()}**")
            classification_descriptions = {
                "invoice": "Invoice documents typically contain billing information, payment details, and itemized lists of products or services.",
                "personal": "Personal documents contain individual information like identification, resumes, or personal records.",
                "accounting": "Accounting documents contain financial data, statements, balance sheets, expense reports, etc.",
                "legal": "Legal documents contain contractual information, legal terms, and legal provisions or requirements.",
                "technical": "Technical documents contain specifications, procedures, or detailed technical information.",
                "regulatory": "Regulatory documents contain compliance information, policies, standards, or regulations.",
                "unknown": "The document type could not be determined with confidence."
            }
            doc_type = st.session_state.document_classification.lower()
            if doc_type in classification_descriptions:
                st.caption(classification_descriptions[doc_type])
        st.subheader("Extract Data from Document")
        if st.session_state.extraction_fields:
            if st.button("Process Document", disabled=st.session_state.processing_document):
                st.session_state.processing_document = True
                st.session_state.extracted_data = None
                output_dir = os.path.join(tempfile.gettempdir(), "extraction_output")
                os.makedirs(output_dir, exist_ok=True)
                config_file = os.path.join(tempfile.gettempdir(), "extraction_fields.json")
                with open(config_file, "w") as f:
                    json.dump(st.session_state.extraction_fields, f, indent=2)
                status = st.empty()
                with st.spinner("Processing document. This may take a few moments..."):
                    try:
                        unified_backend_script = "unified_backend.py"
                        if not os.path.exists(unified_backend_script):
                            unified_backend_script = os.path.join(os.path.dirname(__file__), "unified_backend.py")
                        cmd = [sys.executable, unified_backend_script, file_path, config_file, output_dir]
                        process = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                        if process.returncode != 0:
                            st.error(f"Error processing document: {process.stderr}")
                            status.error("Processing failed. See error details below.")
                            st.code(process.stderr)
                        else:
                            extracted_fields_path = os.path.join(output_dir, "extracted_data.json")
                            if not os.path.exists(extracted_fields_path):
                                alt_path = "results.json"
                                if os.path.exists(alt_path):
                                    extracted_fields_path = alt_path
                            if os.path.exists(extracted_fields_path):
                                with open(extracted_fields_path, 'r') as f:
                                    extracted_data = f.read()
                                try:
                                    parsed_data = json.loads(extracted_data)
                                    st.session_state.extracted_data = parsed_data
                                    st.session_state.extracted_text = True
                                    status.success("Document processed successfully!")
                                except json.JSONDecodeError:
                                    st.session_state.extracted_data = {}
                                    status.warning("Document processed but output is not valid JSON.")
                            else:
                                status.error("Processing completed but extracted data file was not found.")
                    except subprocess.TimeoutExpired:
                        status.error("Processing timed out. The document may be too large or complex.")
                    except Exception as e:
                        status.error(f"An error occurred: {str(e)}")
                st.session_state.processing_document = False
        else:
            st.warning("Please configure extraction fields before processing.")
    else:
        st.info("Upload a file to begin processing. The document will be automatically classified upon upload.")
    col1, col2 = st.columns(2)
    if col1.button("Previous", key="prev2"):
        go_previous()
    if col2.button("Next", key="next2"):
        go_next()

elif current_step == "View Results":
    st.header("Step 3: View Extraction Results")
    if st.session_state.document_classification:
        st.info(f"📄 Document Type: **{st.session_state.document_classification.upper()}**")
    if st.session_state.extracted_data:
        # Call the helper function to display results
        display_extraction_results(st.session_state.extracted_data)
    else:
        st.info("No extraction data available. Please process a document first.")
    if st.button("Previous", key="prev3"):
        go_previous()

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []
    optional_deps = []
    try:
        import openai
    except ImportError:
        missing_deps.append("openai")
    try:
        import PyPDF2
    except ImportError:
        missing_deps.append("PyPDF2")
    try:
        import pytesseract
    except ImportError:
        missing_deps.append("pytesseract")
    try:
        import PyMuPDF
    except ImportError:
        try:
            import fitz
        except ImportError:
            optional_deps.append("PyMuPDF or fitz")
    try:
        from pdf2image import convert_from_path
    except ImportError:
        optional_deps.append("pdf2image")
    try:
        from wand.image import Image
    except ImportError:
        if "pdf2image" in optional_deps:
            optional_deps.append("wand")
    return missing_deps, optional_deps

def main():
    st.set_page_config(page_title="Intelligent Document Processor", layout="wide")
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'document_type' not in st.session_state:
        st.session_state.document_type = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'navigation' not in st.session_state:
        st.session_state.navigation = "home"
    if 'current_step_index' not in st.session_state:
        st.session_state.current_step_index = 0
    missing_deps, optional_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing required dependencies: {', '.join(missing_deps)}. Please install them using `pip install {' '.join(missing_deps)}`")
        st.stop()
    if optional_deps:
        st.warning(f"Some optional dependencies are missing: {', '.join(optional_deps)}. The app will use alternative methods, but installing these may improve performance: `pip install {' '.join(optional_deps)}`")
    st.sidebar.title("Navigation")
    st.sidebar.write("Steps:")
    for i, step in enumerate(steps):
        if st.sidebar.button(step, key=f"nav_{i}"):
            st.session_state.current_step_index = i
            st.rerun()
if __name__ == "__main__":
    main()
