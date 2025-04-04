import os
import sys
import json
import tempfile
import logging
from datetime import datetime
import io
import re
from PIL import Image
import pytesseract
import requests
endpoint = endpoint
api_key = your_api_key

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, create_model



headers = {
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("unified_extractor")


if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    logger.info("Configured Windows unicode support")

OPENAI_API_KEY = "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"

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
        output_folder = os.path.join(tempfile.gettempdir(), "pdf_images_for_extraction")
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
    logger.info("Getting document text from: %s", file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        is_scanned, text = is_scanned_document(file_path)
        if is_scanned:
            logger.info("Processing PDF as scanned document using OCR")
            ocr_text = pdf_to_images(file_path)
            if ocr_text and len(ocr_text.strip()) > 50:
                logger.info("Successfully extracted %d characters using OCR", len(ocr_text))
                return ocr_text
            elif text and len(text.strip()) > 0:
                logger.info("Using text extracted directly (%d chars)", len(text))
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
        logger.error("Unsupported file type: %s", file_extension)
        return "Unsupported document type for processing."

def load_extraction_fields(config_path=None):
    """Load extraction field configuration from JSON or CSV file."""
    logger.info("Loading extraction fields from: %s", config_path)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding="utf-8") as f:
                fields = json.load(f)
                logger.info("Successfully loaded %d fields from config file", len(fields))
                return fields
        except json.JSONDecodeError:
            logger.error("Invalid JSON in config file %s", config_path)
            return []
    else:
        default_path = os.path.join(tempfile.gettempdir(), "extraction_fields.json")
        if os.path.exists(default_path):
            try:
                with open(default_path, 'r', encoding="utf-8") as f:
                    fields = json.load(f)
                    logger.info("Successfully loaded %d fields from default location", len(fields))
                    return fields
            except json.JSONDecodeError:
                logger.error("Error: Invalid JSON in default config file %s", default_path)
                return []
    logger.warning("No field configuration found")
    return []

def construct_unified_prompt(document_text, extraction_fields):
    """
    Construct a unified prompt for document processing that handles classification,
    text, and table extraction.
    """
    field_descriptions = []
    for field in extraction_fields:
        field_name = field["name"]
        field_type = field["type"]
        field_description = field["description"]
        type_annotation = f"(type: {field_type})"
        field_descriptions.append(f"- {field_name} {type_annotation}: {field_description}")
    fields_text = "\n".join(field_descriptions)
    prompt = f"""
You are an advanced document analysis system. You need to:

1. CLASSIFY the document (is it an invoice, personal document, legal document, etc.)?
2. PROCESS the entire document text, including any tables if present.
3. EXTRACT exactly the information requested in the fields list below, in the correct types (string, table, etc.).

For Every document , first study the data to draw some summary from the data as to what the document means and what its trying to say.
Find answers to the relevant questions regarding any document:
1) What does this document mean?
2) what does the data say?
3) What is the deal made/ agreement reached/ questions answered in this document?
..
After you have the answers to all the relevant questions with regards to the document, make a summary out of them,
and keeping the answers and the summary in mind answer the queries/ extract the fields.
FIELDS TO EXTRACT:
{fields_text}

For each field:
- Return data in the correct format.
- If no info is found, return null.
- For table fields, first IDENTIFY and EXTRACT the column names. These are the names found right before the beginning of the recurring patterns of the table.
- Store the column names in a list and use this knowledge to ensure columns are not clubbed together incorrectly.
- Return the column names along with the table data.
- For table data, return a JSON array of objects, where each object represents a row with column names as keys. Example:
  [
    {{"column1": "value1", "column2": "value2"}},
    {{"column1": "value3", "column2": "value4"}}
  ]

IMPORTANT INSTRUCTIONS FOR TABLES:
- If the document contains MULTIPLE tables, separate them into different table objects named "Table1", "Table2", etc.
- Do NOT combine different tables into a single table. Each distinct table in the document should become a separate field.
- CAREFULLY identify column headers for each table. Look for visual or structural clues that indicate column names.
- ENSURE each column is correctly identified as a separate entity - do not club multiple columns together.
- MAINTAIN CONSISTENCY in column names across all rows in a single table.
- For each row in a table, create an object with properly separated key-value pairs.
- Double-check that column data is correctly aligned with column headers.
- For tables where column headers aren't explicit, infer reasonable column names based on data content and structure.
- Pay special attention to spacing and formatting in the document to correctly identify column boundaries.
- Example for multiple tables with properly separated columns:
SPECIAL NOTE:
-Tables may also be single lined. To detect such tables, look for entities present right under what looks like something defining its entity type.
-For example if something that is an address is present under the word address, something resembling a back order is present under the word back order, and there is a consistent pattern following this and having equal spacings between the words, then this is most likely a single lined tables.
-This should be extracted when the user asks for tables.
"Table1": [
  {{"ItemNumber": "1001", "Description": "Office Chair", "Quantity": "5", "UnitPrice": "150.00", "Total": "750.00"}},
  {{"ItemNumber": "1002", "Description": "Desk Lamp", "Quantity": "10", "UnitPrice": "25.00", "Total": "250.00"}}
],
"Table2": [
  {{"InvoiceDate": "2023-05-15", "DueDate": "2023-06-15", "Terms": "Net 30"}},
  {{"ShippingMethod": "Ground", "TrackingNumber": "1Z999AA10123456784"}}
]
Only return the Tables when explicitly asked for the tabular data in the queries. If the queries dont specifically ask for tables dont send back the tables. If its data related to the tables send back only the data related to the tables.
If specifically asked for tables, only then send back the table data.
DOCUMENT TEXT:
{document_text}

Return your final answer as valid JSON:
{{
  "document_classification": "the document type",
  "extracted_data": {{
     "field1": value1,
     "field2": value2,
     ...
     "Table1": {{"columns": ["column1", "column2", ...], "data": [...]}},
     "Table2": {{"columns": ["column3", "column4", ...], "data": [...]}},
     ...
  }}
}}
No markdown, no extra keys.
"""
    return prompt.strip()

# Modified create_dynamic_model to allow null values using Optional
from typing import Optional, Union, List, Dict, Any
def create_dynamic_model(extraction_fields):
    model_fields = {}
    for field_def in extraction_fields:
        field_name = field_def["name"]
        field_type = field_def["type"].lower()
        if field_type == "string":
            model_fields[field_name] = (Optional[str], None)
        elif field_type == "integer":
            model_fields[field_name] = (Optional[int], None)
        elif field_type == "list":
            model_fields[field_name] = (Optional[list], None)
        elif field_type == "table":
            # Allow either a string, a list of dicts, or null.
            model_fields[field_name] = (Optional[Union[str, List[Dict[str, Any]]]], None)
        else:
            model_fields[field_name] = (Optional[str], None)
    return create_model("DynamicExtractionModel", **model_fields)

def process_document(file_path, extraction_fields_config=None, output_dir=None):
    """Process a document to extract requested information using a unified approach."""
    logger.info(f"Starting unified document processing for: {file_path}")

    extraction_fields = load_extraction_fields(extraction_fields_config)
    if not extraction_fields:
        logger.error("No extraction fields configured, cannot proceed")
        return {"error": "No extraction fields configured"}

    if output_dir is None:
        output_dir = tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    document_text = get_document_text(file_path)
    if not document_text or len(document_text.strip()) < 50:
        logger.warning("Minimal text extracted from document")

    prompt = construct_unified_prompt(document_text, extraction_fields)

    logger.info("Sending unified prompt to Azure OpenAI API")
    try:
        # Format for Azure OpenAI API
        messages = [
            {"role": "user", "content": prompt}
        ]

        data = {
            "messages": messages,
            "temperature": 0
        }

        response = requests.post(endpoint, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"Error from Azure OpenAI API: {response.status_code} - {response.text}")
            raise Exception(f"Azure OpenAI API returned status code {response.status_code}: {response.text}")

        content = response.json()['choices'][0]['message']['content']
        logger.debug(f"API response content: {content[:200]}...")

        cleaned_content = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)

        try:
            result = json.loads(cleaned_content)
            logger.info("Successfully parsed JSON response")

            # Process all table fields in the extracted data
            if result.get("extracted_data"):
                extracted_data = result["extracted_data"]
                # Check for any field that might be a table
                for field_name, field_value in list(extracted_data.items()):
                    # Process fields that start with "Table" or contain table data
                    if (isinstance(field_value, str) and
                        (field_name.startswith("Table") or field_name == "Tables")):
                        try:
                            table_data = parse_table_string_to_objects(field_value)
                            extracted_data[field_name] = table_data
                            logger.info(f"Successfully structured {field_name} data as objects")
                        except Exception as e:
                            logger.error(f"Failed to structure {field_name} data: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            result = {
                "document_classification": "unknown",
                "extracted_data": {},
                "error": "Failed to parse response from LLM",
                "raw_response": cleaned_content
            }

        output_file = os.path.join(output_dir, "extracted_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_file}")

        fixed_output = "results.json"
        with open(fixed_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results also saved to {fixed_output}")

        return result

    except Exception as e:
        logger.error(f"Error in API call: {e}")
        error_result = {
            "document_classification": "unknown",
            "extracted_data": {},
            "error": f"API call failed: {str(e)}"
        }

        output_file = os.path.join(output_dir, "extracted_data.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)

        fixed_output = "results.json"
        with open(fixed_output, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)

        return error_result

def parse_table_string_to_objects(table_string):
    """
    Parse a table string into a list of JSON objects where each object represents a row.
    Each object has column names as keys and row values as values.
    """
    logger.info("Parsing table string to structured objects")

    # Try to detect if it's already a JSON structure
    try:
        # Check if the string is already valid JSON
        if table_string.strip().startswith("[") and table_string.strip().endswith("]"):
            parsed = json.loads(table_string)
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                logger.info("Table data already in correct JSON format")
                return parsed
    except:
        pass

    # If we reach here, need to parse the string into structured data
    try:
        # First, try to split by lines
        rows = table_string.strip().split('\n')

        # If only one line, try to split by other patterns
        if len(rows) <= 1:
            # Check for tab-separated data
            if '\t' in table_string:
                rows = table_string.strip().split('\t')
            # Otherwise look for spaces as separators
            else:
                # Split the string based on spaces but keep numbers with decimals intact
                pattern = r'(\d+\.\d+|\d+|\w+)'
                tokens = re.findall(pattern, table_string)

                # Try to infer table structure based on the pattern of data
                # This is a simplified approach - for real-world data this would need to be more sophisticated

                # For this example, let's use a simple approach to create columns
                # Assuming a structure like: Quantity Item Price Total
                if len(tokens) >= 4:
                    column_names = ["Quantity", "Item", "UnitPrice", "Total"]
                    result = []

                    # Group tokens into rows (assuming 4 values per row)
                    for i in range(0, len(tokens), 4):
                        if i + 3 < len(tokens):
                            row_obj = {
                                column_names[0]: tokens[i],
                                column_names[1]: tokens[i+1],
                                column_names[2]: tokens[i+2],
                                column_names[3]: tokens[i+3]
                            }
                            result.append(row_obj)

                    logger.info(f"Created {len(result)} structured table rows from tokens")
                    return result

        # If we reach here, attempt to handle multiple rows
        # Try to identify column headers from the first row, or create generic ones
        column_headers = []

        # If there appears to be a header row
        if len(rows) > 1:
            # Use heuristics to determine if first row is a header
            # For this example, we'll use simple column names if we can't determine headers
            column_headers = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

        result = []
        for i, row in enumerate(rows):
            if not row.strip():
                continue

            # Split row by spaces or tabs
            values = re.findall(r'(\d+\.\d+|\d+|\w+)', row)

            if len(values) > 0:
                row_obj = {}
                for j, value in enumerate(values):
                    if j < len(column_headers):
                        row_obj[column_headers[j]] = value
                    else:
                        row_obj[f"Col{j+1}"] = value

                result.append(row_obj)

        logger.info(f"Parsed table string into {len(result)} structured rows")
        return result

    except Exception as e:
        logger.error(f"Error parsing table string: {e}")
        # If parsing fails, return the original string wrapped in a single-item list
        return [{"raw_data": table_string}]

def main():
    """Main function to process documents from the command line."""
    logger.info("Unified document extraction process started")
    if len(sys.argv) < 2:
        logger.error("Insufficient command line arguments")
        print("Usage: python unified_backend.py <path_to_document> [path_to_fields_config] [output_directory]")
        sys.exit(1)
    document_path = sys.argv[1]
    fields_config = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    if not os.path.exists(document_path):
        logger.error("Document not found: %s", document_path)
        print(f"Document not found: {document_path}")
        sys.exit(1)
    result = process_document(document_path, fields_config, output_dir)
    if result and "error" not in result:
        print("Document processed successfully")
        print("Extraction complete. Results saved to output files.")
    else:
        error_msg = result.get("error", "Unknown error")
        print(f"Document processing failed: {error_msg}")
        sys.exit(1)
    logger.info("Unified document extraction process completed")

if __name__ == "__main__":
    main()
