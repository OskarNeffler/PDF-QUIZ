import PyPDF2
from typing import List, Dict, Any
import re
import os

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file while preserving structure.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        A list of dictionaries, each representing a section of the document with:
        - 'title': Section title if detected
        - 'text': Text content
        - 'page': Page number
        - 'paragraphs': List of paragraphs in the section
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    sections = []
    current_section = {
        'title': None,
        'text': '',
        'page': 1,
        'paragraphs': []
    }
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Process the extracted text to preserve structure
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                for para in paragraphs:
                    # Check if paragraph is a heading (simple heuristic)
                    is_heading = False
                    if len(para) < 100 and (para.isupper() or re.match(r'^[0-9]+\.\s+', para)):
                        is_heading = True
                    
                    if is_heading:
                        # Save previous section if not empty
                        if current_section['paragraphs']:
                            current_section['text'] = '\n\n'.join(current_section['paragraphs'])
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            'title': para,
                            'text': '',
                            'page': page_num + 1,
                            'paragraphs': []
                        }
                    else:
                        current_section['paragraphs'].append(para)
            
            # Add the last section
            if current_section['paragraphs']:
                current_section['text'] = '\n\n'.join(current_section['paragraphs'])
                sections.append(current_section)
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    return sections


def get_document_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing metadata like title, author, etc.
    """
    metadata = {}
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata_dict = pdf_reader.metadata
            
            if metadata_dict:
                # Convert to regular dictionary and clean up keys
                for key, value in metadata_dict.items():
                    if key.startswith('/'):
                        clean_key = key[1:].lower()
                        metadata[clean_key] = value
            
            # Add basic document info
            metadata['page_count'] = len(pdf_reader.pages)
            
    except Exception as e:
        print(f"Warning: Could not extract metadata: {str(e)}")
    
    return metadata 