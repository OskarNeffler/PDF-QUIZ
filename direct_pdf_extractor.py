#!/usr/bin/env python3
"""
📄 Direct PDF Text Extractor
Extract text directly from PDF without AI interpretation
"""
import fitz  # PyMuPDF
from pathlib import Path

def extract_pdf_text_direct(pdf_path: str) -> str:
    """Extract text directly from PDF using PyMuPDF"""
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        all_text = []
        
        print(f"📖 Extracting text from {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text from page
            text = page.get_text()
            
            if text.strip():  # Only add non-empty pages
                all_text.append(f"--- Page {page_num + 1} ---\n{text}")
                print(f"   Page {page_num + 1}: {len(text)} characters")
        
        doc.close()
        
        combined_text = "\n\n".join(all_text)
        print(f"✅ Total extracted: {len(combined_text)} characters")
        
        return combined_text
        
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return ""

def save_extracted_text(text: str, filename: str):
    """Save extracted text to file"""
    try:
        # Ensure extracted_texts directory exists
        Path("extracted_texts").mkdir(exist_ok=True)
        
        filepath = Path("extracted_texts") / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"💾 Saved to: {filepath}")
        print(f"📊 File size: {len(text)} characters")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    # Extract text from Attention PDF
    pdf_file = "Attention_is_all_you_need_v7.pdf"
    
    if Path(pdf_file).exists():
        print(f"🚀 Processing {pdf_file}...")
        
        # Extract text directly
        extracted_text = extract_pdf_text_direct(pdf_file)
        
        if extracted_text:
            # Save to file
            output_filename = "Attention_is_all_you_need_v7_DIRECT.txt"
            save_extracted_text(extracted_text, output_filename)
            
            print(f"\n📈 Comparison:")
            print(f"   Direct extraction: {len(extracted_text)} characters")
            
            # Compare with OpenAI version if it exists
            openai_file = Path("extracted_texts/Attention_is_all_you_need_v7.txt")
            if openai_file.exists():
                with open(openai_file, 'r', encoding='utf-8') as f:
                    openai_text = f.read()
                print(f"   OpenAI extraction: {len(openai_text)} characters")
                print(f"   📊 Direct is {len(extracted_text) / len(openai_text):.1f}x longer!")
        else:
            print("❌ No text extracted")
    else:
        print(f"❌ PDF file not found: {pdf_file}") 