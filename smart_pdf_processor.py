#!/usr/bin/env python3
"""
🔀 Smart PDF Processor - Hybrid Approach
Extract with PyMuPDF, clean with OpenAI for optimal cost/quality balance
"""
import fitz  # PyMuPDF
from pathlib import Path
import json
import time

class SmartPDFProcessor:
    """Hybrid PDF processor: PyMuPDF extraction + OpenAI cleaning"""
    
    def __init__(self):
        self.cache_file = "smart_pdf_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Load cached results"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except:
            self.cache = {}
    
    def save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def get_pdf_hash(self, pdf_path: str) -> str:
        """Generate hash for PDF file"""
        import hashlib
        with open(pdf_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def extract_raw_text(self, pdf_path: str) -> str:
        """Extract raw text using PyMuPDF"""
        print(f"📄 Extracting raw text from {Path(pdf_path).name}...")
        
        doc = fitz.open(pdf_path)
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                # Clean up obvious formatting issues
                text = text.replace('\n\n\n', '\n\n')  # Remove excessive newlines
                text = text.replace('  ', ' ')  # Remove double spaces
                all_text.append(text)
                print(f"   Page {page_num + 1}: {len(text)} characters")
        
        doc.close()
        raw_text = '\n\n'.join(all_text)
        print(f"✅ Extracted {len(raw_text)} characters total")
        return raw_text
    
    def clean_text_with_openai(self, raw_text: str) -> str:
        """Clean and format text using OpenAI"""
        try:
            from openai import OpenAI
            import os
            
            # Load API key
            api_key = None
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            
            if not api_key:
                print("❌ No OpenAI API key found")
                return raw_text
            
            client = OpenAI(api_key=api_key)
            
            # Split text into chunks for processing
            chunk_size = 4000  # Leave room for prompt
            chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
            
            cleaned_chunks = []
            total_cost = 0
            
            for i, chunk in enumerate(chunks):
                print(f"🧹 Cleaning chunk {i+1}/{len(chunks)}...")
                
                prompt = f"""Clean and format this extracted PDF text to make it more readable.

INSTRUCTIONS:
- Fix broken words and sentences
- Remove excessive whitespace and formatting artifacts
- Keep all technical content, definitions, and important information
- Maintain paragraph structure and flow
- Fix hyphenated words that were broken across lines
- Keep academic/technical terminology exactly as intended
- Do NOT summarize or remove content - just clean formatting

TEXT TO CLEAN:
{chunk}"""

                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=0
                    )
                    
                    cleaned_text = response.choices[0].message.content
                    cleaned_chunks.append(cleaned_text)
                    
                    # Estimate cost (rough)
                    input_tokens = len(prompt) / 4
                    output_tokens = len(cleaned_text) / 4
                    chunk_cost = (input_tokens * 0.0015 + output_tokens * 0.002) / 1000
                    total_cost += chunk_cost
                    
                    print(f"   ✅ Cleaned {len(cleaned_text)} characters (~${chunk_cost:.3f})")
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    print(f"   ❌ Error cleaning chunk {i+1}: {e}")
                    cleaned_chunks.append(chunk)  # Use original if cleaning fails
            
            print(f"💰 Total estimated cost: ~${total_cost:.3f}")
            return '\n\n'.join(cleaned_chunks)
            
        except Exception as e:
            print(f"❌ Error in OpenAI cleaning: {e}")
            return raw_text  # Return original if cleaning fails
    
    def process_pdf(self, pdf_path: str, force_reprocess: bool = False) -> str:
        """Process PDF with hybrid approach"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Check cache
        pdf_hash = self.get_pdf_hash(str(pdf_path))
        cache_key = f"{pdf_path.stem}_{pdf_hash}"
        
        if not force_reprocess and cache_key in self.cache:
            print(f"💾 Using cached result for {pdf_path.name}")
            return self.cache[cache_key]['cleaned_text']
        
        print(f"🔀 Processing {pdf_path.name} with hybrid approach...")
        
        # Step 1: Extract raw text with PyMuPDF
        raw_text = self.extract_raw_text(str(pdf_path))
        
        # Step 2: Clean with OpenAI
        cleaned_text = self.clean_text_with_openai(raw_text)
        
        # Step 3: Save results
        result = {
            'pdf_name': pdf_path.name,
            'pdf_hash': pdf_hash,
            'raw_length': len(raw_text),
            'cleaned_length': len(cleaned_text),
            'processed_at': time.time(),
            'raw_text': raw_text,
            'cleaned_text': cleaned_text
        }
        
        self.cache[cache_key] = result
        self.save_cache()
        
        # Save to extracted_texts
        output_file = Path("extracted_texts") / f"{pdf_path.stem}_SMART.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"✅ Saved cleaned text to {output_file}")
        print(f"📊 Raw: {len(raw_text)} → Cleaned: {len(cleaned_text)} characters")
        
        return cleaned_text

def main():
    """Test the smart processor"""
    processor = SmartPDFProcessor()
    
    pdf_file = "Attention_is_all_you_need_v7.pdf"
    if Path(pdf_file).exists():
        try:
            cleaned_text = processor.process_pdf(pdf_file)
            print(f"\n🎉 Success! Processed {len(cleaned_text)} characters")
            print(f"📄 First 200 characters:")
            print(cleaned_text[:200] + "...")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ PDF file not found: {pdf_file}")

if __name__ == "__main__":
    main() 