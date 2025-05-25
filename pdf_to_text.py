#!/usr/bin/env python3
"""
📄 Simple PDF to Text Converter
Extract clean text from PDF files using OpenAI GPT-4 Vision
Includes cache system to avoid reprocessing same files
"""
import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add openai system to path
sys.path.append(str(Path(__file__).parent / "openai_system"))

class PDFCache:
    """Manages cache of processed PDFs"""
    
    def __init__(self, cache_file: str = "pdf_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Warning: Could not save cache: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for unique identification"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return None
    
    def is_processed(self, pdf_path: str) -> bool:
        """Check if PDF has been processed"""
        file_hash = self._get_file_hash(pdf_path)
        if not file_hash:
            return False
        
        # Check if hash exists in cache
        return file_hash in self.cache
    
    def get_processed_info(self, pdf_path: str) -> dict:
        """Get info about processed PDF"""
        file_hash = self._get_file_hash(pdf_path)
        if file_hash and file_hash in self.cache:
            return self.cache[file_hash]
        return None
    
    def mark_processed(self, pdf_path: str, output_path: str = None, text_length: int = 0):
        """Mark PDF as processed"""
        file_hash = self._get_file_hash(pdf_path)
        if not file_hash:
            return
        
        self.cache[file_hash] = {
            'filename': Path(pdf_path).name,
            'processed_date': datetime.now().isoformat(),
            'output_path': output_path,
            'text_length': text_length,
            'full_path': str(Path(pdf_path).absolute())
        }
        
        self._save_cache()
    
    def clear_cache(self):
        """Clear all cache"""
        self.cache = {}
        self._save_cache()
        print("✅ Cache cleared")
    
    def show_cache_stats(self):
        """Show cache statistics"""
        if not self.cache:
            print("📊 Cache is empty")
            return
        
        print(f"📊 Cache Statistics:")
        print(f"   - Processed files: {len(self.cache)}")
        print(f"   - Cache file: {self.cache_file}")
        
        print(f"\n📋 Recently processed:")
        sorted_files = sorted(self.cache.values(), 
                            key=lambda x: x['processed_date'], 
                            reverse=True)
        
        for file_info in sorted_files[:5]:  # Show last 5
            date = datetime.fromisoformat(file_info['processed_date']).strftime('%Y-%m-%d %H:%M')
            print(f"   - {file_info['filename']} ({date})")

# Global cache instance
pdf_cache = PDFCache()

def convert_pdf_to_text(pdf_path: str, output_path: str = None, force: bool = False) -> str:
    """Convert a single PDF to text"""
    try:
        from openai_pdf_extractor import OpenAIPDFExtractor
        
        pdf_name = Path(pdf_path).name
        pdf_stem = Path(pdf_path).stem
        
        # Create extracted_texts directory if it doesn't exist
        extracted_texts_dir = Path("extracted_texts")
        extracted_texts_dir.mkdir(exist_ok=True)
        
        # Default output path in extracted_texts directory
        default_output_path = extracted_texts_dir / f"{pdf_stem}.txt"
        final_output_path = output_path if output_path else default_output_path
        
        # Check cache first
        if not force and pdf_cache.is_processed(pdf_path):
            cached_info = pdf_cache.get_processed_info(pdf_path)
            cached_date = datetime.fromisoformat(cached_info['processed_date']).strftime('%Y-%m-%d %H:%M')
            
            print(f"📋 {pdf_name} already processed ({cached_date})")
            print(f"   Text length: {cached_info['text_length']} characters")
            
            # Try to load existing text from cached path or default path
            text_paths_to_try = [
                cached_info.get('output_path'),
                str(default_output_path),
                output_path
            ]
            
            for text_path in text_paths_to_try:
                if text_path and Path(text_path).exists():
                    try:
                        with open(text_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        print(f"📖 Loaded text from: {text_path}")
                        return text
                    except Exception:
                        continue
            
            # Ask user if they want to skip or reprocess
            if len(sys.argv) == 1:  # Interactive mode
                choice = input(f"🔄 Text file not found. Reprocess? (y/N): ").strip().lower()
                if choice != 'y':
                    return None
            else:
                print("   Text file not found. Reprocessing...")
        
        print(f"📖 Converting {pdf_name} to text...")
        
        # Create extractor
        extractor = OpenAIPDFExtractor()
        if not extractor.setup_success:
            return None
        
        # Extract text
        text = extractor.extract_text(pdf_path)
        
        if text:
            # Always save to the final output path
            with open(final_output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"💾 Saved text to: {final_output_path}")
            
            # Mark as processed in cache with the output path
            pdf_cache.mark_processed(pdf_path, str(final_output_path), len(text))
            
            return text
        else:
            print("❌ Failed to extract text")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def batch_convert_pdfs(input_dir: str, output_dir: str = None, force: bool = False):
    """Convert all PDFs in a directory to text files"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Directory not found: {input_dir}")
        return
    
    # Find all PDFs
    pdf_files = list(input_path.glob('*.pdf'))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {input_dir}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    # Check which files are already processed
    if not force:
        already_processed = [pdf for pdf in pdf_files if pdf_cache.is_processed(str(pdf))]
        if already_processed:
            print(f"📋 {len(already_processed)} files already processed:")
            for pdf in already_processed:
                cached_info = pdf_cache.get_processed_info(str(pdf))
                date = datetime.fromisoformat(cached_info['processed_date']).strftime('%Y-%m-%d')
                print(f"   - {pdf.name} ({date})")
            
            if len(sys.argv) == 1:  # Interactive mode
                choice = input(f"\n🔄 Process all files anyway? (y/N): ").strip().lower()
                force = (choice == 'y')
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = input_path / "extracted_text"
        output_path.mkdir(exist_ok=True)
    
    # Convert each PDF
    processed_count = 0
    skipped_count = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 [{i}/{len(pdf_files)}] Processing {pdf_file.name}...")
        
        # Create output filename
        txt_filename = pdf_file.stem + ".txt"
        txt_path = output_path / txt_filename
        
        # Convert
        text = convert_pdf_to_text(str(pdf_file), str(txt_path), force)
        
        if text:
            print(f"✅ Converted: {len(text)} characters")
            processed_count += 1
        elif text is None and pdf_cache.is_processed(str(pdf_file)):
            print("⏭️  Skipped (already processed)")
            skipped_count += 1
        else:
            print("❌ Failed")
    
    print(f"\n✅ Batch conversion completed!")
    print(f"📊 Processed: {processed_count}, Skipped: {skipped_count}")
    print(f"📁 Text files saved in: {output_path}")

def interactive_mode():
    """Interactive PDF to text conversion"""
    print("🔄 Interactive PDF to Text Converter")
    print("=" * 40)
    
    while True:
        print("\n📋 Options:")
        print("1. Convert single PDF")
        print("2. Convert all PDFs in directory")
        print("3. Show cache statistics")
        print("4. Clear cache")
        print("5. Load text for model")
        print("6. Process texts with model")
        print("7. Exit")
        
        choice = input("\nChoose option (1-7): ").strip()
        
        if choice == "1":
            pdf_path = input("📄 Enter PDF path: ").strip()
            if pdf_path and Path(pdf_path).exists():
                text = convert_pdf_to_text(pdf_path)
                if text:
                    print(f"\n📝 Extracted text ({len(text)} characters):")
                    print("-" * 50)
                    print(text[:500] + "..." if len(text) > 500 else text)
                    print("-" * 50)
                    
                    save = input("\n💾 Save to file? (y/n): ").strip().lower()
                    if save == 'y':
                        output_path = input("📁 Output filename (or press Enter for auto): ").strip()
                        if not output_path:
                            output_path = Path(pdf_path).stem + ".txt"
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        print(f"✅ Saved to: {output_path}")
            else:
                print("❌ PDF file not found")
        
        elif choice == "2":
            dir_path = input("📁 Enter directory path: ").strip()
            if dir_path and Path(dir_path).exists():
                batch_convert_pdfs(dir_path)
            else:
                print("❌ Directory not found")
        
        elif choice == "3":
            pdf_cache.show_cache_stats()
        
        elif choice == "4":
            confirm = input("🗑️  Really clear all cache? (y/N): ").strip().lower()
            if confirm == 'y':
                pdf_cache.clear_cache()
        
        elif choice == "5":
            text_file_path = input("📄 Enter text file path: ").strip()
            if text_file_path and Path(text_file_path).exists():
                text = load_text_for_model(text_file_path)
                if text:
                    print(f"\n📝 Loaded text ({len(text)} characters):")
                    print("-" * 50)
                    print(text[:500] + "..." if len(text) > 500 else text)
                    print("-" * 50)
            else:
                print("❌ Text file not found")
        
        elif choice == "6":
            process_texts_with_model()
        
        elif choice == "7":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

def load_text_for_model(text_file_path: str) -> str:
    """Load existing text file for model processing"""
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📖 Loaded {len(text)} characters from {text_file_path}")
        return text
    except Exception as e:
        print(f"❌ Error loading text: {e}")
        return None

def process_texts_with_model():
    """Process all extracted texts with the model"""
    extracted_texts_dir = Path("extracted_texts")
    
    if not extracted_texts_dir.exists():
        print("❌ No extracted_texts directory found")
        print("   Run some PDF extractions first!")
        return
    
    text_files = list(extracted_texts_dir.glob("*.txt"))
    
    if not text_files:
        print("❌ No text files found in extracted_texts/")
        print("   Run some PDF extractions first!")
        return
    
    print(f"📚 Found {len(text_files)} text files:")
    for i, text_file in enumerate(text_files, 1):
        file_size = text_file.stat().st_size
        print(f"   {i}. {text_file.name} ({file_size} bytes)")
    
    print(f"\n🤖 Ready to process with your model!")
    print(f"   (Model integration coming next...)")
    
    # TODO: Here we will integrate with your trained model
    # For now, just show what texts are available
    return text_files

def main():
    # Check if API key is configured
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY found in .env")
        print("\n Setup required:")
        print("1. Add OPENAI_API_KEY to .env file")
        print("2. See setup_openai.txt for details")
        return
    
    # Check for force flag
    force = '--force' in sys.argv
    if force:
        sys.argv.remove('--force')
    
    # Check command line arguments
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        interactive_mode()
    
    elif len(sys.argv) == 2:
        # Single PDF conversion
        pdf_path = sys.argv[1]
        if Path(pdf_path).exists():
            text = convert_pdf_to_text(pdf_path, force=force)
            if text:
                print(f"\n📝 Extracted text:")
                print(text)
        else:
            print(f"❌ File not found: {pdf_path}")
    
    elif len(sys.argv) == 3:
        # PDF with output file
        pdf_path, output_path = sys.argv[1], sys.argv[2]
        if Path(pdf_path).exists():
            convert_pdf_to_text(pdf_path, output_path, force=force)
        else:
            print(f"❌ File not found: {pdf_path}")
    
    else:
        print("📄 PDF to Text Converter with Cache")
        print("\nUsage:")
        print("  python pdf_to_text.py                      # Interactive mode")
        print("  python pdf_to_text.py file.pdf             # Print text to console")
        print("  python pdf_to_text.py file.pdf output.txt  # Save to file")
        print("  python pdf_to_text.py --force file.pdf     # Force reprocess")

if __name__ == "__main__":
    main() 